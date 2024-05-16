
module Spectral

export Spectrum, Monochromatic, SpectralDist, InterpolatedSpectralDist
export make_cherenkov_spectral_dist, make_biased_cherenkov_spectral_dist
export make_cuda_spectral_dist
export make_cherenkov_spectrum, make_biased_cherenkov_spectrum, make_monochromatic_spectrum

using Distributions
using Adapt
using Interpolations
using Unitful
using Random
using StaticArrays
using LinearAlgebra
using CUDA
using PhysicsTools
using StructTypes
using ..Medium

abstract type SpectralDist <: Sampleable{Univariate,Continuous} end
Base.rand(::AbstractRNG, d::SpectralDist) = error("not implemented for type $(typeof(d))")
Base.rand(d::SpectralDist) = rand(Random.default_rng(), d)

struct InterpolatedSpectralDist{A, T} <: SpectralDist
    interpolated_cdf::A
    normalization::T
end

StructTypes.StructType(::Type{<:InterpolatedSpectralDist}) = StructTypes.Struct()

Adapt.@adapt_structure InterpolatedSpectralDist

Base.rand(rng::AbstractRNG, d::InterpolatedSpectralDist) = d.interpolated_cdf(rand(rng))
Base.rand(rng::AbstractRNG, d::InterpolatedSpectralDist{<:CuTexture}) = @inbounds d.interpolated_cdf[rand(rng)]
Base.rand(rng::AbstractRNG, d::InterpolatedSpectralDist{<:CuDeviceTexture}) = @inbounds d.interpolated_cdf[rand(rng)]



function make_spectral_dist(spect_func, wl_range::Tuple{T, T}, step_size::T=T(1)) where {T <: Real}
    wl_steps = wl_range[1]:step_size:wl_range[2]

    norms = Vector{T}(undef, size(wl_steps, 1))
    norms[1] = 0

    full_norm = integrate_gauss_quad(spect_func, wl_range[1], wl_range[2], 25)

    for i in eachindex(wl_steps)[2:end]
        step = wl_steps[i]
        norms[i] = integrate_gauss_quad(spect_func, wl_range[1], step, 25) / full_norm
    end

    sorting = sortperm(norms)
    return InterpolatedSpectralDist(linear_interpolation(norms[sorting], wl_steps[sorting], extrapolation_bc=zero(T)), full_norm)

end



struct Monochromatic{T} <: SpectralDist
    wavelength::T
end

Base.rand(::AbstractRNG, d::Monochromatic) = d.wavelength
StructTypes.StructType(::Type{<:Monochromatic}) = StructTypes.Struct()
Adapt.@adapt_structure Monochromatic


function make_cuda_spectral_dist(spec::SpectralDist, wl_range::Tuple{T, T}, interp_steps::Integer=30) where {T<:Real}

    eval_knots = range(zero(T), one(T), interp_steps)
    knots = spec.interpolated_cdf(eval_knots)
    spectrum_vals = CuTextureArray(knots)
    spectrum_texture = CuTexture(spectrum_vals; interpolation=CUDA.LinearInterpolation(), normalized_coordinates=true)

    return InterpolatedSpectralDist(spectrum_texture, spec.normalization)
end

struct Spectrum{D <: SpectralDist, S, T<:Real}
    spectral_dist::D
    spectrum::S
    wl_range::Tuple{T, T}
end

function make_cherenkov_spectrum(wl_range, medium)
    sfunc = wl -> frank_tamm(wl, phase_refractive_index(wl, medium)) * 1E9
    d = make_spectral_dist(sfunc, wl_range)
    return Spectrum(d, sfunc, wl_range)
end

function make_biased_cherenkov_spectrum(bias_function, wl_range, medium)
    sfunc = wl -> frank_tamm(wl, phase_refractive_index(wl, medium)) * bias_function(wl)  * 1E9
    d = make_spectral_dist(sfunc, wl_range, eltype(wl_range)(0.5))
    return Spectrum(d, sfunc, wl_range)
end

function make_monochromatic_spectrum(wl)
    return Spectrum(Monochromatic(wl), w -> w == wl ? 1 : 0, (zero(wl), one(wl)))
end

end

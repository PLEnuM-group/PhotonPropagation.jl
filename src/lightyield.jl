module LightYield

export LongitudinalParametersBase
export LongitudinalParameterisation
export get_longitudinal_params
export CherenkovTrackLengthParameters
export longitudinal_profile, cascade_cherenkov_track_length, fractional_contrib_long
export particle_to_lightsource, particle_to_elongated_lightsource, particle_to_elongated_lightsource!
export total_lightyield
export rel_additional_track_length

export AngularEmissionProfile
export PhotonSource, PointlikeIsotropicEmitter, ExtendedCherenkovEmitter, CherenkovEmitter, PointlikeCherenkovEmitter
export AxiconeEmitter, PencilEmitter, PointlikeTimeRangeEmitter, CherenkovTrackEmitter, LightsabreMuonEmitter
export cherenkov_ang_dist, cherenkov_ang_dist_int
export split_source, oversample_source

import Base: @kwdef
using SpecialFunctions: gamma
using StaticArrays
using PhysicalConstants.CODATA2018
using Unitful
using PoissonRandom
using Interpolations
using PoissonRandom
using PhysicsTools
using StructTypes
using StatsBase

using ..Spectral
using ..Medium



c_vac_m_p_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)

struct AngularEmissionProfile{U,T} end

struct CherenkovAngDistParameters{T<:Real}
    a::T
    b::T
    c::T
    d::T
end

# params for e-
STD_ANG_DIST_PARS = CherenkovAngDistParameters(4.27033, -6.02527, 0.29887, -0.00103)

"""
cherenkov_ang_dist(costheta, ref_index)

    Angular distribution of cherenkov photons for EM cascades.

    Taken from https://arxiv.org/pdf/1210.5140.pdf
"""
function cherenkov_ang_dist(
    costheta::Real,
    ref_index::Real,
    ang_dist_pars::CherenkovAngDistParameters=STD_ANG_DIST_PARS)

    cos_theta_c = 1 / ref_index
    a = ang_dist_pars.a
    b = ang_dist_pars.b
    c = ang_dist_pars.c
    d = ang_dist_pars.d

    return a * exp(b * abs(costheta - cos_theta_c)^c) + d
end


"""
    cherenkov_ang_dist_int_analytic(ref_index, lower, upper, ang_dist_pars)

Integral of the cherenkov angular distribution function.
"""
function cherenkov_ang_dist_int_analytic(
    ref_index::Real,
    lower::Real=-1.0,
    upper::Real=1,
    ang_dist_pars::CherenkovAngDistParameters=STD_ANG_DIST_PARS)

    a = ang_dist_pars.a
    b = ang_dist_pars.b
    c = ang_dist_pars.c
    d = ang_dist_pars.d

    cos_theta_c = 1.0 / ref_index

    function indef_int(x)

        function lower_branch(x, cos_theta_c)
            return (1 / c * (c * d * x + (a * (cos_theta_c - x) * gamma(1 / c, -(b * (cos_theta_c - x)^c))) * (-(b * (cos_theta_c - x)^c))^(-1 / c)))
        end

        function upper_branch(x, cos_theta_c)
            return (1 / c * (c * d * x + (a * (cos_theta_c - x) * gamma(1 / c, -(b * (-cos_theta_c + x)^c))) * (-(b * (-cos_theta_c + x)^c))^(-1 / c)))
        end

        peak_val = lower_branch(cos_theta_c - 1e-5, cos_theta_c)

        if x <= cos_theta_c
            return lower_branch(x, cos_theta_c)
        else
            return upper_branch(x, cos_theta_c) + 2 * peak_val
        end
    end

    return indef_int(upper) - indef_int(lower)
end

struct ChAngDistInt
    interpolation
end

"""
    _interp_ch_ang_dist_int(ref_ix_low, ref_ix_hi, steps=0.01)
Interpolate the integral of the Cherenkov angular distance distribution
in the range [ref_ix_low, ref_ix_hi].
"""
function _interp_ch_ang_dist_int(ref_ix_low, ref_ix_hi, steps=0.01)
    ref_ixs = ref_ix_low:steps:ref_ix_hi
    A = map(rfx -> cherenkov_ang_dist_int_analytic(rfx, -1, 1), ref_ixs)
    ChAngDistInt(LinearInterpolation(ref_ixs, A))
end

(f::ChAngDistInt)(ref_ix::Real) = f.interpolation(ref_ix)

cherenkov_ang_dist_int = _interp_ch_ang_dist_int(1.1, 1.5)


"""
    LongitudinalParametersBase
Struct for storing the parameters of the longitudinal Cherenkov lightyield parametrisation.
These parameters are the baseline values at E=1GeV.

For obtaining the energy-correcected parameters, see `get_longitudinal_params`.

See: https://arxiv.org/pdf/1210.5140.pdf
"""
@kwdef struct LongitudinalParametersBase
    alpha::Float64
    beta::Float64
    b::Float64
end

StructTypes.StructType(::Type{<:LongitudinalParametersBase}) = StructTypes.Struct()

const LongitudinalParametersEMinus = LongitudinalParametersBase(alpha=2.01849, beta=1.45469, b=0.63207)
const LongitudinalParametersEPlus = LongitudinalParametersBase(alpha=2.00035, beta=1.45501, b=0.63008)
const LongitudinalParametersGamma = LongitudinalParametersBase(alpha=2.83923, beta=1.34031, b=0.64526)

get_longitudinal_params(::Type{PEPlus}) = LongitudinalParametersEPlus
get_longitudinal_params(::Type{PEMinus}) = LongitudinalParametersEMinus
get_longitudinal_params(::Type{PGamma}) = LongitudinalParametersGamma

"""
    get_longitudinal_params(::Type{ptype}, energy)
Calculate the longitudinal parameters at `energy` (in GeV).
"""
function get_longitudinal_params(::Type{ptype}, energy) where {ptype}
    long_pars = get_longitudinal_params(ptype)
    a = long_pars.alpha + long_pars.beta * log10(energy)
    b = long_pars.b

    return a, b
end

"""
Struct for storing the energy-corrected values of the longitudinal Cherenkov lightyield parametrisation.
"""
@kwdef struct LongitudinalParameterisation{T<:Real}
    a::T
    b::T
    lrad::T
end

StructTypes.StructType(::Type{<:LongitudinalParameterisation}) = StructTypes.Struct()

"""
    LongitudinalParameterisation(energy, medium::MediumProperties, ::Type{ptype})
Construct `LongitudinalParameterisation` for a particle of type `ptype` at energy `energy`
"""
function LongitudinalParameterisation(energy, medium::MediumProperties, ::Type{ptype}) where {ptype}
   
    a, b = get_longitudinal_params(ptype, energy)

    unit_conv = 10 # g/cm^2 / "kg/m^3" in m
    lrad = radiation_length(medium) / material_density(medium) * unit_conv

    T = typeof(energy)

    LongitudinalParameterisation(T(a), T(b), T(lrad))
end

"""
Struct for storing the parameters of the Cherenkov track length parametrisation.
See: https://arxiv.org/pdf/1210.5140.pdf
"""
@kwdef struct CherenkovTrackLengthParameters
    alpha::Float64 # cm
    beta::Float64
    alpha_dev::Float64 # cm
    beta_dev::Float64
end

StructTypes.StructType(::Type{CherenkovTrackLengthParameters}) = StructTypes.Struct()

const CherenkovTrackLengthParametersEMinus = CherenkovTrackLengthParameters(
    alpha=5.3207078881,
    beta=1.00000211,
    alpha_dev=0.0578170887,
    beta_dev=0.5
)

const CherenkovTrackLengthParametersEPlus = CherenkovTrackLengthParameters(
    alpha=5.3211320598,
    beta=0.99999254,
    alpha_dev=0.0573419669,
    beta_dev=0.5
)

const CherenkovTrackLengthParametersGamma = CherenkovTrackLengthParameters(
    alpha=5.3208540905,
    beta=0.99999877,
    alpha_dev=0.0578170887,
    beta_dev=5.66586567
)

get_track_length_params(::Type{PEPlus}) = CherenkovTrackLengthParametersEPlus
get_track_length_params(::Type{PEMinus}) = CherenkovTrackLengthParametersEMinus
get_track_length_params(::Type{PGamma}) = CherenkovTrackLengthParametersGamma

"""
Struct for holding both the longitudinal and the track length parametrisation
"""
@kwdef struct LightyieldParametrisation
    longitudinal::LongitudinalParametersBase
    track_length::CherenkovTrackLengthParameters
end

StructTypes.StructType(::Type{LightyieldParametrisation}) = StructTypes.Struct()


"""
    longitudinal_profile(z::Real, medium::MediumProperties, long_param::LongitudinalParameterisation)

Longitudinal shower profile (PDF) at shower depth z ( in m )
"""
function longitudinal_profile(z::Real, long_param::LongitudinalParameterisation)

    a = long_param.a
    b = long_param.b
    lrad = long_param.lrad

    t = z / lrad

    b * ((t * b)^(a - 1.0) * exp(-(t * b)) / gamma(a))
end

function longitudinal_profile(
    energy, z, medium, ::Type{ptype}) where {ptype<:ParticleType}
    longitudinal_profile(energy, z, medium, LongitudinalParameterisation(energy, medium, ptype))
end

"""
    gamma_cdf(a, b, z)

Cumulative Gamma distribution
\$ int_0^z Gamma(a, b) \$
"""
gamma_cdf(a, b, z) = 1.0 - gamma(a, b * z) / gamma(a)


"""
    integral_long_profile(z_low::Real, z_high::Real, long_param::LongitudinalParameterisation)

Integral of the longitudinal profile from shower depth z_low (in m) to z_high
"""
function integral_long_profile(z_low::Real, z_high::Real, long_param::LongitudinalParameterisation)
    a = long_param.a
    b = long_param.b
    lrad = long_param.lrad

    t_low = z_low / lrad
    t_high = z_high / lrad
    gamma_cdf(a, b, t_high) - gamma_cdf(a, b, t_low)
end

function integral_long_profile(energy::Real, z_low::Real, z_high::Real, medium::MediumProperties, ::Type{ptype}) where {ptype<:ParticleType}
    integral_long_profile(energy, z_low, z_high, medium, LongitudinalParameterisation(energy, medium, ptype))
end


function fractional_contrib_long!(
    z_grid::AbstractVector{T},
    long_param::LongitudinalParameterisation,
    output::AbstractVector{T}
) where {T<:Real}
    if length(z_grid) != length(output)
        error("Grid and output are not of the same length")
    end

    norm = integral_long_profile(z_grid[1], z_grid[end], long_param)

    output[1] = 0
    @inbounds for i in 1:size(z_grid, 1)-1
        output[i+1] = (
            1 / norm * integral_long_profile(z_grid[i], z_grid[i+1], long_param)
        )
    end
    output
end

function fractional_contrib_long!(
    energy,
    z_grid,
    medium,
    ::Type{ptype},
    output) where {ptype<:ParticleType}
    fractional_contrib_long!(z_grid, LongitudinalParameterisation(energy, medium, ptype), output)
end

function fractional_contrib_long(
    energy::Real,
    z_grid::AbstractVector{T},
    medium::MediumProperties,
    pars_or_ptype::Union{LongitudinalParameterisation,ptype}
) where {T<:Real,ptype}
    output = similar(z_grid)
    fractional_contrib_long!(energy, z_grid, medium, pars_or_ptype, output)
end


function cascade_cherenkov_track_length_dev(energy::Real, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha_dev * energy^track_len_params.beta_dev
end
cascade_cherenkov_track_length_dev(energy::Real, ::Type{ptype}) where {ptype<:ParticleType} = cascade_cherenkov_track_length_dev(energy, get_track_length_params(ptype))

"""
    cascade_cherenkov_track_length(energy::Real, track_len_params::CherenkovTrackLengthParameters)

Cherenkov track length for cascades.
energy in GeV

returns track length in m
"""
function cascade_cherenkov_track_length(energy::Real, track_len_params::CherenkovTrackLengthParameters)
    track_len_params.alpha * energy^track_len_params.beta
end


function cascade_cherenkov_track_length(energy::Real, ::Type{ptype}) where {ptype<:ParticleType}
    return cascade_cherenkov_track_length(energy, get_track_length_params(ptype))
end


"""
    rel_additional_track_length_params(ref_index)

Interpolate additional track length parameters, taken from
https://arxiv.org/pdf/1206.5530.pdf using E_cut = 500MeV
"""
function rel_additional_track_length_params(ref_index)

    xs = [1.30, 1.33, 1.36]
    λs = [0.1842, 0.1880, 0.1916]
    κs = [0.0204, 0.0206, 0.0207]

    slope_λ = (λs[end] - λs[1]) / (xs[end] - xs[1])
    slope_κ = (κs[end] - κs[1]) / (xs[end] - xs[1])

    λ = λs[1] + (ref_index - xs[1]) * slope_λ
    κ = κs[1] + (ref_index - xs[1]) * slope_κ

    return λ, κ
end

"""
    rel_additional_track_length(ref_index)
Calculate additional track length for muons.
From https://arxiv.org/pdf/1206.5530.pdf
"""
function rel_additional_track_length(ref_index, energy)
    λ, κ = rel_additional_track_length_params(ref_index)
    return λ + κ * log(energy)
end


function total_lightyield(::Track, energy::Number, length::Number, medium, wl_range)

    # This is probably correct...
    function integrand(wl)
        ref_ix = refractive_index(wl, medium)
        return frank_tamm(wl, ref_ix) * (1 + rel_additional_track_length(ref_ix, energy))
    end
    T = typeof(energy)
    total_contrib = integrate_gauss_quad(integrand, wl_range[1], wl_range[2]) * T(1E9) * length

    #=
    function integrand2(wl)
        lmu = frank_tamm_norm(wl_range, wl -> refractive_index(wl, medium))
        ref_ix = refractive_index(wl, medium)

        fadd(wl) = rel_additional_track_length(refractive_index(wl, medium), energy)
        dfadd(wl) = ForwardDiff.derivative(fadd, wl)

        return (
            frank_tamm(wl, ref_ix) * (1 + rel_additional_track_length(ref_ix, energy)) * 1E9 +
            lmu * dfadd(wl)
        )
    end

    total_contrib_full = integrate_gauss_quad(integrand2, wl_range[1], wl_range[2]) * length
    =#

    return total_contrib
end


function total_lightyield(::Track, particle::Particle, medium, wl_range)
    return total_lightyield(Track(), particle.energy, particle.length, medium, wl_range)
end

function total_lightyield(::Cascade, particle, medium, wl_range)
    total_contrib = (
        frank_tamm_norm(wl_range, wl -> refractive_index(wl, medium)) *
        cascade_cherenkov_track_length(particle.energy, particle.type)
    )
    return total_contrib
end


function total_lightyield(
    particle::Particle{T,PType},
    medium::MediumProperties,
    wl_range
) where {T,PType}
    return total_lightyield(particle_shape(PType), particle, medium, wl_range)
end


abstract type PhotonSource{T} end


function split_source(source::T, parts::Integer) where {T<:PhotonSource}
    if source.photons < parts
        error("Cannot split source. Fewer photons than parts")
    end

    ph_split, remainder = divrem(source.photons, parts)
    out_sources = Vector{T}(undef, parts)

    for i in 1:parts

        if i == parts
            nph = ph_split + remainder
        else
            nph = ph_split
        end

        out_fields = []
        for field in fieldnames(T)
            if field == :photons
                push!(out_fields, nph)
            else
                push!(out_fields, getfield(source, field))
            end
        end

        out_sources[i] = T(out_fields...)
    end

    return out_sources
end



"""
    oversample_source(em::T, factor::Number) where {T <: PhotonSource}

Return a new emitter of type `T` with number of photons increased by `factor`
"""
function oversample_source(em::T, factor::Number) where {T<:PhotonSource}
    args = []
    for field in fieldnames(T)
        if field == :photons
            push!(args, getfield(em, field) * factor)
        else
            push!(args, getfield(em, field))
        end
    end
    return T(args...)
end
struct AxiconeEmitter{T} <: PhotonSource{T}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    photons::Int64
    angle::T
end

StructTypes.StructType(::Type{<:AxiconeEmitter}) = StructTypes.Struct()

struct PencilEmitter{T} <: PhotonSource{T}
    position::SVector{3,T}
    direction::SVector{3,T}
    beam_divergence::T
    time::T
    photons::Int64

end

StructTypes.StructType(::Type{<:PencilEmitter}) = StructTypes.Struct()

struct PointlikeIsotropicEmitter{T} <: PhotonSource{T}
    position::SVector{3,T}
    time::T
    photons::Int64
end

StructTypes.StructType(::Type{<:PointlikeIsotropicEmitter}) = StructTypes.Struct()


struct PointlikeTimeRangeEmitter{T,U} <: PhotonSource{T}
    position::SVector{3,T}
    time_range::Tuple{U,U}
    photons::Int64
end


StructTypes.StructType(::Type{<:PointlikeTimeRangeEmitter}) = StructTypes.Struct()

abstract type CherenkovEmitter{T} <: PhotonSource{T} end

struct ExtendedCherenkovEmitter{T} <: CherenkovEmitter{T}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    photons::Int64
    long_param::LongitudinalParameterisation{T}
end

StructTypes.StructType(::Type{<:ExtendedCherenkovEmitter}) = StructTypes.Struct()


function ExtendedCherenkovEmitter(
    particle::Particle,
    medium::MediumProperties,
    wl_range::Tuple{T,T};
    oversample=1.0
) where {T<:Real}

    long_param = LongitudinalParameterisation(particle.energy, medium, particle.type)
    photons = pois_rand(total_lightyield(particle, medium, wl_range) * oversample)

    ExtendedCherenkovEmitter(particle.position, particle.direction, particle.time, photons, long_param)
end

struct PointlikeCherenkovEmitter{T} <: CherenkovEmitter{T}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    photons::Int64
end

StructTypes.StructType(::Type{<:PointlikeCherenkovEmitter}) = StructTypes.Struct()

function PointlikeCherenkovEmitter(
    particle::Particle,
    medium::MediumProperties,
    wl_range::Tuple{T,T}) where {T<:Real}

    photons = pois_rand(total_lightyield(particle, medium, wl_range))
    PointlikeCherenkovEmitter(particle.position, particle.direction, particle.time, photons)
end

function PointlikeCherenkovEmitter(particle::Particle, photons::Integer)
    PointlikeCherenkovEmitter(particle.position, particle.direction, particle.time, photons)
end

struct CherenkovTrackEmitter{T} <: CherenkovEmitter{T}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::T
    length::T
    photons::Int64
end

StructTypes.StructType(::Type{<:CherenkovTrackEmitter}) = StructTypes.Struct()

function CherenkovTrackEmitter(particle::Particle{T}, medium::MediumProperties, wl_range::Tuple{T,T}) where {T<:Real}
    n_photons = pois_rand(total_lightyield(particle, medium, wl_range))
    return CherenkovTrackEmitter(particle.position, particle.direction, particle.time, particle.length, n_photons)
end

function LightsabreMuonEmitter(particle::Particle{T}, medium::MediumProperties, wl_range::Tuple{T,T}) where {T<:Real}

    lys = Float64[]
    for i in 1:100
        p, secondaries = propagate_muon(particle)
        ly_secondaries = sum(total_lightyield.(secondaries, Ref(medium), Ref(wl_range)))
        push!(lys, ly_secondaries)
    end
    ly_secondaries = mean(lys)

    n_photons = pois_rand(ly_secondaries)
    return CherenkovTrackEmitter(particle.position, particle.direction, particle.time, particle.length, n_photons)
end


end

module Detection

using StaticArrays
using DataFrames
using Interpolations
using LinearAlgebra
using Base.Iterators
using Rotations
using StructTypes
using PhysicsTools
using NeutrinoTelescopeBase

export apply_wl_acceptance
export check_pmt_hit
export InterpQuantumEff





"""
    check_pmt_hit_opening_angle(rel_hit_position, pmt_positions, opening_angle)

Check if the relative hit position is within the opening angle of any PMT.

# Arguments
- `rel_hit_position::SVector{3,<:Real}`: The relative hit position.
- `pmt_positions::AbstractVector{T}`: The positions of the PMTs.
- `opening_angle::Real`: The opening angle threshold.

# Returns
- `j::Int`: The index of the first PMT within the opening angle, or 0 if none.

"""
function check_pmt_hit_opening_angle(
    rel_hit_position::SVector{3,<:Real},
    pmt_positions::AbstractVector{T},
    opening_angle::Real
) where {T<:AbstractVector{<:Real}}

    for (j, pmtpos) in enumerate(pmt_positions)
        angle_pos_pmt = acos(clamp(dot(rel_hit_position, pmtpos), -1.0, 1.0))
        if angle_pos_pmt < opening_angle
            return j
        end
    end
    return 0
end

check_pmt_hit(
    ::AbstractVector,
    ::AbstractVector,
    ::AbstractVector,
    ::PhotonTarget,
    ::Rotation) = error("Not implemented")

"""
    function check_pmt_hit(
        hit_positions::AbstractVector,
        ::AbstractVector,
        ::AbstractVector,
        d::HomogeneousTarget,
        ::Rotation)

    Test if photons hit a pmt and return a vector of hit pmt indices (0 for no hit).
    
    Note: This function effectively resamples the (weighted) photons after photon propagation to hits
    with their natural rate.
"""
function check_pmt_hit(
    hit_positions::AbstractVector,
    ::AbstractVector,
    ::AbstractVector,
    prop_weights::AbstractVector,
    d::HomogeneousTarget,
    ::Rotation)
    
    hit_prob = d.active_area / surface_area(d.shape)
    # Mask is 1 if rand() < hit_prob
    hit_mask = rand(length(hit_positions)) .< hit_prob
    
    return hit_mask

end

function check_pmt_hit(
    hit_positions::AbstractVector{T},
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    target::SphericalMultiPMTTarget,
    orientation::Rotation{3,<:Real}) where {T<:SVector{3,<:Real}}

    # For SphericalMultiPMTTarget, xsec is always the same
    xsec = cross_section(target.shape, first(hit_directions))

    pmt_positions = get_pmt_positions(target, orientation)
    pmt_hit_ids = zeros(Int, length(hit_positions))
    rel_pmt_weights = zeros(Float64, get_pmt_count(target))


    cos_thetas = zeros(size(pmt_positions))

    for (hit_ix, (hp, hd)) in enumerate(zip(hit_positions, hit_directions))
        cos_thetas .= dot.(Ref(hd), pmt_positions)
        projected_areas = sum(cos_thetas[cos_thetas .> 0]) * target.pmt_area

        total_hit_prob = projected_areas / xsec
        if rand() > total_hit_prob
            continue
        end

        for (pmt_ix, pmt_pos) in enumerate(pmt_positions)
            rel_hit_pos = (hp .- target.shape.position) ./ target.shape.radius
            rel_costheta = -dot(rel_hit_pos, pmt_pos) 
            rel_pmt_weights[pmt_ix] = rel_costheta
        end

        w = ProbabilityWeights(rel_pmt_weights)
        pmt_hit_ids[hit_ix] = sample(1:get_pmt_count(target), w)

    end
    return pmt_hit_ids


    #=
    wl_acc = target.wl_acceptance.(hit_wavelengths)
    
    hits = rand(length(hit_positions)) .< wl_acc

    pmt_positions = get_pmt_positions(target, orientation)
    pmt_radius = sqrt(target.pmt_area / π)
    opening_angle = asin(pmt_radius / target.shape.radius)


    pmt_hit_ids = zeros(length(hit_positions))

    tpos = convert(SVector{3,Float64}, target.shape.position)
    rel_pos = hit_positions[hits] .- Ref(tpos)
    rel_pos = rel_pos ./ norm.(rel_pos)
    pmt_hit_ids[hits] .= check_pmt_hit_opening_angle.(rel_pos, Ref(pmt_positions), Ref(opening_angle))

    return pmt_hit_ids
    =#

end

apply_wl_acceptance(
    ::AbstractVector,
    ::AbstractVector,
    ::AbstractVector,
    t::PhotonTarget,
    ::Rotation{3,<:Real}) = error("not implemented for type $(typeof(t))")

apply_wl_acceptance(
    hit_positions::AbstractVector,
    ::AbstractVector,
    ::AbstractVector,
    t::SphericalMultiPMTTarget,
    ::Rotation{3,<:Real}) =  ones(Bool, size(hit_positions))



apply_qe(::AbstractVector, t::PhotonTarget) = error("not implemented for type $(typeof(t))")
    
abstract type PMTAcceptance end


struct SimpleAcceptance{T<:Real} <: PMTAcceptance
    pmt_eff_area::T
end

abstract type QuantumEff end
 
struct InterpQuantumEff{I} <: QuantumEff
    qe::I
end

function InterpQuantumEff(fname::String, relative=false)
    df = DataFrame(CSV.File(fname))
    y = nothing
    if relative
        y = df[:, :rel_acceptance]
    else
        y = df[:, :QE]./100
    end

    interp = linear_interpolation(df[:, :wavelength], y, extrapolation_bc=0.)
    return InterpQuantumEff(interp)
end


(qe::InterpQuantumEff)(x) = qe.qe(x)

apply_qe(wls::AbstractVector, t::SphericalMultiPMTTarget) = t.wl_acceptance.(wls)


include("pom.jl")
include("dom.jl")
include("generic_multipmt.jl")

end

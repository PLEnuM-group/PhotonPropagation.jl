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

    # For POM, xsec is always the same (spherical module)
    xsec = cross_section(target, first(hit_directions))

    total_hit_prob = get_pmt_count(target) * target.acceptance.pmt_eff_area / xsec

    pmt_positions = get_pmt_positions(target, orientation)
    pmt_hit_ids = zeros(Int, length(hit_positions))
    rel_pmt_weights = zeros(Float64, get_pmt_count(target))

    for (hit_ix, hd) in enumerate(hit_directions)
        if rand() > total_hit_prob
            continue
        end

        for (pmt_ix, pmt_pos) in enumerate(pmt_positions)
            rel_costheta = -dot(hd, pmt_pos) 
            rel_pmt_weights[pmt_ix] = rel_costheta
        end

        w = ProbabilityWeights(rel_pmt_weights)
        pmt_hit_ids[hit_ix] = sample(1:get_pmt_count(target), w)


    end



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

apply_qe(::AbstractVector, t::PhotonTarget) = error("not implemented for type $(typeof(t))")
    
abstract type PMTAcceptance end


struct SimpleAcceptance{T<:Real} <: PMTAcceptance
    pmt_eff_area::T
end

include("pom.jl")
include("dom.jl")


end

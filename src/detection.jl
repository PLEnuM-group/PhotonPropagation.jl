module Detection

using StaticArrays
using DataFrames
using Interpolations
using LinearAlgebra
using Base.Iterators
using Rotations
using StructTypes
using PhysicsTools

export geometry_type, TargetShape, Spherical, Rectangular, Circular
export surface_area
export PhotonTarget, PixelatedTarget, SphericalMultiPMTDetector, get_pmt_count
export HomogeneousDetector, RectangularDetector, CircularDetector
export SphericalHomogeneousDetector
export check_pmt_hit
export get_pmt_positions
export apply_wl_acceptance
export DummyTarget, DummyShape
export cross_section

"""
    TargetShape
Base type for photon target shapes.
"""
abstract type TargetShape end

"""
surface_area(::TargetShape)
    Return the surface area of the target shape.
"""
surface_area(::TargetShape) = error("Not implemented")
cross_section(::TargetShape, direction) = error("Not implemented")

"""
    DummyShape{T <: Real} <: TargetShape

Dummy target shape, used to define a position.

### Fields
- `position` - Shape position
"""
struct DummyShape{T <: Real} <: TargetShape
    position::SVector{3, T}
end


"""
    Spherical{T <: Real} <: TargetShape

Spherical target shape

### Fields
- `position` - center point of the sphere (in m)
- `radius` - sphere radius (in m)
"""
struct Spherical{T <: Real} <: TargetShape
    position::SVector{3, T}
    radius::T
end

surface_area(s::Spherical) = 4*π*s.radius^2
cross_section(s::Spherical, direction) = π*s.radius^2

StructTypes.StructType(::Type{<:Spherical}) = StructTypes.Struct()

"""
    Rectangular{T <: Real} <: TargetShape

Rectangular target shape. 
*CAUTION*: The orientation is currently always e_z

### Fields
- `position` - center point of the rectangle (in m)
- `length_x` - length in x direction (in m)
- `length_y` - length in y direction (in m)
"""
struct Rectangular{T <: Real} <: TargetShape
    position::SVector{3, T}
    length_x::T
    length_y::T
end

surface_area(s::Rectangular) = s.length_x * s.length_y

StructTypes.StructType(::Type{<:Rectangular}) = StructTypes.Struct()

"""
    Circular{T <: Real} <: TargetShape

Circular target shape. 
*CAUTION*: The orientation is currently always e_z

### Fields
- `position` - center point of the rectangle (in m)
- `radius` - circle radius (in m)
"""
struct Circular{T <: Real} <: TargetShape 
    position::SVector{3,T}
    radius::T
end

surface_area(s::Circular) = π * s.radius^2

StructTypes.StructType(::Type{<:Circular}) = StructTypes.Struct()

function Base.convert(::Type{Spherical{T}}, x::Spherical) where {T}
    return Spherical(T.(x.position), T(x.radius))
end

function Base.convert(::Type{Rectangular{T}}, x::Rectangular) where {T}
    return Rectangular(T.(x.position), T(x.length_x), T(x.length_y))
end

function Base.convert(::Type{Circular{T}}, x::Circular) where {T}
    return Circular(T.(x.position), T(x.radius))
end



abstract type PhotonTarget{TS<:TargetShape} end
abstract type PixelatedTarget{TS<:TargetShape} <: PhotonTarget{TS} end

struct DummyTarget{TS<:TargetShape} <: PhotonTarget{TS}
    shape::TS
    module_id::UInt16
end

function DummyTarget(position::SVector{3, T}, module_id::Integer) where {T <: Real}
    return DummyTarget(DummyShape(position), UInt16(module_id))
end

geometry_type(::PhotonTarget{TS}) where {TS <:TargetShape} = TS


"""
    struct HomogeneousDetector{TS <: TargetShape} <: PhotonTarget{TS}

A struct representing a homogeneous detector.

# Fields
- `shape::TS`: The shape of the detector.
- `active_area::Float64`: The active area of the detector.
- `module_id::UInt16`: The module ID of the detector.

"""
struct HomogeneousDetector{TS <: TargetShape} <: PhotonTarget{TS}
    shape::TS
    active_area::Float64
    module_id::UInt16
end

StructTypes.StructType(::Type{<:HomogeneousDetector}) = StructTypes.Struct()


"""
    struct SphericalHomogeneousDetector{T} <: PhotonTarget{Spherical{T}}

A struct representing a spherical homogeneous detector.

# Fields
- `shape::Spherical{T}`: The shape of the detector.
- `active_area::Float64`: The active area of the detector.
- `module_id::UInt16`: The module ID of the detector.

"""
struct SphericalHomogeneousDetector{T} <: PhotonTarget{Spherical{T}}
    shape::Spherical{T}
    active_area::Float64
    module_id::Int32
end

StructTypes.StructType(::Type{<:SphericalHomogeneousDetector}) = StructTypes.Struct()

"""
    struct SphericalMultiPMTDetector{N,L,T,I} <: PixelatedTarget{Spherical{T}}

A struct representing a spherical multi-PMT detector.

# Fields
- `shape::Spherical{T}`: The shape of the detector.
- `pmt_area::Float64`: The area of each PMT.
- `pmt_coordinates::SMatrix{2,N,Float64,L}`: The coordinates of the PMTs.
- `wl_acceptance::I`: The wavelength acceptance of the detector.
- `module_id::UInt16`: The ID of the module.

"""
struct SphericalMultiPMTDetector{N,L,T,I} <: PixelatedTarget{Spherical{T}}
    shape::Spherical{T}
    pmt_area::Float64
    pmt_coordinates::SMatrix{2,N,Float64,L}
    wl_acceptance::I
    module_id::UInt16
end

StructTypes.StructType(::Type{<:SphericalMultiPMTDetector}) = StructTypes.Struct()

get_pmt_count(::HomogeneousDetector) = 1
get_pmt_count(::SphericalMultiPMTDetector{N,L}) where {N,L} = N
get_pmt_count(::Type{SphericalMultiPMTDetector{N,L}}) where {N,L} = N

"""
    get_pmt_positions(target, orientation)

Compute the positions of the photomultiplier tubes (PMTs) in a pixelated target.

# Arguments
- `target`: A `PixelatedTarget` object representing the pixelated target.
- `orientation`: A `Rotation{3,<:Real}` object representing the orientation of the target.

# Returns
An array of `SVector{3, eltype(target.pmt_coordinates)}` representing the positions of the PMTs.

"""
function get_pmt_positions(
    target::PixelatedTarget,
    orientation::Rotation{3,<:Real})

    pmt_positions::Vector{SVector{3,eltype(target.pmt_coordinates)}} = [
        orientation * sph_to_cart(det_θ, det_ϕ)
        for (det_θ, det_ϕ) in eachcol(target.pmt_coordinates)
    ]

    return pmt_positions
end

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
        d::HomogeneousDetector,
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
    d::HomogeneousDetector,
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
    target::SphericalMultiPMTDetector,
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

module Detection
using StaticArrays
using CSV
using DataFrames
using Interpolations
using Unitful
using LinearAlgebra
using Base.Iterators
using JSON
using Rotations

using PhysicsTools

export PhotonTarget, DetectionSphere, MultiPMTDetector, get_pmt_count
export geometry_type, Spherical, Rectangular, RectangularDetector, Circular, CircularDetector
export check_pmt_hit
export get_pmt_positions
export make_detector_cube, make_targets, make_detector_hex
export area_acceptance

const PROJECT_ROOT = pkgdir(Detection)

abstract type TargetShape end
struct Spherical{T <: Real} <: TargetShape
    position::SVector{3, T}
    radius::T
end

struct Rectangular{T <: Real} <: TargetShape
    position::SVector{3, T}
    length_x::T
    length_y::T
end

struct Circular{T <: Real} <: TargetShape 
    position::SVector{3,T}
    radius::T
end


function Base.convert(::Type{Spherical{T}}, x::Spherical) where {T}
    return Spherical(T.(x.position), T(x.radius))
end

function Base.convert(::Type{Rectangular{T}}, x::Rectangular) where {T}
    return Rectangular(T.(x.position), T(x.length_x), T(x.length_y))
end

function Base.convert(::Type{Circular{T}}, x::Circular) where {T}
    return Circular(T.(x.position), T(x.radius))
end

abstract type PhotonTarget{T<:TargetShape} end
abstract type PixelatedTarget{T<:TargetShape} <: PhotonTarget{T} end

geometry_type(::PhotonTarget{TS}) where {TS <:TargetShape} = TS


struct HomogeneousDetector{T <: Real, TS <: TargetShape} <: PhotonTarget{TS}
    shape::TS
    pmt_area::T
    module_id::UInt16
end

struct MultiPMTDetector{T<:Real,N,L, TS <: TargetShape} <: PixelatedTarget
    position::SVector{3,T}
    radius::T
    pmt_area::T
    pmt_coordinates::SMatrix{2,N,T,L}
    module_id::UInt16
end






function Base.convert(::Type{DetectionSphere{T}}, x::DetectionSphere) where {T}
    pos = T.(x.position)
    radius = T.(x.radius)
    pmt_area = T(x.pmt_area)
    return DetectionSphere(pos, radius, x.n_pmts, pmt_area, x.module_id)
end


function Base.convert(::Type{MultiPMTDetector{T, N, L}}, x::MultiPMTDetector) where {T,N,L}
    pos = T.(x.position)
    radius = T.(x.radius)
    pmt_area = T(x.pmt_area)
    pmt_coordinates = SMatrix{2, N, T, L}(x.pmt_coordinates)
    return MultiPMTDetector(pos, radius, pmt_area, pmt_coordinates, x.module_id)
end

# Assumes rectangle orientation is e_z
struct RectangularDetector{T<:Real} <: PhotonTarget
    position::SVector{3,T}
    length_x::T
    length_y::T
    module_id::UInt16
end
geometry_type(::Type{<:RectangularDetector}) = Rectangular()

struct CircularDetector{T<:Real} <: PhotonTarget
    position::SVector{3,T}
    radius::T
    module_id::UInt16
end
geometry_type(::Type{<:CircularDetector}) = Circular()

JSON.lower(d::MultiPMTDetector) = Dict(
    "pos" => d.position,
    "radius" => d.radius,
    "pmt_area" => d.pmt_area,
    "pmt_coordinates" => d.pmt_coordinates,
    "module_id" => Int(d.module_id))

get_pmt_count(::DetectionSphere) = 1
get_pmt_count(::MultiPMTDetector{T,N,L}) where {T,N,L} = N
get_pmt_count(::Type{MultiPMTDetector{T,N,L}}) where {T,N,L} = N

function Base.convert(::Type{MultiPMTDetector{T}}, x::MultiPMTDetector) where {T}

    pos = T.(x.position)
    radius = T(x.radius)
    pmt_area = T(x.pmt_area)
    pmt_coordinates = T.(x.pmt_coordinates)

    return MultiPMTDetector(pos, radius, pmt_area, pmt_coordinates, x.module_id)
end


function get_pmt_positions(
    target::PixelatedTarget,
    orientation::Rotation{3,<:Real})

    pmt_positions::Vector{SVector{3,eltype(target.pmt_coordinates)}} = [
        orientation * sph_to_cart(det_θ, det_ϕ)
        for (det_θ, det_ϕ) in eachcol(target.pmt_coordinates)
    ]

    return pmt_positions
end

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

check_pmt_hit(hit_positions::AbstractVector, ::AbstractVector, ::AbstractVector, ::DetectionSphere, ::Rotation) = ones(length(hit_positions))

function check_pmt_hit(
    hit_positions::AbstractVector{T},
    ::AbstractVector,
    ::AbstractVector,
    target::PixelatedTarget,
    orientation::Rotation{3,<:Real}) where {T<:SVector{3,<:Real}}

    pmt_positions = get_pmt_positions(target, orientation)
    pmt_radius = sqrt(target.pmt_area / π)
    opening_angle = asin(pmt_radius / target.radius)

    tpos = convert(SVector{3,Float64}, target.position)
    rel_pos = hit_positions .- Ref(tpos)
    rel_pos = rel_pos ./ norm.(rel_pos)
    pmt_hit_ids = check_pmt_hit_opening_angle.(rel_pos, Ref(pmt_positions), Ref(opening_angle))

    return pmt_hit_ids

end

function area_acceptance(::SVector{3,<:Real}, target::DetectionSphere)
    total_pmt_area = target.n_pmts * target.pmt_area
    detector_surface = 4 * π * target.radius^2

    return total_pmt_area / detector_surface
end

area_acceptance(::SVector{3,<:Real}, ::MultiPMTDetector) = 1
area_acceptance(::SVector{3,<:Real}, ::RectangularDetector) = 1
area_acceptance(::SVector{3,<:Real}, ::CircularDetector) = 1

struct PMTWavelengthAcceptance
    interpolation::Interpolations.Extrapolation

    PMTWavelengthAcceptance(interpolation::Interpolations.Extrapolation) = error("default constructor disabled")
    function PMTWavelengthAcceptance(xs::AbstractVector, ys::AbstractVector)
        new(LinearInterpolation(xs, ys))
    end
end

(f::PMTWavelengthAcceptance)(wavelength::Real) = f.interpolation(wavelength)
(f::PMTWavelengthAcceptance)(wavelength::Unitful.Length) = f.interpolation(ustrip(u"nm", wavelength))


abstract type PositionalAcceptance end

include("pom.jl")

end

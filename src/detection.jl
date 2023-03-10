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

export PhotonTarget, DetectionSphere, p_one_pmt_acc, MultiPMTDetector, get_pmt_count
export geometry_type, Spherical, Rectangular, RectangularDetector, Circular, CircularDetector
export check_pmt_hit
export make_detector_cube, make_targets, make_detector_hex
export area_acceptance

const PROJECT_ROOT = pkgdir(Detection)

abstract type PhotonTarget end
abstract type PixelatedTarget <: PhotonTarget end

abstract type TargetShape end
struct Spherical <: TargetShape end
struct Rectangular <: TargetShape end
struct Circular <: TargetShape end

struct DetectionSphere{T<:Real} <: PhotonTarget
    position::SVector{3,T}
    radius::T
    n_pmts::Int64
    pmt_area::T
    module_id::UInt16
end

geometry_type(::Type{<:DetectionSphere}) = Spherical()

function Base.convert(::Type{DetectionSphere{T}}, x::DetectionSphere) where {T}
    pos = T.(x.position)
    radius = T.(x.radius)
    pmt_area = T(x.pmt_area)
    return DetectionSphere(pos, radius, x.n_pmts, pmt_area, x.module_id)
end


struct MultiPMTDetector{T<:Real,N,L} <: PixelatedTarget
    position::SVector{3,T}
    radius::T
    pmt_area::T
    pmt_coordinates::SMatrix{2,N,T,L}
    module_id::UInt16
end
geometry_type(::Type{<:MultiPMTDetector}) = Spherical()

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
        orientation * sph_to_cart(det_??, det_??)
        for (det_??, det_??) in eachcol(target.pmt_coordinates)
    ]

    return pmt_positions
end

function check_pmt_hit(
    rel_hit_position::SVector{3,<:Real},
    pmt_positions::AbstractVector{T},
    opening_angle::Real
) where {T<:AbstractVector{<:Real}}

    for (j, pmtpos) in enumerate(pmt_positions)
        if acos(clamp(dot(rel_hit_position, pmtpos), -1.0, 1.0)) < opening_angle
            return j
        end
    end
    return 0
end

check_pmt_hit(hit_positions::AbstractVector, ::DetectionSphere, ::Rotation) = ones(length(hit_positions))

function check_pmt_hit(
    hit_positions::AbstractVector{T},
    target::PixelatedTarget,
    orientation::Rotation{3,<:Real}) where {T<:SVector{3,<:Real}}

    pmt_positions = get_pmt_positions(target, orientation)
    pmt_radius = sqrt(target.pmt_area / ??)
    opening_angle = asin(pmt_radius / target.radius)

    tpos = convert(SVector{3,Float64}, target.position)
    rel_pos = hit_positions .- Ref(tpos)
    rel_pos = rel_pos ./ norm.(rel_pos)
    pmt_hit_ids = check_pmt_hit.(rel_pos, Ref(pmt_positions), Ref(opening_angle))

    return pmt_hit_ids

end

function area_acceptance(::SVector{3,<:Real}, target::DetectionSphere)
    total_pmt_area = target.n_pmts * target.pmt_area
    detector_surface = 4 * ?? * target.radius^2

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


df = CSV.read(joinpath(PROJECT_ROOT, "assets/PMTAcc.csv",), DataFrame, header=["wavelength", "acceptance"])

p_one_pmt_acc = PMTWavelengthAcceptance(df[:, :wavelength], df[:, :acceptance])
end

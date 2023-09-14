
using Polynomials
using Distributions
using CSV
using Interpolations
using DataFrames

export DOMAcceptance, DOM

struct DOMAcceptance{I}
    poly_ang::Polynomial{Float64} 
    polymax::Float64
    int_wl::I
end

function DOMAcceptance(fname_ang::String, fname_wl::String)
    input = parse.(Float64, readlines(fname_ang))

    df = CSV.read(fname_wl, DataFrame)

    target_radius = 0.16510
    dom_area = π*target_radius^2.
    dom_eff = (df[:, :acc]/dom_area) # Relative to 0 deg injection angle
    wl_input = linear_interpolation(df[:, :wl], dom_eff, extrapolation_bc=0.)

    return DOMAcceptance(Polynomial(input[2:end]), input[1], wl_input)
end


struct DOM{T} <: PhotonTarget{Spherical{T}}
    shape::Spherical{T}
    pmt_area::Float64
    acceptance::DOMAcceptance
    module_id::UInt16

    function DOM{T}(position::SVector{3, T}, module_id::Integer) where {T <: Real}
        PROJECT_ROOT = pkgdir(@__MODULE__)

        pmt_area = (0.3048 / 2)^2 * π
        target_radius = 0.16510

        shape = Spherical(T.(position), T(target_radius))

        acceptance = DOMAcceptance(
            joinpath(PROJECT_ROOT, "assets/as.set50_p0=-0.27_p1=-0.042"),
            joinpath(PROJECT_ROOT, "assets/icecube_wl_acc.csv"),
        )
        
        dom = new{T}(shape, pmt_area, acceptance, UInt16(module_id))
        return dom
    end

    function DOM{T}(
        shape::Spherical{T},
        pmt_area::Float64,
        acceptance::DOMAcceptance,
        module_id::UInt16) where {T <: Real}

        return new{T}(shape, pmt_area, acceptance, module_id)

    end
end

DOM(position::SVector{3, T}, module_id::Integer) where {T} = DOM{T}(position, module_id)

function Base.convert(::Type{DOM{T}}, x::DOM) where {T}
    shape = convert(Spherical{T}, x.shape)
    return DOM{T}(shape, x.pmt_area, x.acceptance, x.module_id)
end

get_pmt_count(::DOM) = 1
get_pmt_count(::Type{<:DOM}) = 1

StructTypes.StructType(::Type{<:DOM}) = StructTypes.CustomStruct()
StructTypes.lower(x::DOM) = (x.shape.position, x.module_id)
StructTypes.lowertype(::Type{DOM{T}}) where {T} = Tuple{SVector{3, T}, UInt16}
StructTypes.construct(::Type{DOM{T}}, x::Tuple{SVector{3, T}, UInt16}) where {T} = DOM(x[1], x[2])


function check_pmt_hit(
    hit_positions::AbstractVector,
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    prop_weight::AbstractVector,
    target::DOM,
    orientation::Rotation{3,<:Real})

    rotated = Ref(inv(orientation)) .* hit_directions  
    coszeniths = dot.(rotated, Ref([0, 0, -1]))

    wl_acc = target.acceptance.int_wl.(hit_wavelengths)
    ang_acc = target.acceptance.poly_ang.(coszeniths) 
 
    surv_prob = ang_acc .* prop_weight .* wl_acc

    samples = rand(length(coszeniths))
    accepted = samples .< surv_prob
    return accepted
end
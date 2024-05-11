export POMAcceptance, POM
export POMRelativeAcceptance
export get_pom_pmt_group
export make_pom_pmt_coordinates
export POMQuantumEff
export apply_qe

using HDF5
using StatsBase
using JSON3
using CSV
using MultivariateStats
using StructTypes
using Distributions

#=
struct POMAcceptance{I} <: PMTAcceptance
    pos_acc_grp_1::Array{Float64, 2}
    pos_acc_grp_2::Array{Float64, 2}
    PPCA_grp_1::PPCA{Float64}
    PPCA_grp_2::PPCA{Float64}
    bin_edges_1::Vector{Float64}
    bin_edges_2::Vector{Float64}
    pos_wl_acc::I    
end

StructTypes.StructType(::Type{Matrix{N}}) where N<:Number = StructTypes.CustomStruct()
StructTypes.lower(matrix::Matrix{N}) where N<:Number = (content=vec(matrix), size=size(matrix))
StructTypes.lowertype(::Type{Matrix{N}}) where N<:Number = @NamedTuple{content::Vector{N}, size::Tuple{Int, Int}}
function Matrix{N}(matrix::@NamedTuple{content::Vector{N}, size::Tuple{Int, Int}}) where N<:Number 
   return N.(reshape(matrix.content, (matrix.size)))
end

StructTypes.StructType(::Type{<:PPCA}) = StructTypes.Struct()


function POMAcceptance(pmt_acc_fname::String)
    fid = h5open(pmt_acc_fname, "r")
    pos_acc_1 = fid["acc_pmt_grp_1"][:, :]
    pos_acc_2 = fid["acc_pmt_grp_2"][:, :]
    att = attrs(fid)
    edges_x = JSON3.read(att["bin_edges_1"], Vector{Float64})
    edges_y = JSON3.read(att["bin_edges_2"], Vector{Float64})
    ppca_1 = JSON3.read(att["PPCA_grp_1"], PPCA{Float64})
    ppca_2 = JSON3.read(att["PPCA_grp_2"], PPCA{Float64})

    wl_acc_x = fid["wl_acceptance_factor_x"][:]
    wl_acc_y = fid["wl_acceptance_factor_y"][:]
    close(fid)
     
    total_acc = linear_interpolation(wl_acc_x, wl_acc_y, extrapolation_bc=0.)
   
    return POMAcceptance(
        pos_acc_1,
        pos_acc_2,
        ppca_1,
        ppca_2,
        edges_x,
        edges_y,
        total_acc)
end
=#



abstract type QuantumEff end
 
struct POMQuantumEff{I} <: QuantumEff
    rel_acceptance::I
end


function POMQuantumEff(fname::String)
    df = DataFrame(CSV.File(fname))
    interp = linear_interpolation(df[:, :wavelength], df[:, :rel_acceptance] ./ maximum(df[:, :rel_acceptance]), extrapolation_bc=0.)
    return POMQuantumEff(interp)
end


struct POMAcceptance{I} <: PMTAcceptance
    sigma_1::Float64
    sigma_2::Float64
    pos_wl_acc_1::I
    pos_wl_acc_2::I
end

function POMAcceptance(pmt_acc_fname::String)
    fid = h5open(pmt_acc_fname, "r")
    wls = fid["wavelengths"][:]
    total_acc_1 = linear_interpolation(wls, fid["acc_pmt_grp_1"][:], extrapolation_bc=0.)
    total_acc_2 = linear_interpolation(wls, fid["acc_pmt_grp_2"][:], extrapolation_bc=0.)
   
    sigma_1 = read(fid["sigma_grp_1"])
    sigma_2 = read(fid["sigma_grp_2"])

    return POMAcceptance(sigma_1, sigma_2, total_acc_1, total_acc_2)
end



"""
    POMRelativeAcceptance{I} <: PMTAcceptance

This parametrization uses a unit acceptance for PMTs in group 2 and a relative
acceptance (relative to group 2) for group 1. This is intended to be used when
the emission spectrum is already biased wrt. to the acceptance
"""
struct POMRelativeAcceptance{I} <: PMTAcceptance
    sigma_1::Float64
    sigma_2::Float64
    rel_pos_wl_acc_1::Float64
    pos_wl_acc_2::I
end

function POMRelativeAcceptance(pmt_acc_fname::String)
    fid = h5open(pmt_acc_fname, "r")
    wls = fid["wavelengths"][:]
    rel_acc_1 = fid["rel_acc_pmt_grp_1"][]
    total_acc_2 = linear_interpolation(wls, fid["acc_pmt_grp_2"][:], extrapolation_bc=0.)
    
    sigma_1 = read(fid["sigma_grp_1"])
    sigma_2 = read(fid["sigma_grp_2"])
    return POMRelativeAcceptance(sigma_1, sigma_2, rel_acc_1, total_acc_2)
end


struct POM{T, A <: PMTAcceptance, Q<:QuantumEff} <: PixelatedTarget{Spherical{T}}
    shape::Spherical{T}
    pmt_area::Float64
    pmt_coordinates::SMatrix{2,16,Float64}
    acceptance::A
    quantum_eff::Q
    module_id::UInt16
end


function POM(position::SVector{3, T}, module_id::Integer, acceptance_type::Type{<:PMTAcceptance}=POMAcceptance) where {T <: Real}
    PROJECT_ROOT = pkgdir(@__MODULE__)

    pmt_area = (75e-3 / 2)^2 * π
    target_radius = 0.3

    shape = Spherical(T.(position), T(target_radius))

    if acceptance_type <: POMRelativeAcceptance
        acceptance = POMRelativeAcceptance(
            joinpath(PROJECT_ROOT, "assets/rel_pmt_acc.hd5"),
        )
    elseif acceptance_type <: POMAcceptance
        acceptance = POMAcceptance(
            joinpath(PROJECT_ROOT, "assets/pmt_acc.hd5"),
        )
    else
        error("Unknown acceptance type: $acceptance_type")
    end

    qe = POMQuantumEff(joinpath(PROJECT_ROOT, "assets/PMTAcc.csv"),)

    pom = POM(shape, pmt_area, make_pom_pmt_coordinates(Float64), acceptance, qe, UInt16(module_id))
    return pom
end


function Base.convert(::Type{POM{T}}, x::POM) where {T}
    shape = convert(Spherical{T}, x.shape)
    return POM(shape, x.pmt_area, x.pmt_coordinates, x.acceptance, x.quantum_eff, x.module_id)
end

get_pmt_count(::POM) = 16
get_pmt_count(::Type{<:POM}) = 16

StructTypes.StructType(::Type{<:POM}) = StructTypes.CustomStruct()
StructTypes.lower(x::POM) = (x.shape.position, x.module_id)
StructTypes.lowertype(::Type{POM{T}}) where {T} = Tuple{SVector{3, T}, UInt16}
StructTypes.construct(::Type{POM{T}}, x::Tuple{SVector{3, T}, UInt16}) where {T} = POM(x[1], x[2])


function make_pom_pmt_coordinates(T::Type)

    coords = Matrix{T}(undef, 2, 16)

    #upper 
    coords[1, 1:4] .= deg2rad(90 - 25)
    coords[2, 1:4] = (range(0; step=π / 2, length=4))

    # upper 2
    coords[1, 5:8] .= deg2rad(90 - 57.5)
    coords[2, 5:8] = (range(π / 4; step=π / 2, length=4))

    # lower 2
    coords[1, 9:12] .= deg2rad(90 + 25)
    coords[2, 9:12] = [π/2, 0, 3*π/2, π]

    # lower
    coords[1, 13:16] .= deg2rad(90 + 57.5)
    coords[2, 13:16] = [π/4, 7/4*π, 5/4*π, 3/4*π]

    R = calc_rot_matrix(SA[0.0, 0.0, 1.0], SA[1.0, 0.0, 0.0])
    @views for col in eachcol(coords)
        cart = sph_to_cart(col[1], col[2])
        col[:] .= cart_to_sph((R * cart)...)
    end

    return SMatrix{2,16}(coords)
end



function calc_relative_pmt_coords(position::AbstractVector, direction::AbstractVector, pmt_coords_cart::AbstractVector)
    pos_norm = position ./ norm(position)
    rel_costheta = dot(pos_norm, pmt_coords_cart)

    in_pos_to_pmt = pmt_coords_cart .- pos_norm

    proj_pmt_in = in_pos_to_pmt .- (dot(in_pos_to_pmt, pos_norm) .* pos_norm)
    proj_in_dir_inpo = direction .- ((dot(direction, pos_norm) .* pos_norm))

    photon_dir_phi = acos(clamp((dot(proj_pmt_in, proj_in_dir_inpo) / (norm(proj_pmt_in) * norm(proj_in_dir_inpo))), -1, 1))
    photon_dir_costheta  = (dot(.-pos_norm, direction))

    return rel_costheta, photon_dir_costheta, sin(photon_dir_phi)
end

function get_pom_pmt_group(pmt_ix)
   return SA[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2][pmt_ix]
end

#=
function check_pmt_hit(
    hit_positions::AbstractVector,
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    prop_weight::AbstractVector,
    target::POM,
    orientation::Rotation{3,<:Real})


    pmt_positions = get_pmt_positions(target, orientation)

    bins_x = target.acceptance.bin_edges_1
    bins_y = target.acceptance.bin_edges_2
    
    wl_acceptance = target.acceptance.pos_wl_acc.(hit_wavelengths)

    prob_vec = zeros(get_pmt_count(target))
    pmt_hit_ids = zeros(length(hit_positions))

    ppcas = [target.acceptance.PPCA_grp_1, target.acceptance.PPCA_grp_2]
    acceptances = [target.acceptance.pos_acc_grp_1, target.acceptance.pos_acc_grp_2]


    @inbounds for (hit_id, (hit_pos, hit_dir)) in enumerate(zip(hit_positions, hit_directions))
        
        # Continue to next photon if this one doesn't survive propagation
        if rand() > prop_weight[hit_id]
            continue
        end
        
        rel_pos = (hit_pos .- target.shape.position) ./ target.shape.radius

        # Calc hit fraction per PMT
        for (pmt_ix, pmt_pos) in enumerate(pmt_positions)
    
            pmt_grp = get_pom_pmt_group(pmt_ix)

            pos_dir = hcat(calc_relative_pmt_coords(rel_pos, hit_dir, pmt_pos)...)
            pos_dir_traf = predict(ppcas[pmt_grp], pos_dir')

            i = clamp(searchsortedlast(bins_x, pos_dir_traf[1]), 1, length(bins_x)-1)
            j = clamp(searchsortedlast(bins_y, pos_dir_traf[2]), 1, length(bins_y)-1)

            prob_vec[pmt_ix] = acceptances[pmt_grp][i, j] .* wl_acceptance[hit_id]
        end            
        
        no_hit = reduce(*, 1 .- prob_vec)
        hit_prob = 1 - no_hit

        if rand() < hit_prob
            w = ProbabilityWeights(prob_vec)
            pmt_hit_ids[hit_id] = sample(1:length(pmt_positions), w)
        end
    end

    return pmt_hit_ids

end
=#

function apply_wl_acceptance(
    hit_positions::AbstractVector,
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    target::POM,
    orientation::Rotation{3,<:Real})

    # This is currently factored into the check_pmt_hit function
    accepted = ones(Bool, size(hit_wavelengths))
    return accepted
end

"""
    check_pmt_hit(
        hit_positions::AbstractVector,
        hit_directions::AbstractVector,
        hit_wavelengths::AbstractVector,
        target::POM{<:Real, POMRelativeAcceptance},
        orientation::Rotation{3,<:Real})

    Convert photon hits into PMT hits, returning a PMT id for each photon that is detected by a PMT
    and `0` for photons that did not hit a PMT.
    
    Here, we only apply the relative wavelength acceptance of PMT group 1 wrt. group 2.
    Intended to be used with a biased emission spectrum.

    NOTE: DO NOT USE
"""
function check_pmt_hit(
    hit_positions::AbstractVector,
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    target::POM{<:Real, <:POMRelativeAcceptance},
    orientation::Rotation{3,<:Real})

    error("Not Implemented")
    pmt_positions = get_pmt_positions(target, orientation)
    n_pmt = get_pmt_count(target)

    rel_total_acc_1 = target.acceptance.rel_pos_wl_acc_1

    prob_vec = zeros(get_pmt_count(target))
    pmt_hit_ids = zeros(length(hit_positions))

    dists_1 = Rayleigh(target.acceptance.sigma_1)
    dists_2 = Rayleigh(target.acceptance.sigma_2)

    @inbounds for (hit_id, hit_pos) in enumerate(hit_positions)
        
        rel_pos = (hit_pos .- target.shape.position) ./ target.shape.radius

        # Calc hit fraction per PMT
        for (pmt_ix, pmt_pos) in enumerate(pmt_positions)
    
            pmt_grp = get_pom_pmt_group(pmt_ix)
            rel_costheta = dot(rel_pos, pmt_pos)
            pt = acos(clamp(rel_costheta, -1, 1))

            # Reweighting
            # Base distribution is cospt ~ U[-1, 1] -> acos(cospt) ~ 0.5*sin(pt)
            pdf_eval = pmt_grp == 1 ? pdf(dists_1, pt) : pdf(dists_2, pt)
            rel_weight = pdf_eval / (0.5 .* sin(pt))

            hit_a_pmt_prob = pmt_grp == 1 ? rel_total_acc_1/n_pmt : 1/n_pmt

            prob_vec[pmt_ix] = rel_weight * hit_a_pmt_prob
        end

        no_hit = reduce(*, 1 .- prob_vec)
        hit_prob = 1 - no_hit
        @show hit_prob

        if rand() < hit_prob
            w = ProbabilityWeights(prob_vec)
            pmt_hit_ids[hit_id] = sample(1:length(pmt_positions), w)
        end
    end
    return pmt_hit_ids
end


function check_pmt_hit(
    hit_positions::AbstractVector,
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    target::POM{<:Real, <:POMAcceptance},
    orientation::Rotation{3,<:Real})

    pmt_positions = get_pmt_positions(target, orientation)

    total_acc_1::Vector{Float64} = target.acceptance.pos_wl_acc_1.(hit_wavelengths)
    total_acc_2::Vector{Float64} = target.acceptance.pos_wl_acc_2.(hit_wavelengths)

    prob_vec = zeros(get_pmt_count(target))
    pmt_hit_ids = zeros(length(hit_positions))

    dists_1 = Rayleigh(target.acceptance.sigma_1)
    dists_2 = Rayleigh(target.acceptance.sigma_2)

    @inbounds for (hit_id, hit_pos) in enumerate(hit_positions)

        rel_pos = (hit_pos .- target.shape.position) ./ target.shape.radius

        # Calc hit fraction per PMT
        for (pmt_ix, pmt_pos) in enumerate(pmt_positions)
    
            pmt_grp = get_pom_pmt_group(pmt_ix)
            rel_costheta = dot(rel_pos, pmt_pos)
            pt = acos(clamp(rel_costheta, -1, 1))

            # Reweighting
            # Base distribution is cospt ~ U[-1, 1] -> acos(cospt) ~ 0.5*sin(pt)
            pdf_eval = pmt_grp == 1 ? pdf(dists_1, pt) : pdf(dists_2, pt)
            rel_weight = sin(pt) != 0 ?  pdf_eval / (0.5 .* sin(pt)) : 0

            hit_a_pmt_prob = pmt_grp == 1 ? total_acc_1[hit_id] : total_acc_2[hit_id]

            # Reweight to pmt angular acceptance
            prob_vec[pmt_ix] = rel_weight * hit_a_pmt_prob #* π # hit_a_pmt_prob # 

        end

        # hit_a_pmt_prob is the probability to hit any pmt from a pmt group assuming
		# a uniform photon flux. Each pmt group contains 8 pmts, so divide by 8 to #account for the overcounting.
		# For a uniform photon flux, `hit_prob` should be total_acc_1 + total_acc_2
        prob_vec ./= 8
        
        hit_prob = sum(prob_vec)


         # Did we hit any pmt
        if rand() >=  hit_prob # ( total_acc_1[hit_id] + total_acc_2[hit_id])
            # no hit
            continue
        end

        w = ProbabilityWeights(prob_vec)
        pmt_hit_ids[hit_id] = sample(1:length(pmt_positions), w)
    end
    return pmt_hit_ids
end
        
apply_qe(wavelength::AbstractVector, t::POM) = t.quantum_eff.rel_acceptance(wavelength)
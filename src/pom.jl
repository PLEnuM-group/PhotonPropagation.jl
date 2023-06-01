export POMAcceptance, POM
export get_pom_pmt_group

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


struct POM{T,N,L} <: PixelatedTarget{Spherical{T}}
    shape::Spherical{T}
    pmt_area::Float64
    pmt_coordinates::SMatrix{2,N,Float64,L}
    acceptance::POMAcceptance
    module_id::UInt16
   
end




get_pmt_count(::POM{T,N,L}) where {T,N,L} = N
get_pmt_count(::Type{POM{T,N,L}}) where {T,N,L} = N



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
   return (div.(0:15,  4) .% 2)[pmt_ix] .+1
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

function check_pmt_hit(
    hit_positions::AbstractVector,
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    prop_weight::AbstractVector,
    target::POM,
    orientation::Rotation{3,<:Real})


    pmt_positions = get_pmt_positions(target, orientation)

    total_acc_1 = target.acceptance.pos_wl_acc_1.(hit_wavelengths)
    total_acc_2 = target.acceptance.pos_wl_acc_2.(hit_wavelengths)

    prob_vec = zeros(get_pmt_count(target))
    pmt_hit_ids = zeros(length(hit_positions))

    dists = [Rayleigh(target.acceptance.sigma_1), Rayleigh(target.acceptance.sigma_2)]
    accs = [total_acc_1, total_acc_2]

    @inbounds for (hit_id, hit_pos) in enumerate(hit_positions)
        
        # Continue to next photon if this one doesn't survive propagation
        if rand() > prop_weight[hit_id]
            continue
        end
        
        rel_pos = (hit_pos .- target.shape.position) ./ target.shape.radius

        # Calc hit fraction per PMT
        for (pmt_ix, pmt_pos) in enumerate(pmt_positions)
    
            pmt_grp = get_pom_pmt_group(pmt_ix)
            rel_costheta = dot(rel_pos, pmt_pos)
            pt = acos(clamp(rel_costheta, -1, 1))

            # Reweighting
            # Base distribution is cospt ~ U[-1, 1] -> acos(cospt) ~ 0.5*sin(pt)
            rel_weight = pdf(dists[pmt_grp], pt) / (0.5 .* sin(pt))

            hit_a_pmt_prob = accs[pmt_grp][hit_id]

            prob_vec[pmt_ix] = rel_weight * hit_a_pmt_prob
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
        
        
export POMAcceptance, POM
using HDF5
using StatsBase
using JSON3
using CSV
using MultivariateStats
using StructTypes

struct POMAcceptance{I} <: PMTAcceptance
    pos_acc_grp_1::Array{Float64, 3}
    pos_acc_grp_2::Array{Float64, 3}
    PPCA_grp_1::PPCA{Float64}
    PPCA_grp_2::PPCA{Float64}
    bin_edges_1::Vector{Float64}
    bin_edges_2::Vector{Float64}
    pos_wl_acc::I    
end

StructTypes.StructType(::Type{PPCA}) = StructTypes.Struct()


function POMAcceptance(pmt_acc_fname::String)
    fid = h5open(pmt_acc_fname, "r")
    pos_acc_1 = fid["acc_pmt_grp_1"][:, :]
    pos_acc_2 = fid["acc_pmt_grp_2"][:, :]
    att = attrs(fid)
    edges_x = Vector{Float64}(JSON3.read(att["bin_edges_1"]))
    edges_y = Vector{Float64}(JSON3.read(att["bin_edges_2"]))
    ppca_1 = PPCA{Float64}(JSON3.read(att["PPCA_grp_1"]))
    ppca_2 = PPCA{Float64}(JSON3.read(att["PPCA_grp_2"]))

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

    photon_dir_phi = acos((dot(proj_pmt_in, proj_in_dir_inpo) / (norm(proj_pmt_in) * norm(proj_in_dir_inpo))))
    photon_dir_costheta  = (dot(.-pos_norm, direction))

    return rel_costheta, photon_dir_costheta, sin(photon_dir_phi)
end

function calc_relative_pmt_coords(position::AbstractMatrix, direction::AbstractMatrix, pmt_coords_cart::AbstractVector)

    pts = Vector{Float64}(undef, size(position, 1))
    dts = Vector{Float64}(undef, size(position, 1))
    dps = Vector{Float64}(undef, size(position, 1))

    @inbounds for (i, (ipcr, ipr)) in enumerate(zip(eachrow(position), eachrow(direction)))
    
        pt, dt, dp = calc_relative_pmt_coords(ipcr, ipr, pmt_coords_cart)
        pts[i] = pt
        dts[i] = dt
        dps[i] = dp
    end

    return pts, dts, dps
end

function get_pom_pmt_group(pmt_ix)
    # pmt_groups = [vcat(1:4, 10:13), vcat(5:9, 13:16)]
    pmt_group = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2]

    return pmt_group[pmt_ix]

end


function check_pmt_hit(
    hit_positions::AbstractVector,
    hit_directions::AbstractVector,
    hit_wavelengths::AbstractVector,
    prop_weight::AbstractVector,
    target::POM,
    orientation::Rotation{3,<:Real})


    pmt_positions = get_pmt_positions(target, orientation)

    bins_x = target.acceptance.bin_edges_x
    bins_y = target.acceptance.bin_edges_y
    
    wl_acceptance = target.acceptance.pos_wl_acc.(hit_wavelengths)

    prob_vec = zeros(get_pmt_count(target))
    pmt_hit_ids = zeros(length(hit_positions))

    ppcas = [target.acceptance.PPCA_grp_1, target.acceptance.PPCA_grp_2]


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

            prob_vec[pmt_ix] = target.acceptance.pos_acc[i, j] .* wl_acceptance[hit_id]
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

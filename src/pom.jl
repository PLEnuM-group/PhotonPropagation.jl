export POMAcceptance, POM
using HDF5
using StatsBase
using JSON3
using CSV

struct POMAcceptance{I} <: PMTAcceptance
    pos_acc::Array{Float64, 3}
    bin_edges_x::Vector{Float64}
    bin_edges_y::Vector{Float64}
    bin_edges_z::Vector{Float64}
    pos_wl_acc::I
    
end

function POMAcceptance(pmt_acc_fname::String, pmt_wl_acc_fname::String)
    fid = h5open(pmt_acc_fname, "r")
    pos_acc = fid["pos_acceptance"][:, :, :]
    att = attrs(fid)
    edges_x = Vector{Float64}(JSON3.read(att["bin_edges_x"]))
    edges_y = Vector{Float64}(JSON3.read(att["bin_edges_y"]))
    edges_z = Vector{Float64}(JSON3.read(att["bin_edges_z"]))

    wl_acc_x = fid["wl_acceptance_factor_x"][:]
    wl_acc_y = fid["wl_acceptance_factor_y"][:]
    close(fid)

     
    acc_glass = linear_interpolation(wl_acc_x, wl_acc_y, extrapolation_bc=0.)

    df = CSV.read(pmt_wl_acc_fname, DataFrame, header=["wavelength", "acceptance"])
    # acc_pmt_wl = linear_interpolation(df[:, :wavelength], df[:, :acceptance], extrapolation_bc=0.)

    total_acc = linear_interpolation(
        df[:, :wavelength],
        acc_glass.(df[:, :wavelength]), # .* acc_pmt_wl.(df[:, :wavelength]),
        extrapolation_bc=0.
    )

    return POMAcceptance(pos_acc, edges_x, edges_y, edges_z, total_acc)
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



function calc_relative_pmt_coords(rot_mat::AbstractMatrix, in_position::AbstractVector, in_direction::AbstractVector)

    in_pos_rot = rot_mat * in_position
    in_dir_rot = rot_mat * in_direction

    in_pos_rot_rot_sph = cart_to_sph(in_pos_rot)



    # Calculate phi direction relative to glass position 
    # by rotating around e_z
    phi = cart_to_cyl(in_pos_rot)[2]
    Rs = AngleAxis.(-phi, 0, 0, 1)
    in_dir_rot_rel_ez_sph = cart_to_sph(Rs * in_dir_rot)
    
    return hcat(cos.(in_pos_rot_rot_sph[1]), in_pos_rot_rot_sph[2], cos(in_dir_rot_rel_ez_sph[1]), in_dir_rot_rel_ez_sph[2])

end

function calc_relative_pmt_coords(pmt_coords, in_position::AbstractMatrix, in_direction::AbstractMatrix)

    # Rotate pmt to e_z
    R = calc_rot_matrix(pmt_coords, [0, 0, 1])

    return calc_relative_pmt_coords.(Ref(R), eachrow(in_position), eachrow(in_direction))
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
    bins_z = target.acceptance.bin_edges_z
    
    wl_acceptance = target.acceptance.pos_wl_acc.(hit_wavelengths)

    rot_mats = calc_rot_matrix.(pmt_positions, Ref([0, 0, 1]))
    prob_vec = zeros(get_pmt_count(target))
    pmt_hit_ids = zeros(length(hit_positions))


    @inbounds for (hit_id, (hit_pos, hit_dir)) in enumerate(zip(hit_positions, hit_directions))
        
        # Continue to next photon if this one doesn't survive propagation
        if rand() > prop_weight[hit_id]
            continue
        end
        
        rel_pos = (hit_pos .- target.shape.position) ./ target.shape.radius

        # Calc hit fraction per PMT
        for (pmt_ix, rot_mat) in enumerate(rot_mats)
    
            pos_dir = calc_relative_pmt_coords(rot_mat, rel_pos, hit_dir)

            i = clamp(searchsortedlast(bins_x, pos_dir[1]), 1, length(bins_x)-1)
            j = clamp(searchsortedlast(bins_y, pos_dir[3]), 1, length(bins_y)-1)
            k = clamp(searchsortedlast(bins_z, pos_dir[4]), 1, length(bins_z)-1)

            prob_vec[pmt_ix] = target.acceptance.pos_acc[i, j, k] .* wl_acceptance[hit_id]
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

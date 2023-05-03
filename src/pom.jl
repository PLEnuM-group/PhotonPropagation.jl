export p_one_pmt_wl_acc, POM
using HDF5
using StatsBase
using JSON3

df = CSV.read(joinpath(PROJECT_ROOT, "assets/PMTAcc.csv",), DataFrame, header=["wavelength", "acceptance"])
p_one_pmt_wl_acc = PMTWavelengthAcceptance(df[:, :wavelength], df[:, :acceptance])

struct POMPositionalAcceptance <: PositionalAcceptance
    acc_hist::Array{Float64, 4}
    bin_edges_x::Vector{Float64}
    bin_edges_y::Vector{Float64}
    bin_edges_z::Vector{Float64}
end

function POMPositionalAcceptance(filename::String)
    fid = h5open(filename, "r")
    acc = fid["acceptance"][:, :, :, :]
    att = attrs(fid)
    edges_x = JSON3.read(att["bin_edges_x"])
    edges_y = JSON3.read(att["bin_edges_y"])
    edges_z = JSON3.read(att["bin_edges_z"])
    close(fid)
    return POMPositionalAcceptance(acc, edges_x, edges_y, edges_z)
end

p_one_pmt_acc = POMPositionalAcceptance(joinpath(PROJECT_ROOT, "assets/pmt_acc_3d.hd5"))



struct POM{T<:Real,N,L} <: PixelatedTarget
    position::SVector{3,T}
    radius::T
    pmt_area::T
    pmt_coordinates::SMatrix{2,N,T,L}
    module_id::UInt16
end

function Base.convert(::Type{POM{T}}, x::POM) where {T}

    pos = T.(x.position)
    radius = T(x.radius)
    pmt_area = T(x.pmt_area)
    pmt_coordinates = T.(x.pmt_coordinates)

    return POM(pos, radius, pmt_area, pmt_coordinates, x.module_id)
end

geometry_type(::Type{<:POM}) = Spherical()
get_pmt_count(::POM{T,N,L}) where {T,N,L} = N
get_pmt_count(::Type{POM{T,N,L}}) where {T,N,L} = N

area_acceptance(::SVector{3,<:Real}, ::POM) = 1

JSON.lower(d::POM) = Dict(
    "pos" => d.position,
    "radius" => d.radius,
    "pmt_area" => d.pmt_area,
    "pmt_coordinates" => d.pmt_coordinates,
    "module_id" => Int(d.module_id))

function calc_relative_pmt_coords(rot_mat::AbstractMatrix, in_position::AbstractVector, in_direction::AbstractVector)

    in_pos_rot = rot_mat * in_position
    in_dir_rot = rot_mat * in_direction

    in_pos_rot_rot_sph = cart_to_sph(in_pos_rot)

    # Calculate phi direction relative to glass position 
    # by rotating around e_z
    phi = cart_to_cyl(in_pos_rot)[2]
    Rs = AngleAxis.(-phi, 0, 0, 1)
    in_dir_rot_rel_ez_sph = cart_to_sph(Rs * in_dir_rot)
    
    return hcat(in_pos_rot_rot_sph..., in_dir_rot_rel_ez_sph...)

end

function calc_relative_pmt_coords(pmt_coords, in_position::AbstractMatrix, in_direction::AbstractMatrix)

    # Rotate pmt to e_z
    R = calc_rot_matrix(pmt_coords, [0, 0, 1])

    return calc_relative_pmt_coords.(Ref(R), eachrow(in_position), eachrow(in_direction))
end

function check_pmt_hit(
    hit_positions::AbstractVector{T},
    hit_directions::AbstractVector{T},
    target::POM,
    orientation::Rotation{3,<:Real}) where {T<:SVector{3,<:Real}}

    pmt_positions = get_pmt_positions(target, orientation)

    bins_x = p_one_pmt_acc.bin_edges_x
    bins_y = p_one_pmt_acc.bin_edges_y
    bins_z = p_one_pmt_acc.bin_edges_z

    rot_mats = calc_rot_matrix.(pmt_positions, Ref([0, 0, 1]))
    prob_vec = zeros(get_pmt_count(target))

    pmt_hit_ids = zeros(length(hit_positions))

    for (hit_id, (hit_pos, hit_dir)) in enumerate(zip(hit_positions, hit_directions))
        rel_pos = (hit_pos .- target.position) ./ target.radius

        # Calc hit fraction per PMT
        for (pmt_ix, rot_mat) in enumerate(rot_mats)
    
            pos_dir = calc_relative_pmt_coords(rot_mat, rel_pos, hit_dir)

            i = clamp(searchsortedlast(bins_x, pos_dir[1]), 1, length(bins_x)-1)
            j = clamp(searchsortedlast(bins_y, pos_dir[3]), 1, length(bins_y)-1)
            k = clamp(searchsortedlast(bins_z, pos_dir[4]), 1, length(bins_z)-1)

            prob_vec[pmt_ix] = p_one_pmt_acc.acc_hist[i, j, k]
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
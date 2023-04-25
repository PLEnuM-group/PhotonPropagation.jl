export p_one_pmt_wl_acc, POM
using HDF5
using StatsBase
using JSON3

df = CSV.read(joinpath(PROJECT_ROOT, "assets/PMTAcc.csv",), DataFrame, header=["wavelength", "acceptance"])
p_one_pmt_wl_acc = PMTWavelengthAcceptance(df[:, :wavelength], df[:, :acceptance])

struct POMPositionalAcceptance <: PositionalAcceptance
    acc_hist::Array{Float64, 3}
    bin_edges_x::Vector{Float64}
    bin_edges_y::Vector{Float64}
    bin_edges_z::Vector{Float64}
end

function POMPositionalAcceptance(filename::String)
    fid = h5open(filename, "r")
    acc = fid["acceptance"][:, :, :]
    att = attrs(fid)
    edges_x = JSON3.read(att["bin_edges_x"])
    edges_y = JSON3.read(att["bin_edges_y"])
    edges_z = JSON3.read(att["bin_edges_z"])
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


function calc_relative_pmt_coords(pmt_coords, in_position::AbstractMatrix, in_direction::AbstractMatrix)

    # Rotate pmt to e_z
    R = calc_rot_matrix(pmt_coords, [0, 0, 1])

    in_dir_rot = map(v -> R*v, eachrow(in_position))
    glass_pos_rot = map(v -> R*v, eachrow(in_direction))
   
    glass_pos_rot_sph = cart_to_sph.(glass_pos_rot)

    # Calculate phi direction relative to glass position 
    # by rotating around e_z
    phi = map(x -> cart_to_cyl(x)[2], glass_pos_rot)
    Rs = AngleAxis.(-phi, 0, 0, 1)

    in_dir_rot_rel_ez = Rs .* in_dir_rot
    in_dir_rot_rel_ez_sph = cart_to_sph.(in_dir_rot_rel_ez)
    
    return reduce(hcat, glass_pos_rot_sph), reduce(hcat, in_dir_rot_rel_ez_sph)
end


function calc_relative_pmt_coords(pmt_coords, in_position::AbstractVector, in_direction::AbstractVector)
    return calc_relative_pmt_coords(pmt_coords, reduce(hcat, in_position)', reduce(hcat, in_direction)')
end


function check_pom_pmt_hit(pmt_positions, hit_positions::AbstractVector, hit_directions::AbstractVector)

    bins_x = p_one_pmt_acc.bin_edges_x
    bins_y = p_one_pmt_acc.bin_edges_y
    bins_z = p_one_pmt_acc.bin_edges_z

    n_hits = length(hit_positions)
    n_pmts = length(pmt_positions)

    prob_vec = zeros(n_pmts, n_hits)

    # Fill probability vector
    for (pmt_ix, pmt_vec) in enumerate(pmt_positions)
        pos_dir = vcat(calc_relative_pmt_coords(pmt_vec, hit_positions, hit_directions)...)
    
        i = clamp.(searchsortedlast.(Ref(bins_x), pos_dir[1, :]), 1, length(bins_x)-1)
        j = clamp.(searchsortedlast.(Ref(bins_y), pos_dir[3, :]), 1, length(bins_y)-1)
        k = clamp.(searchsortedlast.(Ref(bins_z), pos_dir[4, :]), 1, length(bins_z)-1)

        prob_vec[pmt_ix, :] = getindex.(Ref(p_one_pmt_acc.acc_hist), i, j, k) # p_one_pmt_acc.acc_hist[i, j, k]      
    end
    
    # TODO: in principle probabilities could become > 1, so add poisson sampling here
    total_prob = sum(prob_vec, dims=1)[:]

    pmt_ixs = zeros(n_hits)
    sucess = rand(n_hits) .< total_prob
    
    w = ProbabilityWeights.(eachcol(prob_vec[:, sucess]), total_prob[sucess])

    pmt_ixs[sucess] = sample.(Ref(1:length(pmt_positions)), w)

    return pmt_ixs
end


function check_pmt_hit(
    hit_positions::AbstractVector{T},
    hit_directions::AbstractVector{T},
    target::POM,
    orientation::Rotation{3,<:Real}) where {T<:SVector{3,<:Real}}

    pmt_positions = get_pmt_positions(target, orientation)

    rel_positions = (hit_positions .- Ref(target.position)) ./ target.radius



    pmt_hit_ids = check_pom_pmt_hit(pmt_positions, rel_positions, hit_directions)

    return pmt_hit_ids

end
using PhotonPropagation
using PhysicsTools
using StaticArrays
using CairoMakie
using DataFrames
using AbstractMediumProperties
using KM3NeTMediumProperties
using NeutrinoTelescopes
using LinearAlgebra
using CUDA
using LLVM.Interop




function line_sphere_intersection(center::AbstractVector{T}, radius::T, pos::AbstractVector{T}, dir::AbstractVector{T}) where {T <: Real}
    dpos = pos .- center
    target_rsq = radius^2

    a::T = dot(dir, dpos)
    pp_norm_sq::T = sum(dpos .^ 2)

    b::T = a^2 - pp_norm_sq + target_rsq

    isec = b >= 0

    b = abs(b)
    assume(b >= 0)

    d::T = ifelse(
        isec,
        -a - sqrt(b),
        T(NaN))

    return (isec, d)
end



function check_intersection(target_collection::GenericTargetCollection, pos::SVector{3,T}, dir::SVector{3,T}, step_size::T) where {T<:Real}
    isec = false
    d = NaN32

    for position in target_collection.positions
        isec, d = check_intersection(target_collection.shape, pos, dir, step_size)
        if isec
            return isec, d
        end
    end

    return (isec, d)
end


struct Test{N, T}
    positions::SVector{N, T}
end

isbitstype(Test{3, SVector{3, Float32}})


function check_intersection(target_collection::LineTargetCollection, pos::SVector{3,T}, dir::SVector{3,T}, step_size::T) where {T<:Real}
    
    pos_xy = pos[1:2]
    dir_xy = dir[1:2]
    
    for (line_ix, line_pos) in enumerate(eachcol(target_collection.line_positions))
        isec_line, d_line = line_sphere_intersection(line_pos, target_collection.shape.radius, pos_xy, dir_xy)

        if isec_line
            for (z_ix, z_pos) in enumerate(target_collection.z_positions)
                tpos = SA[line_pos[1], line_pos[2], z_pos]
                isec_mod, d_mod = line_sphere_intersection(tpos, target_collection.shape.radius, pos, dir)

                if isec_mod && (d_mod > 0) & (d_mod < step_size)
                    return (true, (line_ix, z_ix))
                end
            end
        end
    end

    return false, T(NaN)
end

function LineTargetCollection(lines_pos::SMatrix{2, N, T}, z_pos::SVector{M, T}) where {N, M, T}

    LINE_T = SVector{M, T}
    lines = SVector{N, LINE_T}

end

positions = Detectors.hex_grid_positions(6, 50; truncate=1)

first(eachcol(positions))

collect(range(0,1000,20))

col = LineTargetCollection(SMatrix{2, 70, Float32}(positions), SVector{20, Float32}(LinRange(0, 1000, 20)), Circular(1f0))
col2 = GenericTargetCollection(SVector{70*20}([SVector{3, Float32}([x, y, z]) for (x, y) in eachcol(positions) for z in (LinRange(0, 1000, 20))]), Circular(1f0))




line_xy = @SMatrix [0f0 10f0 20f0 ; 0f0 10f0 20f0]
z_pos = SA_F32[-10, 0, 10]

col = LineTargetCollection(line_xy, z_pos, Circular(1f0))

pos = SA_F32[-5, 1, 0]
dir = SA_F32[1, 0, 0]

check_intersection(col, pos, dir, 10f0)
check_intersection(col2, pos, dir, 10f0)



detector = make_hex_detector(6, 50, 20, 50, truncate=1)

pos_xyz = reduce(hcat, [mod.shape.position for line in detector for mod in line ])


unique_x = (sort(unique(round.(pos_xyz[1, :], digits=1))))
cell_width_x = minimum(diff(unique_x))
min_x = minimum(pos_xyz[1, :]) - cell_width_x / 2
max_x = maximum(pos_xyz[1, :]) + cell_width_x / 2
grid_x = min_x:cell_width_x:max_x

unique_y = (sort(unique(round.(pos_xyz[2, :], digits=1))))
cell_width_y = minimum(diff(unique_y))
min_y = minimum(pos_xyz[2, :]) - cell_width_y / 2
max_y = maximum(pos_xyz[2, :]) + cell_width_y / 2
grid_y = min_y:cell_width_y:max_y

unique_z = (sort(unique(round.(pos_xyz[3, :], digits=1))))
cell_width_z = minimum(diff(unique_z))
min_z = minimum(pos_xyz[3, :]) - cell_width_z / 2
max_z = maximum(pos_xyz[3, :]) + cell_width_z / 2
grid_z = min_z:cell_width_z:max_z




p0 = [28, -78, -250]

dir = [-3., 5., 7.]
dir = dir ./ hypot(dir...)

struct RegularGrid3D{T}
    x_lims::NTuple{2, T}
    x_step::T
    y_lims::NTuple{2, T}
    y_step::T
    z_lims::NTuple{2, T}
    z_step::T
end



function get_grid_coord(pos, grid::RegularGrid3D)
    x_along = (pos[1] - grid.x_lims[1]) / grid.x_step
    y_along = (pos[2] - grid.y_lims[1]) / grid.y_step
    z_along = (pos[3] - grid.z_lims[1]) / grid.z_step

    return (x_along, y_along, z_along)
end

function get_world_coord(grid_coord, grid::RegularGrid3D)

    return (
        grid.x_lims[1] + grid_coord[1] * grid.x_step,
        grid.y_lims[1] + grid_coord[2] * grid.y_step,
        grid.z_lims[1] + grid_coord[3] * grid.z_step
    )

end

function get_cell_index_from_grid_coord(grid_coords, grid::RegularGrid3D)
    n_cells_x = floor((grid.x_lims[2] - grid.x_lims[1]) / grid.x_step)
    n_cells_y = floor((grid.y_lims[2] - grid.y_lims[1]) / grid.y_step)
    return grid_coords[3] *n_cells_x * n_cells_y + grid_coords[2]*n_cells_x + grid_coords[1]
end

function get_cell_index(pos, grid::RegularGrid3D)
    grid_coords = floor.(get_grid_coord(pos, grid))
    return get_cell_index_from_grid_coord(grid_coords, grid)
end

function get_cell_coords(cell_index, grid::RegularGrid3D)
    n_cells_x = (grid.x_lims[2] - grid.x_lims[1]) / grid.x_step
    n_cells_y = (grid.y_lims[2] - grid.y_lims[1]) / grid.y_step
    cell_z, rem = divrem(cell_index, n_cells_x*n_cells_y)
    cell_y, cell_x = divrem(rem, n_cells_x)

    return cell_x * grid.x_step + grid.x_lims[1], cell_y * grid.y_step + grid.y_lims[1], cell_z * grid.z_step + grid.z_lims[1]
end

get_cell_coords(2, grid)

function walk_grid(pos, dir, grid::RegularGrid3D{T}) where T
    grid_pos = get_grid_coord(pos, grid)
    
    cell_index = get_cell_index_from_grid_coord(floor.(grid_pos), grid)

    grid_pos_along_x = grid_pos[1] - floor(grid_pos[1])
    grid_pos_along_y = grid_pos[2] - floor(grid_pos[2])
    grid_pos_along_z = grid_pos[3] - floor(grid_pos[3])

    eps = T(1E-10)

    step_to_edge_x = dir[1] > 0 ? (1 - grid_pos_along_x) / dir[1] : grid_pos_along_x / abs(dir[1]) + eps
    step_to_edge_y = dir[2] > 0 ? (1 - grid_pos_along_y) / dir[2] : grid_pos_along_y / abs(dir[2]) + eps
    step_to_edge_z = dir[3] > 0 ? (1 - grid_pos_along_z) / dir[3] : grid_pos_along_z / abs(dir[3]) + eps

    if (step_to_edge_x < step_to_edge_y) && (step_to_edge_x < step_to_edge_z)
        return cell_index, step_to_edge_x * grid.x_step
    elseif (step_to_edge_y < step_to_edge_x) && (step_to_edge_y < step_to_edge_z)
        return cell_index, step_to_edge_y * grid.y_step
    else
        return cell_index, step_to_edge_z * grid.z_step
    end

end




grid = RegularGrid3D((min_x, max_x), cell_width_x, (min_y, max_y), cell_width_y, (min_z, max_z), cell_width_z)

grid_coord = (get_grid_coord(p0, grid))
grid_coord = floor.(grid_coord)
pgrid = get_world_coord(grid_coord, grid)

pos_xy = reduce(hcat, [first(line).shape.position[1:2] for line in detector])
pos_xz = reduce(hcat, [mod.shape.position[[1, 3]] for line in detector for mod in line ])


fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
scatter!(ax1, pos_xy)
vlines!(ax1, grid_x)
hlines!(ax1, grid_y)

scatter!(ax1, p0[1], p0[2])
scatter!(ax1, pgrid[1], pgrid[2])

scatter!(ax2, pos_xz)
vlines!(ax2, grid_x)
hlines!(ax2, grid_z)

scatter!(ax2, p0[1], p0[3])
scatter!(ax2, pgrid[1], pgrid[3])

p1 = hcat(p0, p0 + 500*dir)

lines!(ax1, p1[1:2, :], color=:black)
lines!(ax2, p1[[1, 3], :], color=:black)
fig

pwalk = p0
for i in 1:10
    cell_index, dist_to_edge = walk_grid(pwalk, dir, grid)
    println(cell_index)
    pwalk += dist_to_edge * dir
    scatter!(ax1, pwalk[1], pwalk[2])
    scatter!(ax2, pwalk[1], pwalk[3])

end

for i in 1:170
    coords = get_cell_coords(i, grid)
    text!(ax1, coords[1] + 5, coords[2] + 5, text="$i", markerspace =  :data, fontsize=8)
end
fig



test = pwalk ./ p0
test /= hypot(test...)

hypot(test...)

dir

medium = KM3NeTMediumArca(1f0, 1f0, 0.17f0)

p = Particle(SA_F32[20, 10, 240], SA_F32[0, 1, 0], 0f0, Float32(1E4), 0f0, PEMinus)

wl_range = (300f0, 800f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)
em = ExtendedCherenkovEmitter(p, medium, spectrum)

photons = [PhotonPropagationCuda.initialize_photon_state(em, medium, spectrum.spectral_dist) for _ in 1:100000]

world_size_x = 500
world_size_y = 500
world_size_z = 600


voxel_size_x = 50
voxel_size_y = 50
voxel_size_z = 50

n_voxels_x = floor(Int32, world_size_x / voxel_size_x)
n_voxels_y = floor(Int32, world_size_y / voxel_size_y)
n_voxels_z = floor(Int32, world_size_z / voxel_size_z)


mod_voxel_map = zeros(Int32, n_voxels_x, n_voxels_y, n_voxels_z)
mod_voxel_map[voxel_ix_x+1, voxel_ix_y+1, voxel_ix_z+1] = 1





voxel_ix_x, pos_in_vox_x = divrem(floor(Int32, photons[1].position[1]), voxel_size_x)
voxel_ix_y, pos_in_vox_y = divrem(floor(Int32, photons[1].position[2]), voxel_size_y)
voxel_ix_z, pos_in_vox_z = divrem(floor(Int32, photons[1].position[3]), voxel_size_z)

vox_ix = (voxel_ix_z) * n_voxels_x * n_voxels_y + (voxel_ix_y) * n_voxels_x + voxel_ix_x+1
mod_voxel_map[:][vox_ix]





voxel_id_start_x = photons[1].position[1] % voxel_size_x
voxel_id_start_y = photons[1].position[2] % voxel_size_y
voxel_id_start_z = photons[1].position[3] % voxel_size_z






photons[1].position




function propagate_state(photon_state)


end
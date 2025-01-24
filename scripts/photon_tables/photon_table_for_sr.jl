using PhotonPropagation
using PhysicsTools
using StaticArrays
using CairoMakie
using GeoMakie
using LinearAlgebra
using Rotations
using DataFrames
using Healpix
using JLD2
using SymbolicRegression
using LoopVectorization
using Bumper
using NaNMath

function transform_input(pmt_pos, particle_pos, particle_dir)

    # Rotate the particle position and direction such that the PMT is at the origin and the z-axis is aligned with the PMT
    rot = calc_rot_matrix(pmt_pos, [0, 0, 1])

    part_pos_rot = rot * particle_pos
    part_dir_rot = rot * particle_dir

    # Convert the rotated particle position to cylindrical coordinates
    pos_cyl = cart_to_cyl(part_pos_rot)

    # Calculate Rotation matrix that rotates the particle direction to the xz-plane
    rotm = RotZ(-pos_cyl[2])

    # Apply the rotation matrix to the particle direction
    part_dir_rot_xz = rotm * part_dir_rot
    part_dir_rot_xz_sph = cart_to_sph(part_dir_rot_xz)

    # We dont have to apply the rotation matrix to the particle position as we are only interested in the zenith angle
    part_pos_rot_sph = cart_to_sph(part_pos_rot ./ norm(part_pos_rot))

    return part_pos_rot_sph[1], part_dir_rot_xz_sph[1], part_dir_rot_xz_sph[2]
end


hbc, hbg = make_hit_buffers();

g = 0.95f0
abs_scale = 1f0
sca_scale = 1f0

medium = CascadiaMediumProperties(g, abs_scale, sca_scale)
target = POM(SA_F32[0, 0, 0], UInt16(1))
wl_range = (300.0f0, 800.0f0)

spectrum = make_cherenkov_spectrum(wl_range, medium)

pos_theta = 1.9f0
pos_phi = 0.7f0
dir_theta = 0.8f0
dir_phi = 3.5f0
dir = sph_to_cart(dir_theta, dir_phi)
energy = 5E4

distances = [0.5f0, 1, 3, 5, 10, 15, 20, 30f0, 40, 80, 150]
pos_cos_thetas = -1:0.1:1
all_photons = []
all_photons_maps = []
for dist in distances
    for pcth in pos_cos_thetas
        pos_theta = Float32(acos(pcth))
        pos = dist .* sph_to_cart(pos_theta, pos_phi)

        particle = Particle(
                    pos,
                    dir,
                    0.0f0,
                    Float32(energy),
                    0.0f0,
                    PHadronShower
                )
        source = ExtendedCherenkovEmitter(particle, medium, spectrum)
        setup = PhotonPropSetup(source, target, medium, spectrum, 1)
        photons = propagate_photons(setup, hbc, hbg)

        m = HealpixMap{Float64, RingOrder}(1)
        m.pixels[:] .= 0

        for ph_dir in photons.direction
            ath, aph = vec2ang(ph_dir...)
            pix = ang2pix(m, ath, aph)
            m.pixels[pix] += 1
        end
        

        #push!(all_photons, photons)
        push!(all_photons_maps, (map=m, pos_ct = pcth, dist=dist))
    end
end

jldsave("/home/wecapstor3/capn/capn100h/new_sr_photon_tables.jld2", photon_maps=all_photons_maps)

all_photons_maps = jldopen("/home/wecapstor3/capn/capn100h/new_sr_photon_tables.jld2")["photon_maps"]

import LossFunctions: SupervisedLoss
struct LogLPLoss{P} <: SupervisedLoss end

LogLPLoss(p::Number) = LogLPLoss{p}()

function (loss::LogLPLoss{P})(prediction, target) where {P}
    if prediction <= 0
        return Inf
    end

    return (abs(log(prediction+1) - log(target+1)))^P
end

const LogL2Loss = LogLPLoss{2}

X = reduce(hcat, [data.pos_ct, data.dist] for data in all_photons_maps)
y = [data.map.pixels[1] for data in all_photons_maps]

outdir = "/home/wecapstor3/capn/capn100h/sr_out_test"

square(x) = x^2

n_workers = Threads.nthreads()

opt = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=[exp, NaNMath.acos, tan, sqrt, square],
    populations=3*n_workers,
    population_size=150,
    ncycles_per_iteration=200,
    turbo = true,
    mutation_weights = MutationWeights(optimize=0.005, randomize=0.05),
    bumper = true,
    warmup_maxsize_by=nothing,
    complexity_of_constants=1,
    complexity_of_variables=1,
    complexity_of_operators = [
        (^) => 2,
    ],
    #parsimony = 0.01,
    #adaptive_parsimony_scaling=1000,
    elementwise_loss=LogL2Loss(),
    output_directory = outdir,
    save_to_file = true,
    progress=true,
    #fraction_replaced_hof=0.15,
    #dimensional_constraint_penalty=dimensional_constraint_penalty,
    use_frequency=true,
)

state, hof = equation_search(
    X,
    y,
    niterations=2500,
    options=opt,
    parallelism=:multithreading,
    variable_names=["pos_ct", "dist"],
    #X_units=X_units,
    runtests=false,
    return_state=true,
    #saved_state=state,
    #logger=logger
)

dominating = calculate_pareto_frontier(hof)
y_eval, _ = eval_tree_array(dominating[28].tree, X, opt)
y_eval_rs = reshape(y_eval, (length(pos_cos_thetas), length(distances), ))
y_rs = reshape(y, (length(pos_cos_thetas), length(distances), ))

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, distances, y_rs[4, :])
lines!(ax, distances, y_eval_rs[4, :])
fig


fig = Figure()
ax = Axis(fig[1, 1])
pos = 10 .*  sph_to_cart(pos_theta, pos_phi)

dir2d = [dir[1], dir[2]]
dir2d ./= norm(dir2d) / 15

arrows!(ax, [pos[1]], [pos[2]], [dir2d[1]], [dir2d[2]])
arc!(Point2f(0), 0.3, -π, π)
fig
m = HealpixMap{Float64, RingOrder}(1)
dir_pix = pix2ang(m, 7)
arrows!(ax, [0], [0], [dir_pix[1]], [dir_pix[2]])
fig

fig = Figure()
ax = Axis(fig[1 ,1], yscale=log10, xscale=log10)
for i in 1:12
    pixel_counts = [map.pixels[i] for map in all_photons_maps]
    lines!(ax, distances, pixel_counts, label="$i")
end
axislegend(ax)
fig

rad2deg(dot(pmt_coords[6], pos ./norm(pos)))

30 .* sph_to_cart(pos_theta, pos_phi)


fig = Figure()
ax = Axis(fig[1 ,1], yscale=log10)
pmt_coords = get_pmt_positions(target, RotMatrix3(I))
for i in 1:16
    pmtc = pmt_coords[i]
    
    cvals = Float64[]
    for dist in distances
        pos = dist .* sph_to_cart(pos_theta, pos_phi)
        t_p_theta, t_d_theta, t_d_phi = transform_input(pmtc, pos, dir)
        push!(cvals, t_p_theta)
    end
    lines!(ax, distances, cvals)
end
fig



m.pixels


dist = 7
pos = dist .* sph_to_cart(pos_theta, pos_phi)

particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PHadronShower
        )
source = ExtendedCherenkovEmitter(particle, medium, spectrum)
setup = PhotonPropSetup(source, target, medium, spectrum, 1)
photons = propagate_photons(setup, hbc, hbg)


nside = 1
m = HealpixMap{Float64, RingOrder}(nside)
m.pixels[:] .= 0

for ph_dir in photons.direction
    ath, aph = vec2ang(ph_dir...)
    pix = ang2pix(m, ath, aph)
    if m.pixels[pix] == UNSEEN
        m.pixels[pix] = 1
    else
        m.pixels[pix] += 1
    end
end

image, mask, anymasked = mollweide(m)

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, image, colorscale=log10)
fig







all_hits = reduce(vcat, all_hits)

pmt_sel = 8
pmt_coords = get_pmt_positions(target, RotMatrix3(I))[pmt_sel]

unique(all_hits[:, :t_dir_phi])

combined = combine(groupby(all_hits[all_hits.pmt_id .== pmt_sel, :], :distance), :total_weight => sum)

lines(combined.distance, combined.total_weight_sum)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, combined.distance, combined.t_pos_theta)
lines!(ax, combined.distance, combined.t_dir_theta)
lines!(ax, combined.distance, combined.t_dir_phi)
fig


pos_sph = reduce(hcat, cart_to_sph.(photons.position ./ target.shape.radius))
fig = Figure()
ax = GeoAxis(fig[1,1])
sp = scatter!(ax, rad2deg.(pos_sph[2, :]), rad2deg.(pos_sph[1, :]) .-90, color=(:black, 0.2), markersize=1)



pos_sph = reduce(hcat, cart_to_sph.(hits.position ./ target.shape.radius))
sp = scatter!(ax, rad2deg.(pos_sph[2, :]), rad2deg.(pos_sph[1, :]) .-90, color=(:red, 0.4), markersize=4)
fig

unique(hits.pmt_id)
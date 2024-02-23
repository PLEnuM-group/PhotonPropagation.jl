using PhotonPropagation
using StaticArrays
using Random
using DataFrames
using Distributions
using PhysicsTools
using JLD2
using Base.Iterators
rng = MersenneTwister(42)
hbc, hbg = make_hit_buffers()

wl_range = (300.0f0, 800.0f0)

g = 0.95f0
target = POM(SA_F32[0, 0, 0], UInt16(1))


function sample_position_sphere(rng, max_radius)
    r = sqrt(rand(rng, Uniform(0, max_radius^2)))
    theta = acos(rand(rng, Uniform(-1, 1)))
    phi = rand(rng, Uniform(0, 2*π))

    pos = r .* sph_to_cart(theta, phi)
    return pos
end

function sample_uniform_direction(rng)

    dir_theta = acos(rand(rng, Uniform(-1, 1)))
    dir_phi = rand(rng, Uniform(0, 2*π))

    dir = sph_to_cart(dir_theta, dir_phi)
    return dir
end



workdir = ENV["ECAPSTOR"]
outfile_c = joinpath(workdir, "surrogate_validation_sets/cascades.jdl2")
jldopen(outfile_c, "w") do file
end

outfile_t = joinpath(workdir, "surrogate_validation_sets/tracks.jdl2")
jldopen(outfile_t, "w") do file
end

types = [:track, :cascade]

for (i, t) in product(1:200, types)
    position = sample_position_sphere(rng, 100)
    direction = sample_uniform_direction(rng)
    energy = 10 .^ rand(rng, Uniform(2, 6))

    ptype = t==:track ? PMuPlus : PEPlus
    plength = t==:track ? 1E4 : 0.

    particle = Particle(position, direction, 0., energy, plength, ptype)
    particle = convert(Particle{Float32}, particle)

    abs_scale = Float32(1 + randn(rng) * 0.05)
    sca_scale = Float32(1 + randn(rng) * 0.05)
    medium = make_cascadia_medium_properties(g, abs_scale, sca_scale)
    spectrum = make_cherenkov_spectrum(wl_range, medium)

    lst = t==:track ? FastLightsabreMuonEmitter : ExtendedCherenkovEmitter
    lightsource = lst(particle, medium, spectrum)

    setup = PhotonPropSetup(lightsource, target, medium, spectrum, 1)

    photons = propagate_photons(setup, hbc, hbg, copy_output=true)
    if nrow(photons) == 0
        continue
    end
    hits = make_hits_from_photons(photons, setup)
    if nrow(hits) == 0
        continue
    end
    calc_pe_weight!(hits, setup)
    outfile = t==:track ? outfile_t : outfile_c

    jldopen(outfile, "r+") do file
        dataset_id =  length(file) + 1
        grp = JLD2.Group(file, "dataset_$dataset_id")
        grp["hits"] = hits
        grp["particle"] = particle
        grp["abs_scale"] = abs_scale
        grp["sca_scale"] = sca_scale
    end
end
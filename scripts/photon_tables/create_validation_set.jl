using PhotonPropagation
using StaticArrays
using Random
using DataFrames
using Distributions
using PhysicsTools
using JLD2
using Base.Iterators
using ProgressBars
using ArgParse
import Base.GC: gc

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


function main(parsed_args)

    jldopen(parsed_args[:o], "w") do file
    end

    rng = MersenneTwister(parsed_args[:s])
    hbc, hbg = make_hit_buffers(Float32, 0.4)
    wl_range = (300.0f0, 800.0f0)
    g = 0.95f0
    target = POM(SA_F32[0, 0, 0], UInt16(1))

    if parsed_args[:mode] == :lightsabre
        ptype = PMuPlus
    elseif parsed_args[:mode] == :extended
        ptype = PEPlus
    else
        ptype = PHadronShower
    end
    plength = parsed_args[:mode] ==:lightsabre ? 1E4 : 0.
    lst = parsed_args[:mode] ==:lightsabre ? FastLightsabreMuonEmitter : ExtendedCherenkovEmitter

    for _ in 1:parsed_args[:n]
        position = sample_position_sphere(rng, 100)
        direction = sample_uniform_direction(rng)
        energy = 10 .^ rand(rng, Uniform(2, 7))

        particle = Particle(position, direction, 0., energy, plength, ptype)
        particle = convert(Particle{Float32}, particle)

        abs_scale = Float32(1 + randn(rng) * 0.05)
        sca_scale = Float32(1 + randn(rng) * 0.05)
        medium = CascadiaMediumProperties(g, abs_scale, sca_scale)
        spectrum = make_cherenkov_spectrum(wl_range, medium)
        
        lightsource = lst(particle, medium, spectrum)

        setup = PhotonPropSetup(lightsource, target, medium, spectrum, 1, photon_scaling=parsed_args[:photon_scaling])

        photons = propagate_photons(setup, hbc, hbg, copy_output=false)
        if nrow(photons) == 0
            continue
        end
        hits = make_hits_from_photons(photons, setup)
        if nrow(hits) == 0
            continue
        end
        calc_pe_weight!(hits, setup)

        jldopen(parsed_args[:o], "r+") do file
            dataset_id =  length(file) + 1
            grp = JLD2.Group(file, "dataset_$dataset_id")
            grp["hits"] = hits
            grp["particle"] = particle
            grp["abs_scale"] = abs_scale
            grp["sca_scale"] = sca_scale
        end
        gc()
    end
end

mode_choices = ["extended", "lightsabre", "hadronic"]

s = ArgParseSettings()
@add_arg_table s begin
    "-o"
    help = "Output file"
    required = true
    "-n"
    help = "Number of events"
    required = true
    arg_type = Int64
    "--mode"
    help = "Simulation Mode;  must be one of " * join(mode_choices, ", ", " or ")
    range_tester = (x -> x in mode_choices)
    required = true
    "-s"
    help = "Seed"
    arg_type = Int64
    required = true
    "--photon_scaling"
    help = "Oversampling ratio"
    arg_type = Float64
    default = 1.
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

main(parsed_args)
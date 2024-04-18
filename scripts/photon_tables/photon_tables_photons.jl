using PhotonPropagation
using StaticArrays
using Random
using DataFrames
using Distributions
using Rotations
using LinearAlgebra
using HDF5
using Sobol
using ArgParse
using PhysicsTools
using JSON3
import Base.GC.gc

include("utils.jl")


function expected_det_photons_extended(setup)

    target = first(setup.targets)
    source = first(setup.sources)

    dist = norm(target.shape.position .- source.position)
    att_len = absorption_length(450., setup.medium)
    n = 0.1
    expected_photons = n*1E9*target.shape.radius^2 / (4*dist^2) * exp(-dist / att_len)

    return expected_photons
end


function run_sim(
    energy,
    distance,
    dir_costheta,
    dir_phi,
    output_fname,
    seed,
    hit_buffer_cpu,
    hit_buffer_gpu,
    mode=:extended,
    g=0.95,
    abs_scale=1.,
    sca_scale=1.
    )

    direction::SVector{3,Float32} = sph_to_cart(acos(dir_costheta), dir_phi)

    if mode == :bare_infinite_track || mode == :lightsabre_muon
        r = direction[2] > 0 ? direction[1] / direction[2] : zero(direction[1])
        ppos = SA_F32[distance/sqrt(1 + r^2), -r*distance/sqrt(1 + r^2), 0]

    else
        ppos = SA_F32[0, 0, distance]
    end

    sim_attrs = Dict(
        "energy" => energy,
        "mode" => string(mode),
        "distance" => distance,
        "dir_costheta" => dir_costheta,
        "dir_phi" => dir_phi,
        "seed" => seed,
        "source_pos" => JSON3.write(ppos),
        "g" => g,
        "abs_scale" => abs_scale,
        "sca_scale" => sca_scale
    )

    base_weight = 1.0
    photons = nothing

    setup = make_setup(mode, ppos, direction, energy, seed, g=g, abs_scale=Float32(abs_scale), sca_scale=Float32(sca_scale))

    if mode == :extended
        exp_ph = expected_det_photons_extended(setup)
        if exp_ph > 1E5
            factor = 1E5/exp_ph
            setup.sources[1] = rescale_source(setup.sources[1], factor)
            base_weight /= factor
            println("Expected more then 1E5 photons ($exp_ph) rescale. New base weight: $base_weight")
        end
    end


    failure_mode = -1

    while failure_mode != 0
        prop_source = setup.sources[1]
        
        if prop_source.photons > 1E13
            println("More than 1E13 photons, skipping")
            return nothing
        end

        if failure_mode == 2
            setup.sources[1] = rescale_source(prop_source, 10)
            base_weight /= 10.0
        elseif failure_mode == 1
            setup.sources[1] = rescale_source(prop_source, 0.1)
            base_weight *= 10.0
        end

        try
            photons = propagate_photons(setup, hit_buffer_cpu, hit_buffer_gpu)
        catch e
            if isa(e, PhotonPropOOMException)
                println("Photon prop ran out of memory, rescaling.")
                failure_mode = 1
                continue
            else
                throw(e)
            end
        end
        
        if nrow(photons) < 100
            println("Too few photons, rescaling")
            failure_mode = 2 
            continue        
        end  
        
        failure_mode = 0
    end

    nph_sim = nrow(photons)

    println("Sent: $(setup.sources[1].photons) photons. Received: $(nph_sim)")
    # if more than 1E6 photons make it to the module,
    # take the first 1E6 and scale weights

    n_ph_limit = 100000

    if nph_sim > n_ph_limit
        photons = photons[1:n_ph_limit, :]
        base_weight *= nph_sim / n_ph_limit
    end

    calc_time_residual!(photons, setup)
    transform!(photons, :position => (p -> reduce(hcat, p)') => [:pos_x, :pos_y, :pos_z])
    transform!(photons, :direction => (p -> reduce(hcat, p)') => [:dir_x, :dir_y, :dir_z])
    calc_total_weight!(photons, setup)
    photons[!, :total_weight] .*= base_weight

    save_hdf!(
        output_fname,
        "photons",
        Matrix{Float64}(photons[:, [:tres, :pos_x, :pos_y, :pos_z, :dir_x, :dir_y, :dir_z, :total_weight, :module_id, :wavelength]]),
        sim_attrs)
    return nothing

end

function get_abs_sca_scale(rng=Random.default_rng())
    abs_scale = 1 + randn(rng)*0.05
    sca_scale = 1 + randn(rng)*0.05

    return abs_scale, sca_scale
end
function run_sims(parsed_args)

    #=
    parsed_args = Dict("n_sims"=>1, "n_skip"=>0)
    =#

    n_sims = parsed_args["n_sims"]
    n_skip = parsed_args["n_skip"]
    mode = Symbol(parsed_args["mode"])
    e_min = parsed_args["e_min"]
    e_max = parsed_args["e_max"]
    dist_min = parsed_args["dist_min"]
    dist_max = parsed_args["dist_max"]
    g = parsed_args["g"]

    perturb_medium = parsed_args["perturb_medium"]

    
    hbc, hbg = make_hit_buffers()

    if mode == :extended || mode == :lightsabre

        sobol = skip(
            SobolSeq(
                [log10(e_min), log10(dist_min), -1, 0],
                [log10(e_max), log10(dist_max), 1, 2 * π]),
            n_sims * n_skip)

        for i in 1:n_sims
            abs_scale, sca_scale = perturb_medium ? get_abs_sca_scale() : (1., 1.)
            pars = next!(sobol)
            energy = 10^pars[1]
            distance = Float32(10^pars[2])
            dir_costheta = pars[3]
            dir_phi = pars[4]

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, hbc, hbg, mode, g, abs_scale, sca_scale)
            gc()
        end
    elseif mode == :bare_infinite_track
        sobol = skip(
            SobolSeq(
                [log10(dist_min), -1, 0],
                [log10(dist_max), 1, 2 * π]),
            n_sims * n_skip)

        for i in 1:n_sims
            abs_scale, sca_scale = perturb_medium ? get_abs_sca_scale() : (1., 1.)
            pars = next!(sobol)
            energy = 1E5
            distance = Float32(10^pars[1])
            dir_costheta = pars[2]
            dir_phi = pars[3]

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, hbc, hbg, mode, g, abs_scale, sca_scale)
            gc()
        end
        
    elseif mode == :pointlike_cherenkov
        sobol = skip(
            SobolSeq([log10(dist_min), -1], [log10(dist_max), 1]),
            n_sims * n_skip)

        for i in 1:n_sims
            abs_scale, sca_scale = perturb_medium ? get_abs_sca_scale() : (1., 1.)
            pars = next!(sobol)
            energy = 1E5
            distance = Float32(10^pars[1])
            dir_costheta = pars[2]
            dir_phi = 0

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip,  hbc, hbg, mode, g, abs_scale, sca_scale)
            gc()
        end
    else
        error("Unkown mode: $mode")
    end

end

s = ArgParseSettings()

mode_choices = ["extended", "bare_infinite_track", "pointlike_cherenkov", "lightsabre"]

@add_arg_table s begin
    "--output"
    help = "Output filename"
    arg_type = String
    required = true
    "--n_sims"
    help = "Number of simulations"
    arg_type = Int
    required = true
    "--n_skip"
    help = "Skip multiple of nsims in Sobol sequence"
    arg_type = Int
    required = false
    default = 0
    "--mode"
    help = "Simulation Mode;  must be one of " * join(mode_choices, ", ", " or ")
    range_tester = (x -> x in mode_choices)
    default = "extended"
    "--e_min"
    help = "Minimum energy"
    arg_type = Float64
    required = false
    default = 100.0
    "--e_max"
    help = "Maximum energy"
    arg_type = Float64
    required = false
    default = 1E5
    "--dist_min"
    help = "Minimum distance"
    arg_type = Float64
    required = false
    default = 10.0
    "--dist_max"
    help = "Maximum distance"
    arg_type = Float64
    required = false
    default = 150.0
    "--g"
    help = "Mean scattering angle"
    arg_type = Float32
    required = false
    default = 0.95f0
    "--perturb_medium"
    help = "Randomly sample abs / sca scales"
    action = :store_true
end
parsed_args = parse_args(ARGS, s)


run_sims(parsed_args)

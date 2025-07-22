using PhotonPropagation
using PhysicsTools
using StaticArrays
using LinearAlgebra
using Rotations
using DataFrames
using JLD2
using Distributions
using StatsBase
using Arrow
using ArgParse
using Logging
using NeutrinoTelescopeBase

logger = ConsoleLogger()
global_logger(logger)

hbc, hbg = make_hit_buffers(Float32, 0.3);

function make_particle(mode, pos, dir, energy, medium, spectrum)
    if mode == "em_shower"
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus
        )
        source = ExtendedCherenkovEmitter(particle, medium, spectrum)
    elseif mode == "hadronic_shower"
            particle = Particle(
                pos,
                dir,
                0.0f0,
                Float32(energy),
                0.0f0,
                PHadronShower
            )
            source = ExtendedCherenkovEmitter(particle, medium, spectrum)
    elseif mode == "bare_infinite_track"
        length = 400f0
        ppos = pos .- length/2 .* dir
        

        particle = Particle(
            ppos,
            dir,
            0.0f0,
            Float32(energy),
            length,
            PMuMinus
        )

        source = CherenkovTrackEmitter(particle, medium, spectrum)    
    elseif mode == "lightsabre"
        length = 1000f0
        ppos = pos .- length/2 .* dir
        
        particle = Particle(
            Float32.(ppos),
            Float32.(dir),
            0.0f0,
            Float32(energy),
            length,
            PMuMinus
        )

        source = FastLightsabreMuonEmitter(particle, medium, spectrum)

    elseif mode == "pointlike_cherenkov"
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus)
        source = PointlikeChernekovEmitter(particle, medium, spectrum)
    else
        error("unknown mode $mode")
    end

    return source

end


function rand_vec()
    ct = rand(Uniform(-1, 1))
    phi = rand(Uniform(0, 2π))
    return sph_to_cart(acos(ct), phi)
end



function main(args)
    g = 0.95f0
    pwf = 0.2f0
    abs_scale = 1f0
    sca_scale = 1f0

    if !isnothing(args["seed"])
        seed = args["seed"]
    else
        seed = rand(Int64)
    end
    @show seed

    if !args["randomize"] && (isnothing(args["energy"]) || isnothing(args["distance"]))
        error("Need to provide energy and distance when not randomizing.")
    end

    target = nothing
    if args["om_type"] == "POM"
        target = POM(SA_F32[0, 0, 0], UInt16(1))
    else
        target = make_generic_multipmt_om(SA_F32[0, 0, 0], args["om_radius"], 1)
    end

    medium = CascadiaMediumProperties(g, pwf, abs_scale, sca_scale)
    
    wl_range = (300.0f0, 800.0f0)
    spectrum = make_cherenkov_spectrum(wl_range, medium)

    writer = open(Arrow.Writer, args["outfile"])

    for isim in 1:args["nsims"]
        @info "Starting simulation $isim"

        dir = Float32.(rand_vec())

        if !args["randomize"]
            dist = dist
            energy = args["energy"]
        else
            dist = 10 .^rand(Uniform(1, log10(300)))
            energy = 10 .^rand(Uniform(2, 6))
        end

        pos = Float32(dist) * Float32.(rand_vec())

        if mode == :bare_infinite_track || mode == :lightsabre_muon
            r = dir[2] != 0 ? dir[1] / dir[2] : zero(dir[1])
            pos_rot = SA_F32[dist/sqrt(1 + r^2), -r*dist/sqrt(1 + r^2), 0]
            R = rand(RotMatrix3)

            pos = R * pos_rot
            dir = R * dir
        end
        
        source = make_particle(args["mode"], pos, dir, energy, medium, spectrum)

        @info "Energy: $energy \t Photons: $(source.photons)"

        setup = PhotonPropSetup([source], [target], medium, spectrum, seed)

        failure_mode = -1
        base_weight = 1.0
        photons = nothing

        while failure_mode != 0
            prop_source = setup.sources[1]
            
            if prop_source.photons > 1E13
                @warn "More than 1E13 photons, skipping"
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
                photons = propagate_photons(setup, hbc, hbg, copy_output=true)
            catch e
                if isa(e, PhotonPropOOMException)
                    @info "Photon prop ran out of memory, rescaling."
                    failure_mode = 1
                    continue
                else
                    throw(e)
                end
            end
            
            if nrow(photons) < 100
                @info "Too few photons, rescaling"
                failure_mode = 2 
                continue        
            end  
            
            failure_mode = 0
        end
        
        photons[:, :total_weight] .*= base_weight

        select!(photons, Not([:abs_weight, :n_steps, :initial_direction, :dist_travelled]))

        hits = make_hits_from_photons(photons, setup, RotMatrix3(I))
        calc_pe_weight!(hits, [target])

      
        ws = Weights(hits.total_weight)
        per_pmt_counts = counts(Int64.(hits.pmt_id), 1:get_pmt_count(target), ws)


        pos_theta, pos_phi = cart_to_sph(pos ./ norm(pos))
        dir_theta, dir_phi = cart_to_sph(dir)

        Arrow.write(writer, (
            sim_ix=[isim],
            pos=[pos],
            dir=[dir],
            pos_t=[pos_theta],
            pos_phi=[pos_phi],
            dir_t=[dir_theta],
            dir_phi=[dir_phi],
            hits=[per_pmt_counts],
            energy=[energy],
            dist=[dist]))
    end

    close(writer)
end

mode_choices = ["em_shower", "hadronic_shower", "bare_infinite_track", "pointlike_cherenkov", "lightsabre"]
om_choices = ["POM", "generic_multipmt"]
s = ArgParseSettings()
@add_arg_table s begin
    "--outfile", "-o"
        help = "Output file"
        arg_type = String
        required = true
    "--nsims", "-n"
        help = "Number of simulations"
        arg_type = Int
        default = 5
    "--energy", "-e"
        help = "Energy of the particle"
        arg_type = Float64
        required = false
    "--dist", "-d"
        help = "Distance from the module"
        arg_type = Float64
        required = false
    "--randomize"
        help ="Randomize distance and energy"
        action = :store_true
    "--seed", "-s"
        help = "Seed for the simulation"
        arg_type = Int64
    "--mode", "-m"
        help = "Simulation Mode;  must be one of " * join(mode_choices, ", ", " or ")
        range_tester = (x -> x in mode_choices)
        arg_type = String
        required = true
    "--om_type"
        default = "generic_multipmt"
         range_tester = (x -> x in om_choices)
        arg_type = String
        help = "OM Type; must be one of " * join(om_choices, ", ", " or ")
    "--n_pmt"
        default = 20
        arg_type = Int64
        help = "Number of PMTs on module (for generic_multipmt)"
    "--om_radius"
        default = 0.25f0
        arg_type = Float32
        help = "OM Radius (for generic_multipmt)"
end

args = parse_args(s)
res = main(args)

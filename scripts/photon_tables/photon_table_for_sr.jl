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


hbc, hbg = make_hit_buffers(Float32, 0.5);

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
        length = 400f0
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

    medium = CascadiaMediumProperties(g, pwf, abs_scale, sca_scale)
    target = POM(SA_F32[0, 0, 0], UInt16(1))
    wl_range = (300.0f0, 800.0f0)
    spectrum = make_cherenkov_spectrum(wl_range, medium)

    writer = open(Arrow.Writer, args["outfile"])

    for isim in 1:args["nsims"]

        dir = Float32.(rand_vec())
        pos = Float32(args["dist"]) * Float32.(rand_vec())

        if mode == :bare_infinite_track || mode == :lightsabre_muon
            r = dir[2] != 0 ? dir[1] / dir[2] : zero(dir[1])
            pos_rot = SA_F32[args["dist"]/sqrt(1 + r^2), -r*args["dist"]/sqrt(1 + r^2), 0]
            R = rand(RotMatrix3)

            pos = R * pos_rot
            dir = R * dir
        end
        

        source = make_particle(args["mode"], pos, dir, args["energy"], medium, spectrum)

        setup = PhotonPropSetup([source], [target], medium, spectrum, seed)

        failure_mode = -1
        base_weight = 1.0
        photons = nothing

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
                photons = propagate_photons(setup, hbc, hbg, copy_output=true)
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
        
        photons[:, :total_weight] .*= base_weight

        hits = make_hits_from_photons(photons, setup, RotMatrix3(I))
        calc_pe_weight!(hits, [target])

        

        ws = Weights(hits.total_weight)
        per_pmt_counts = counts(Int64.(hits.pmt_id), 1:16, ws)


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
            energy=[args["energy"]],
            dist=[args["dist"]]))
    end

    close(writer)
end

mode_choices = ["em_shower", "hadronic_shower", "bare_infinite_track", "pointlike_cherenkov", "lightsabre"]

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
        required = true
    "--dist", "-d"
        help = "Distance from the module"
        arg_type = Float64
        required = true
    "--seed", "-s"
        help = "Seed for the simulation"
        arg_type = Int64
    "--mode", "-m"
        help = "Simulation Mode;  must be one of " * join(mode_choices, ", ", " or ")
        range_tester = (x -> x in mode_choices)
        arg_type = String
        required = true
end

args = parse_args(s)
res = main(args)

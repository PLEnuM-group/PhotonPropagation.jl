module Processing
using NeutrinoTelescopeBase
using CherenkovMediumBase
using DataFrames
using Rotations
using LinearAlgebra
using StaticArrays
using StructTypes
using Distributions
using PhysicsTools
using StatsBase
using PoissonRandom
using ..Medium
using ..Spectral
using ..Detection
using ..LightYield
using ..PhotonPropagationCuda
using ..Calc
using ..PhotonPropagationSetup

export make_hits_from_photons, propagate_photons
export calc_total_weight!
export calc_number_of_steps
export calc_pe_weight!
export propagate_particles
export resample_hits
export add_pmt_noise!




function calc_total_weight!(df::AbstractDataFrame, setup::PhotonPropSetup)

    if nrow(df) == 0
        return df
    end

    abs_length = absorption_length.(Ref(setup.medium), df[:, :wavelength])
    df[!, :abs_weight] = exp.(-Float64.(df[:, :dist_travelled] ./ abs_length))
    #df[!, :ref_ix] = refractive_index.(df[:, :wavelength], Ref(setup.medium))
    df[!, :total_weight] = df[:, :base_weight] .* df[:, :abs_weight]

    return df
end


function propagate_photons(setup::PhotonPropSetup, steps=15; reserved_memory_fraction=0.7)
    hit_buffer_cpu, hit_buffer_gpu = make_hit_buffers(Float32, reserved_memory_fraction)
    return propagate_photons(setup, hit_buffer_cpu, hit_buffer_gpu, steps)
end

function propagate_photons(setup::PhotonPropSetup, hit_buffer_cpu, hit_buffer_gpu, steps=15; copy_output=false)   

    
    if typeof(setup.targets) <: DetectorLines
        hits, n_ph_sim = run_photon_prop_no_local_cache!(
            setup.sources, setup.targets, setup.medium, setup.spec_dist, setup.seed,
            hit_buffer_cpu, hit_buffer_gpu, n_steps=steps, photon_scaling=setup.photon_scaling
            )
    else
        hits, n_ph_sim = run_photon_prop_no_local_cache!(
            setup.sources, [targ.shape for targ in setup.targets], setup.medium, setup.spec_dist, setup.seed,
            hit_buffer_cpu, hit_buffer_gpu, n_steps=steps, photon_scaling=setup.photon_scaling
            )
    end

    df = DataFrame(hits, copycols=copy_output)

    mod_ids = [targ.module_id for targ in setup.targets]
    df[!, :module_id] .= mod_ids[df[:, :module_id]]
    df[!, :base_weight] .= sum([source.photons for source in setup.sources]) / n_ph_sim
    calc_total_weight!(df, setup)

    return df
end



"""
    function make_hits_from_photons(
        df::AbstractDataFrame,
        setup::PhotonPropSetup,
        target_orientation::AbstractMatrix{<:Real}
    )

Convert photons to pmt_hits.
"""
function make_hits_from_photons(
    df::AbstractDataFrame,
    setup::PhotonPropSetup,
    target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I),
    apply_wl_acc=true)

    return make_hits_from_photons(df, setup.targets, target_orientation, apply_wl_acc)
end


"""
    function make_hits_from_photons(
        df::AbstractDataFrame,
        targets::AbstractArray{<:PhotonTarget},
        target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I))
    )

Convert photons to pmt_hits. Does not yet apply propagation weights.
"""
function make_hits_from_photons(
    df::AbstractDataFrame,
    targets::AbstractArray{<:PhotonTarget},
    target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I),
    apply_wl_acc=true)

    targ_id_map = Dict([target.module_id => target for target in targets])

    if "pos_x" ∉ names(df)
        transform!(df, :position => (p -> reduce(hcat, p)') => [:pos_x, :pos_y, :pos_z])
        transform!(df, :direction => (p -> reduce(hcat, p)') => [:dir_x, :dir_y, :dir_z])
    end

    hits = []
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]

        positions = vec.(eachrow(Matrix(subdf[:, [:pos_x, :pos_y, :pos_z]])))
        directions = vec.(eachrow(Matrix(subdf[:, [:dir_x, :dir_y, :dir_z]])))

        pos::Vector{SVector{3, Float64}} = positions
        dir::Vector{SVector{3, Float64}} = directions
        wl::Vector{Float64} = subdf[:, :wavelength]
        pmt_ids = check_pmt_hit(pos, dir, wl, target, target_orientation)
        mask = pmt_ids .> 0

        if apply_wl_acc
            mask .&= apply_wl_acceptance(pos, dir, wl, target, target_orientation)
        end

        h = DataFrame(copy(subdf[mask, :]))
        h[!, :pmt_id] .= pmt_ids[mask]
        push!(hits, h)

    end

    return reduce(vcat, hits)
end

function resample_hits(hits::AbstractDataFrame, replace=false)

    new_df = DataFrame[]
    for (key, subdf) in pairs(groupby(hits, [:module_id, :pmt_id]))
        w = ProbabilityWeights(subdf[:, :total_weight])
        wsum = sum(w)
        nhits = pois_rand(wsum)
        ixs = sample(1:nrow(subdf), w, nhits, replace=replace)
        push!(new_df, DataFrame(subdf[ixs, :], copycols=true))
    end
    return reduce(vcat, new_df)
end



"""
    calc_number_of_steps(sca_len, cutoff_distance, percentile=0.9)

Calculate the number of photon propagation steps required so that a fraction `percentile`
of all photons has travelled further than cutoff_distance.
"""
function calc_number_of_steps(sca_len, cutoff_distance, percentile=0.9)
    n_steps = ceil(Int64, cutoff_distance / sca_len)
    while true
        d = Erlang(n_steps, sca_len)
        if quantile(d, 1-percentile) > cutoff_distance
            break
        end
        n_steps += 1
    end
    return n_steps
end

function calc_pe_weight!(photons::AbstractDataFrame, targets::AbstractVector{<:PhotonTarget})
    targ_id_map = Dict([target.module_id => target for target in targets])

    if "pos_x" ∉ names(photons)
        transform!(photons, :position => (p -> reduce(hcat, p)') => [:pos_x, :pos_y, :pos_z])
        transform!(photons, :direction => (p -> reduce(hcat, p)') => [:dir_x, :dir_y, :dir_z])
    end

    photons[!, :qe_weight] .= 0.

    for (key, subdf) in pairs(groupby(photons, :module_id))
        target = targ_id_map[key.module_id]

        wl::Vector{Float64} = subdf[:, :wavelength]
        
        qe_weights = apply_qe(wl, target)
        subdf[!, :qe_weight] .= qe_weights
        subdf[!, :total_weight] .*= qe_weights

    end
    return photons
end

function calc_pe_weight!(photons::AbstractDataFrame, setup::PhotonPropSetup)
    targets = setup.targets
    return calc_pe_weight!(photons, targets)
end



function propagate_particles(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, seed, medium::MediumProperties, hit_buffer_cpu, hit_buffer_gpu)

    wl_range = (300.0f0, 800.0f0)
    spectrum = make_cherenkov_spectrum(wl_range, medium)

    sources = [particle_shape(p) isa Cascade ?
               ExtendedCherenkovEmitter(convert(Particle{Float32}, p), medium, spectrum) :
               FastLightsabreMuonEmitter(convert(Particle{Float32}, p), medium, spectrum)
               for p in particles]

    targets_c::Vector{POM{Float32}} = convert(Vector{POM{Float32}}, targets)

    photon_setup = PhotonPropSetup(sources, targets_c, medium, spectrum, seed)
    photons = propagate_photons(photon_setup, hit_buffer_cpu, hit_buffer_gpu, copy_output=true)

    if nrow(photons) == 0
        return nothing
    end
    calc_time_residual!(photons, photon_setup)

    rot = RotMatrix3(I)

    hits = make_hits_from_photons(photons, photon_setup, rot)
    calc_pe_weight!(hits, photon_setup)
    return hits
end

function propagate_particles(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, seed, medium::MediumProperties)
    hbc, hbg = make_hit_buffers()
    return propagate_particles(particles, targets, seed, medium, hbc, hbg)
end


function add_pmt_noise!(hits::AbstractVector, targets; noise_rate=1E4, time_window=1E4)
    uni = Uniform(-time_window/2, time_window/2)
    noise_rate = noise_rate *1E-9
    n_pmt = get_pmt_count(first(targets))

    data_ix = LinearIndices((1:n_pmt, eachindex(targets)))
    #ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))

    for (pmt_ix, tix) in product(1:n_pmt, eachindex(targets))
        
        data_vec = hits[data_ix[pmt_ix, tix]]
        target = targets[tix]

        noise_hits = pois_rand(noise_rate*time_window)
        
        if noise_hits > 0
            noise_times = rand(uni, noise_hits)

            append!(data_vec, noise_times)
            sort!(data_vec)
        end
    end
end

end
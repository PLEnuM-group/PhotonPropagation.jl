using PhotonPropagation
using StaticArrays
using PhysicsTools
using CairoMakie
using CSV
using DataFrames
using JSON3
using StatsBase
using Rotations
using LinearAlgebra
using Format
buffer_cpu, buffer_gpu = make_hit_buffers();

function json_p_to_particle(json_p)
    return Particle(
        SVector{3, Float32}(vec(json_p[:pos])),
        SVector{3, Float32}(vec(json_p[:dir])),
        Float32(json_p[:time]),
        Float32(json_p[:energy]),
        Float32(0),
        PEPlus
        )
end


mean_sca_angle = 0.95f0
medium = make_cascadia_medium_properties(mean_sca_angle, 1f0, 1.f0)

wl_range = (300f0, 800f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)
target_dummy = DOM(SVector{3, Float32}(SA_F32[0, 0, 0]), 1)
spectrum_biased = make_biased_cherenkov_spectrum(target_dummy.acceptance.int_wl, wl_range, medium)


data_icetray = JSON3.read("/home/wecapstor3/capn/capn100h/events_converted.json")
all_sims = []

for event in data_icetray

    losses = json_p_to_particle.(event[:losses])
    targets_dom = DOM[]
    targets_pom = POM[]
    ix_hit_map = Dict()
    for (i, h) in enumerate(event[:hits])
        tpos = h[:om_position]
        om = DOM(SVector{3, Float32}(tpos), i)
        push!(targets_dom, om)

        om = POM(SVector{3, Float32}(tpos), i)
        push!(targets_pom, om)
        ix_hit_map[i] = h[:total_hits_icetray]
    end

    sources_biased = ExtendedCherenkovEmitter.(losses, Ref(medium), Ref(spectrum_biased))
    sources= ExtendedCherenkovEmitter.(losses, Ref(medium), Ref(spectrum))

    setup = PhotonPropSetup(sources_biased, targets_dom, medium, spectrum_biased, 1)
    @time photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
    hits_dom = make_hits_from_photons(photons, setup, RotMatrix3(I), false)
    #calc_pe_weight!(hits, setup)
    sim_results_dom = combine(groupby(hits_dom, :module_id), :total_weight => sum => :hits_dom)

    setup = PhotonPropSetup(sources, targets_pom, medium, spectrum, 1)
    @time photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
    hits_pom = make_hits_from_photons(photons, setup)
    calc_pe_weight!(hits_pom, setup)

    sim_results_pom = combine(groupby(hits_pom, :module_id), :total_weight => sum => :hits_pom)

    sim_results = innerjoin(sim_results_dom, sim_results_pom, on=:module_id)

    module_ids = sim_results[:, :module_id]

    sim_results[!, :hits_icetray] .= [get(ix_hit_map, Int64.(mid), 0) for mid in module_ids]
    sim_results[!, :hits_ratio_pom] .= sim_results[:, :hits_icetray] ./ sim_results[:, :hits_pom]
    sim_results[!, :hits_ratio_dom] .= sim_results[:, :hits_icetray] ./ sim_results[:, :hits_dom]

    push!(all_sims, sim_results)
end

all_sims = reduce(vcat, all_sims)


fig, ax, h = hist(all_sims[:, :hits_ratio_pom], axis=(; xlabel="Hits ratio (icetray / julia)", ylabel="Counts"),
     bins=0:1:40, label=format("POM Mean: {:.2f}", mean(all_sims[:, :hits_ratio_pom])))
hist!(ax, all_sims[:, :hits_ratio_dom], label=format("DOM Mean: {:.2f}", mean(all_sims[:, :hits_ratio_dom])), bins=0:1:40)
axislegend()
fig
mean(all_sims[:, :hits_ratio_dom])

hist((all_sims[:, :hits_pom] ./ all_sims[:, :hits_dom]), axis=(; xlabel="Hits ratio", ylabel="Counts"))

mean((all_sims[:, :hits_pom] ./ all_sims[:, :hits_dom]))

1/2.5
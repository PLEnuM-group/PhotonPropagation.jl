using PhotonPropagation
using StaticArrays
using PhysicsTools
using CairoMakie
using CSV
using Format
using DataFrames

buffer_cpu, buffer_gpu = make_hit_buffers();

mean_sca_angle = 0.95f0
medium = make_cascadia_medium_properties(mean_sca_angle, 1f0, 1.1f0)

tpos = SA_F32[0f0, 50f0, 50f0]
module_id = 1
target = POM(tpos, module_id)
spectrum = Monochromatic(450f0)

axi = AxiconeEmitter(SA_F32[0,0,0], SA_F32[0, 0, 1], 0f0, Int64(1E10), deg2rad(45f0), deg2rad(0.5f0))
led = CollimatedIsotropicEmitter(SA_F32[0,0,0], 0f0, Int64(1E10), cos(deg2rad(60f0)))

# Setup propagation
seed = 1
z_positions = 50:50:300
gs = [0.9, 0.93, 0.95, 0.99]
hit_stats = []

for g in gs
    medium = make_cascadia_medium_properties(Float32(g), 1f0, 1f0)
    for zpos in z_positions
        target = POM(SA_F32[0f0, 50f0, Float32(zpos)], module_id)
        setup = PhotonPropSetup([axi], [target], medium, spectrum, seed, 1.)
        photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
        hits = make_hits_from_photons(photons, setup)
        calc_pe_weight!(hits, setup)
        hits[!, :g] .= g
        hits[!, :zpos] .= zpos
        push!(hit_stats, hits)
    end
end

hit_stats_same = []

for g in gs
    medium = make_cascadia_medium_properties(Float32(g), 1f0, 1f0)
    for zpos in z_positions
        target = POM(SA_F32[0f0, 0f0, Float32(zpos)], module_id)
        setup = PhotonPropSetup([axi], [target], medium, spectrum, seed, 1.)
        photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
        hits = make_hits_from_photons(photons, setup)
        calc_pe_weight!(hits, setup)
        hits[!, :g] .= g
        hits[!, :zpos] .= zpos
        push!(hit_stats_same, hits)
    end
end

hit_stats_led = []

for g in gs
    medium = make_cascadia_medium_properties(Float32(g), 1f0, 1f0)
    for zpos in z_positions
        target = POM(SA_F32[0f0, 50f0, Float32(zpos)], module_id)
        setup = PhotonPropSetup([led], [target], medium, spectrum, seed, 1.)
        photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
        hits = make_hits_from_photons(photons, setup)
        calc_pe_weight!(hits, setup)
        hits[!, :g] .= g
        hits[!, :zpos] .= zpos
        push!(hit_stats_led, hits)
    end
end

hit_stats_led_same = []

for g in gs
    medium = make_cascadia_medium_properties(Float32(g), 1f0, 1f0)
    for zpos in z_positions
        target = POM(SA_F32[0f0, 0f0, Float32(zpos)], module_id)
        setup = PhotonPropSetup([led], [target], medium, spectrum, seed, 1.)
        photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
        hits = make_hits_from_photons(photons, setup)
        calc_pe_weight!(hits, setup)
        hits[!, :g] .= g
        hits[!, :zpos] .= zpos
        push!(hit_stats_led_same, hits)
    end
end



hit_stats = reduce(vcat, hit_stats)
hit_stats_summary = combine(groupby(hit_stats, [:g, :zpos]), :total_weight => sum)

hit_stats_same = reduce(vcat, hit_stats_same)
hit_stats_same_summary = combine(groupby(hit_stats_same, [:g, :zpos]), :total_weight => sum)

hit_stats_led = reduce(vcat, hit_stats_led)
hit_stats_led_summary = combine(groupby(hit_stats_led, [:g, :zpos]), :total_weight => sum)

hit_stats_led_same = reduce(vcat, hit_stats_led_same)
hit_stats_led_same_summary = combine(groupby(hit_stats_led_same, [:g, :zpos]), :total_weight => sum)



fig = Figure(size=(1000, 1000))
ax = Axis(fig[1, 1], yscale=log10, xlabel="Z-Positions (m)", ylabel="Received Fraction", title="Axicone Different Line")
for (groupn, group) in pairs(groupby(hit_stats_summary, :g))
    lines!(ax, group.zpos, group.total_weight_sum ./ axi.photons, label=format("g: {:.2f}", groupn[1]))
end
axislegend("Mean Scattering Angle")
fig

ax = Axis(fig[1, 2], yscale=log10, xlabel="Z-Positions (m)", ylabel="Received Fraction", title="Axicon Same Line")
for (groupn, group) in pairs(groupby(hit_stats_same_summary, :g))
    lines!(ax, group.zpos, group.total_weight_sum ./ axi.photons, label=format("g: {:.2f}", groupn[1]))
end
fig


ax = Axis(fig[2, 1], yscale=log10, xlabel="Z-Positions (m)", ylabel="Received Fraction", title="LED Different Line")
for (groupn, group) in pairs(groupby(hit_stats_led_summary, :g))
    lines!(ax, group.zpos, group.total_weight_sum ./ axi.photons, label=format("g: {:.2f}", groupn[1]))
end


ax = Axis(fig[2, 2], yscale=log10, xlabel="Z-Positions (m)", ylabel="Received Fraction", title="LED Different Line")
for (groupn, group) in pairs(groupby(hit_stats_led_same_summary, :g))
    lines!(ax, group.zpos, group.total_weight_sum ./ axi.photons, label=format("g: {:.2f}", groupn[1]))
end
fig



fig = Figure()
ax = Axis(fig[1, 1], xlabel="Z-Positions (m)", ylabel="Number of Hits")
for (groupn, group) in pairs(groupby(hit_stats_summary, :g))
    lines!(ax, group.zpos, group.total_weight_sum, label=format("g: {:.2f}", groupn[1]))
end
axislegend("Mean Scattering Angle")
fig

all_photons[1]
z_positions = 0:50:300
all_photons2 = []
medium = make_cascadia_medium_properties(0.9f0, 1f0, 1f0)
for zpos in z_positions
    target = POM(SA_F32[0f0, 50f0, Float32(zpos)], module_id)
    setup = PhotonPropSetup([axi], [target], medium, spectrum, seed, 1.)
    photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
    hits = make_hits_from_photons(photons, setup)
    calc_pe_weight!(hits, setup)
    push!(all_photons2, hits)
end


all_photons3 = []
medium = make_cascadia_medium_properties(0.95f0, 1f0, 1f0)
for zpos in z_positions
    target = POM(SA_F32[0f0, 100f0, Float32(zpos)], module_id)
    setup = PhotonPropSetup([axi], [target], medium, spectrum, seed, 1.)
    photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
    hits = make_hits_from_photons(photons, setup)
    calc_pe_weight!(hits, setup)
    push!(all_photons3, hits)
end

all_photons4 = []
medium = make_cascadia_medium_properties(0.9f0, 1f0, 1f0)
for zpos in z_positions
    target = POM(SA_F32[0f0, 100f0, Float32(zpos)], module_id)
    setup = PhotonPropSetup([axi], [target], medium, spectrum, seed, 1.)
    photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
    hits = make_hits_from_photons(photons, setup)
    calc_pe_weight!(hits, setup)
    push!(all_photons4, hits)
end

fig, ax, _ = lines(
    z_positions,
    [sum(ph.total_weight) for ph in all_photons],
    axis=(; xlabel="Z-Position (m)", ylabel="Number of Hits", yscale=log10),
    label="g=0.95")
lines!(
    ax,
    z_positions,
    [sum(ph.total_weight) for ph in all_photons2],
    label="g=0.9")
axislegend(ax)
fig

fig, ax, _ = lines(z_positions, [sum(ph.total_weight) for ph in all_photons3], axis=(; yscale=log10))
lines!(ax, z_positions, [sum(ph.total_weight) for ph in all_photons4])
fig


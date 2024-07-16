using PhotonPropagation
using StaticArrays
using PhysicsTools
using CairoMakie
using CSV
using DataFrames
using LinearAlgebra
using GLM
using Distributions
using AbstractMediumProperties
using PoissonRandom
using Base.Iterators
buffer_cpu, buffer_gpu = make_hit_buffers();


target_positions = [SA_F32[0, 0, -300], SA_F32[0, 0, -150], SA_F32[0, 0, -100], SA_F32[0, 0, -75], SA_F32[0, 0, -50], SA_F32[0, 50, -150], SA_F32[0, 50, -100], SA_F32[0, 50, -75], SA_F32[0, 50, -50]]
targets = POM.(target_positions, 1:length(target_positions))

sim_abs_scale = [0.7f0, 1f0, 1.5f0, 2f0]
sim_sca_scale = [0.7f0, 1f0, 1.5f0, 2f0]
sim_g = [0.85f0, 0.9f0, 0.92f0, 0.95f0, 0.99f0]
wl = 450f0
spectrum = Monochromatic(wl)

sim_results = DataFrame()
for (abs, sca, g) in product(sim_abs_scale, sim_sca_scale, sim_g)

    medium = CascadiaMediumProperties(g, abs, sca)


    #source = FastLightsabreMuonEmitter(p, medium, spectrum)

    source = PointlikeIsotropicEmitter(
        SA_F32[0, 0, 0],
        0f0,
        Int64(1E9)
    )


    seed = 1
    setup = PhotonPropSetup([source], targets, medium, spectrum, seed, 1.)
    # Run photon propagation
    @time photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
    
    module_id_dist = DataFrame([(module_id=m.module_id, distance= norm(Float64.(m.shape.position))) for m in targets])

    hits_per_mod = combine(groupby(photons, :module_id), :total_weight => sum)
    hits_per_mod = sort(innerjoin(module_id_dist, hits_per_mod, on=:module_id), :distance)
    hits_per_mod[!, :nhits_d_corrected] .= Int64.(pois_rand.(hits_per_mod[:, :total_weight_sum] .* hits_per_mod[:, :distance].^2) )
  
    ols = glm(@formula(nhits_d_corrected ~ distance), hits_per_mod, Poisson(), LogLink())

    push!(sim_results, (data=hits_per_mod, ols=ols, abs=absorption_length(wl, medium), sca=scattering_length(wl, medium), g=g, att_len=1 ./ round.(confint(ols); digits=5)[2, :]))
end

sim_results

sim_results[!, :att_len_mid] = .-mean.(sim_results[!, :att_len])

sim_results[!, :att_len_calc] =  1 ./ (1 ./ sim_results[!, :abs]  + 1 ./ sim_results[!, :sca])

sim_results[!, :att_len_calc_eff] =  1 ./ (1 ./ sim_results[!, :abs]  .+ (1 .-sim_results[!, :g]) ./ sim_results[!, :sca])

sim_results

sel = sim_results[:, ]

sel =groupby(sim_results, :abs)[(;abs=55.4f0)]

lines(sel[:, :sca], (sel[:, :att_len_mid]))

mask= sim_results[:, :att_len_mid] .< 40 .&& sim_results[:, :att_len_mid] .> 20

sim_results[mask, :]

scatter(sim_results[mask, :abs], sim_results[mask, :sca], color=sim_results[mask, :g])


fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10)

    lines!(ax, hits_per_mod[:, :distance], hits_per_mod[:, :nhits_d_corrected])
    fig



absorption_length(wl, medium)
scattering_length(wl, medium)

1/ (1/2 + 1/2)

1 / (1/(scattering_length(wl, medium)) + 1/absorption_length(wl, medium))
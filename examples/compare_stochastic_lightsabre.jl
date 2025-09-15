using PhotonPropagation
using StaticArrays
using PhysicsTools
using CairoMakie
using CSV
using DataFrames
using LinearAlgebra
using StatsBase
using ProposalInterface

buffer_cpu, buffer_gpu = make_hit_buffers();


medium = CascadiaMediumProperties()

wl_range = (300f0, 800f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)

# We first define a `particle` and then convert into a light source
energy = Float32(1E5)
direction = SA_F32[0., 1., 0.]
pos = SA_F32[0, 0, 0]
len = Float32(1E4)
t0 = 0f0
#p = Particle(pos, direction, t0, energy, len, PMuPlus)
p = Particle(pos, direction, t0, energy, len, PMuPlus)

pp = ProposalInterface.make_propagator(PMuPlus)
p_prop, stoch, cont = propagate_muon(p, propagator=pp, seed=2)

source_bare = CherenkovTrackEmitter(p, medium, spectrum)
sources_stoch = ExtendedCherenkovEmitter.(convert.(Particle{Float32}, stoch), Ref(medium), Ref(spectrum))

source_lightsabre = FastLightsabreMuonEmitter(p, medium, spectrum)

det = POM(SA_F32[10, 10, 10], 1)

seed = 1

setup_lightsabre = PhotonPropSetup([source_bare; source_lightsabre], [det], medium, spectrum, seed)
# Run photon propagation
photons_lightsabre = propagate_photons(setup_lightsabre, buffer_cpu, buffer_gpu, copy_output=true)

setup_stoch = PhotonPropSetup([source_bare; sources_stoch], [det], medium, spectrum, seed)

photons_stoch = propagate_photons(setup_stoch, buffer_cpu, buffer_gpu, copy_output=true)

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
bins = 0:10:500
hist!(ax, photons_stoch.time, weights=photons_stoch.total_weight, bins=bins, color=(:blue, 0.7), label="Full Stochastic")
hist!(ax, photons_lightsabre.time, weights=photons_lightsabre.total_weight, bins=bins, color=(:orange, 0.7), label="Lightsabre")
ylims!(ax, 1E-2, 1E3)
axislegend(ax)
fig

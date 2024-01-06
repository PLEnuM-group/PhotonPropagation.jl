### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 6d7c401a-ac83-11ee-24f7-3510dd251473
begin
	using PhotonPropagation
	using StaticArrays
	using PhysicsTools
	using CairoMakie
end

# ╔═╡ 86f89ecd-bbed-4589-991f-c9c04cbdce22
md"""
# Photon Propagation Demo

For photon propagation we need a few different ingredients:
* a photon source
* a detection target
* a medium

Let's set up the medium first.

## Medium setup

A medium has to implement various method related to the optical properties, as well as a few other physical properties.
Convenience constructors for water (with guesstimates of the properties at cascadia basin) as well as ice (deep homogeneous clear ice at Icecube) exist. Here we use the convenience constructor for water.
"""

# ╔═╡ 40c6cec6-108e-4a62-8001-d654c0b5780d
# We can configure the mean (cos) scattering angle (used in the Henyey-Greenstein scattering function), as well as absorption and scattering length scaling.
mean_sca_angle = 0.95f0
abs_scale = 1f0
sca_scale = 1f0
medium = make_cascadia_medium_properties(mean_sca_angle, abs_scale, sca_scale)

# ╔═╡ 022c1191-b77e-42a1-a3eb-c630725880a4
md"""
## Light source setup 

Next, we will configure our light source. Various different lightsources are implemented (see `src/lightyield.jl`).
Let's simulate the light yield of a lightsabre muon (energy losses are averaged over many muon propagations). The `FastLightsabreMuonEmitter` uses an energy dependent
parametrization of PROPOSAL simulations of muons. We also have to provide a spectrum, for which we can use the convenience function `make_cherenkov_spectrum`.
"""

# ╔═╡ 504cfcdd-b2f1-49ca-aac4-875dbae0fa29
# We first define a `particle` and then convert that into a light source
energy = Float32(1E5)
direction = SA_F32[0., 1., 0.]
pos = SA_F32[0, 0, 0]
len = Float32(1E4)
t0 = 0f0
p = Particle(pos, direction, t0, energy, len, PMuPlus)

wl_range = (300f0, 800f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)
source = FastLightsabreMuonEmitter(p, medium, spectrum)


# ╔═╡ 82e1689d-029f-4fb5-ac49-ff7896a5b985
md"""
## Target setup

Finally, we have to provide the target. Here we'll use a P-ONE module.
"""

# ╔═╡ 50c96a7e-fc4f-421b-a586-2c22af2fe53a
position = SA_F32[0f0, 30f0, 30f0]
module_id = 1
target = POM(position, module_id)


# Setup propagation
seed = 1
# Allocate buffers where photons will be stored
buffer_cpu, buffer_gpu = make_hit_buffers()

setup = PhotonPropSetup([source], [target], medium, spectrum, seed)

# Run photon propagation
photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)

# Plot arrival time distribution
hist(photons[:, :time], weight=photons[:, :total_weight], bins=0:10:500)

# Plot arrival spectrum
hist(photons[:, :wavelength], weight=photons[:, :total_weight])

# ╔═╡ 0fe88393-a25b-4d6f-921d-df893b20a595
md"""
## Multiple Sources

we can also propagate multiple sources at the same time. As an example, we will use PROPOSAl to propagate a muon and use the actual losses instead of the lightsabre
approximation.
"""

# ╔═╡ b496274d-6a93-4cc3-a3bb-55ebe47b7148
prop_p, losses = propagate_muon(p)
losses_f32 = convert.(Ref(Particle{Float32}), losses)

# Convert losses into emitters
sources = ExtendedCherenkovEmitter.(losses_f32, Ref(medium), Ref(spectrum))

setup = PhotonPropSetup(sources, [target], medium, spectrum, seed)

# Run photon propagation
photons_no_approx = propagate_photons(setup, buffer_cpu, buffer_gpu, true)

# Plot arrival time distribution
bins = 0:10:500
fig = Figure()
ax = Axis(fig[1,1])
hist!(ax, photons[:, :time], weight=photons[:, :total_weight], bins=bins, label="Lightsabre")
hist!(ax, photons_no_approx[:, :time], weight=photons_no_approx[:, :total_weight], bins=bins, label="Full Losses")
axislegend()
fig

# ╔═╡ Cell order:
# ╠═6d7c401a-ac83-11ee-24f7-3510dd251473
# ╠═86f89ecd-bbed-4589-991f-c9c04cbdce22
# ╠═40c6cec6-108e-4a62-8001-d654c0b5780d
# ╠═022c1191-b77e-42a1-a3eb-c630725880a4
# ╠═504cfcdd-b2f1-49ca-aac4-875dbae0fa29
# ╠═82e1689d-029f-4fb5-ac49-ff7896a5b985
# ╠═50c96a7e-fc4f-421b-a586-2c22af2fe53a
# ╠═0fe88393-a25b-4d6f-921d-df893b20a595
# ╠═b496274d-6a93-4cc3-a3bb-55ebe47b7148

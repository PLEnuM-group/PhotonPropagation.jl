using PhotonPropagation
using StaticArrays
using PhysicsTools
using CairoMakie


"""
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

# We can configure the mean (cos) scattering angle (used in the Henyey-Greenstein scattering function), as well as absorption and scattering length scaling.
mean_sca_angle = 0.95f0
abs_scale = 1f0
sca_scale = 1f0
medium = make_cascadia_medium_properties(mean_sca_angle, abs_scale, sca_scale)

"""
## Light source setup 

Next, we will configure our light source. Various different lightsources are implemented (see `src/lightyield.jl`).
Let's simulate the light yield of a lightsabre muon (energy losses are averaged over many muon propagations). The `FastLightsabreMuonEmitter` uses an energy dependent
parametrization of PROPOSAL simulations of muons. We also have to provide a spectrum, for which we can use the convenience function `make_cherenkov_spectrum`.

"""
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

"""
## Target setup

Finally, we have to provide the target. Here we'll use a P-ONE module.

"""
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


"""
## Multiple Sources

we can also propagate multiple sources at the same time. As an example, we will use PROPOSAl to propagate a muon and use the actual losses instead of the lightsabre
approximation.
"""

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



#################

# Target Shape
module_position = SA[0., 0., 10.]
module_radius = 0.3
active_area = 16 * π * (0.0762)^2
shape = Spherical(module_position, module_radius)

# convert to Float32 for fast computation on gpu
shape = convert(Spherical{Float32}, shape)

# Setup target
target = HomogeneousDetector(shape, active_area, UInt16(1))

# Setup source
pos = SA_F32[0., 0., 0.]
source = PointlikeIsotropicEmitter(pos, 0f0, 100000)

# Setup medium
mean_sca_angle = 0.95f0
medium = make_cascadia_medium_properties(mean_sca_angle)

# Setup spectrum
spectrum = make_monochromatic_spectrum(450f0)

seed = 1

# Setup propagation
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)

# Run propagation

hbc, hbg = make_hit_buffers()

alloc = @allocated photons = propagate_photons(setup, hbc, hbg)
alloc |> Base.format_bytes

# Propagate photons from an EM Cascade
energy = Float32(1E5)
direction = SA_F32[0., 1., 0.]
p = Particle(pos, direction, 0f0, energy, 0f0, PEMinus)

# Wavelength range for Cherenkov emission
wl_range = (200f0, 800f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)
source = ExtendedCherenkovEmitter(p, medium, spectrum)

setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
photons = propagate_photons(setup, hbc, hbg)


# Propagate photons for a lightsabre muon

p = Particle(pos, direction, 0f0, energy, 400f0, PMuPlus)
wl_range = (300f0, 800f0)
source_muon = LightsabreMuonEmitter(p, medium, spectrum)

setup = PhotonPropSetup([source_muon], [target], medium, spectrum, seed)
photons = propagate_photons(setup, hbc, hbg)



positions = [SA_F32[0, 0, 0], SA_F32[-10, 0, 0], SA_F32[10, 0, 0]]
mod_ids = UInt16.(1:3)
modules = POM.(positions, mod_ids)

pos = SA_F32[0., 30, 0.]
direction = SA_F32[0., 1., 0.]
p = Particle(pos, direction, 0f0, Float32(1E4), 0f0, PEPlus)
wl_range = (200f0, 800f0)
medium = make_cascadia_medium_properties(0.95f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)
source = ExtendedCherenkovEmitter(p, medium, spectrum)

setup = PhotonPropSetup([source], modules, medium, spectrum, 1)
photons = propagate_photons(setup, hbc, hbg)


# Propagate photons in water and ice and compare
module_position = SA[0., 0., 10.]
module_radius = 0.3
active_area = 16 * π * (0.0762)^2
shape = Spherical(module_position, module_radius)

# convert to Float32 for fast computation on gpu
shape = convert(Spherical{Float32}, shape)

# Setup target
target = HomogeneousDetector(shape, active_area, UInt16(1))


# Setup source
pos = SA_F32[0., 0., 0.]
source = PointlikeIsotropicEmitter(pos, 0f0, 10000000)


# Setup medium
mean_sca_angle = 0.95f0
medium_water = make_cascadia_medium_properties(mean_sca_angle)
medium_ice = make_homogenous_clearice_properties(Float32)

medium_ice = convert(HomogenousIceProperties{Float32}, medium_ice)

# Setup spectrum
spectrum = Monochromatic(450f0)

seed = 1

# Setup propagation
setup_ice = PhotonPropSetup([source], [target], medium_ice, spectrum, seed)
setup_water = PhotonPropSetup([source], [target], medium_water, spectrum, seed)

sca = scattering_length(300., medium_water)
geo_distance = norm(source.position .- target.shape.position)
cutoff_time = 200
cutoff_distance = geo_distance + cutoff_time*group_velocity(450., medium_water)
calc_number_of_steps(sca, cutoff_distance, 0.999)



# Run propagation
photons_ice = propagate_photons(setup_ice, hbc, hbg, 31)
photons_water = propagate_photons(setup_water, hbc, hbg, 10)

bins = 40:0.5:80

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)

hist!(ax, photons_ice[:, :time], weight=photons_ice[:, :total_weight], bins=bins, fillto=1)
hist!(ax, photons_water[:, :time], weight=photons_water[:, :total_weight], bins=bins, fillto=1)
ylims!(1, 1E4)
#hist!(ax, photons_water[:, :time], weight=photons_water[:, :total_weight], bins=bins)
fig


# Propagate photons to a DOM and plot detected photons
module_position = SA_F32[0., 0., 10.]
target = DOM(module_position, 1)
energy = Float32(1E5)

theta = deg2rad(30f0)
phi = deg2rad(30f0)
direction = sph_to_cart(theta, phi)
pos = SA_F32[0., 0., 0.]
p = Particle(pos, direction, 0f0, energy, 400f0, PMuPlus)
wl_range = (300f0, 800f0)

medium_ice = make_homogenous_clearice_properties(Float32)
spectrum = make_cherenkov_spectrum(wl_range, medium_ice)
spectrum_biased = make_biased_cherenkov_spectrum(target.acceptance.int_wl, wl_range, medium_ice)

source_muon = LightsabreMuonEmitter(p, medium_ice, spectrum)
source_muon_biased = LightsabreMuonEmitter(p, medium_ice, spectrum_biased)

seed = 1

setup = PhotonPropSetup([source_muon], [target], medium_ice, spectrum, seed)
hbc, hbg = make_hit_buffers()
photons = propagate_photons(setup, hbc, hbg)
hits = make_hits_from_photons(photons, setup, RotMatrix3(I), true)


setup_biased = PhotonPropSetup([source_muon_biased], [target], medium_ice, spectrum_biased, seed)
photons_biased = propagate_photons(setup_biased, hbc, hbg)
hits_biased = make_hits_from_photons(photons_biased, setup, RotMatrix3(I), false)


zen = cos.(hits[:, :dir_z])


fig, ax, _ = hist(cos.(hits[:, :dir_z]), weights=hits[:, :total_weight], )
hist!(ax, cos.(hits_biased[:, :dir_z]), weights=hits_biased[:, :total_weight])
fig

bins = 0:20.:300
fig, ax, _ = hist(hits[:, :time], weights=hits[:, :total_weight], bins=bins)
hist!(ax, hits_biased[:, :time], weights=hits_biased[:, :total_weight], bins=bins)
fig






# Save output
#=
hits = make_hits_from_photons(photons, setup)
event_record = Dict(:hits=>hits, :sources=>[source], :event_id=>uuid4())

using HDF5

fid = h5open("test.hd5", "w")

fid["test"] = hits[:, [:time]]


save_event("test.hd5", event_record)
=#
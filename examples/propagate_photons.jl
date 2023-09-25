using PhotonPropagation
using StaticArrays
using PhysicsTools
using CUDA
using JSON3
using CairoMakie
using Distributions
using LinearAlgebra
using Rotations

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

alloc = @allocated photons = propagate_photons(setup)
alloc / 1024 / 1024

rand(setup.spec_dist)

hbc, hbg = make_hit_buffers()

alloc = @allocated photons = propagate_photons(setup, hbc, hbg)
alloc / 1024 / 1024

CUDA.reclaim()

# Propagate photons from an EM Cascade
energy = Float32(1E5)
direction = SA_F32[0., 1., 0.]
p = Particle(pos, direction, 0f0, energy, 0f0, PEMinus)

# Wavelength range for Cherenkov emission
wl_range = (200f0, 800f0)
source = ExtendedCherenkovEmitter(p, medium, wl_range)

spectrum = CherenkovSpectrum(wl_range, medium)
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
photons = propagate_photons(setup)


# Propagate photons for a lightsabre muon

p = Particle(pos, direction, 0f0, energy, 400f0, PMuPlus)
wl_range = (300f0, 800f0)
source_muon = LightsabreMuonEmitter(p, medium, wl_range)

@show source_muon.photons, source.photons


setup = PhotonPropSetup([source_muon], [target], medium, spectrum, seed)
photons = propagate_photons(setup)



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
medium_ice = make_homogenous_clearice_properties()

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
photons_ice = propagate_photons(setup_ice, 31)
photons_water = propagate_photons(setup_water, 10)

bins = 40:0.5:80

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)

hist!(ax, photons_ice[:, :time], weight=photons_ice[:, :total_weight], bins=bins, fillto=1)
hist!(ax, photons_water[:, :time], weight=photons_water[:, :total_weight], bins=bins, fillto=1)
ylims!(1, 1E4)
#hist!(ax, photons_water[:, :time], weight=photons_water[:, :total_weight], bins=bins)
fig


# Propagate photons to a DOM and plot detected photons
module_position = SA[0., 0., 10.]
target = DOM(module_position, 1)
energy = Float32(1E5)

theta = deg2rad(30f0)
phi = deg2rad(30f0)
direction = sph_to_cart(theta, phi)
pos = SA_F32[0., 0., 0.]
p = Particle(pos, direction, 0f0, energy, 400f0, PMuPlus)
wl_range = (300f0, 800f0)

medium_ice = make_homogenous_clearice_properties()
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

fig, ax, _ = hist(cos.(hits[:, :dir_z]), weights=hits[:, :total_weight])
hist!(ax, cos.(hits_biased[:, :dir_z]), weights=hits_biased[:, :total_weight])
fig

fig, ax, _ = hist(hits[:, :time], weights=hits[:, :total_weight])
hist!(ax, hits_biased[:, :time], weights=hits_biased[:, :total_weight])
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
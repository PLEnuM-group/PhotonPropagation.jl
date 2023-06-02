using PhotonPropagation
using StaticArrays
using PhysicsTools
using CUDA
using JSON3


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
position = SA_F32[0., 0., 0.]
source = PointlikeIsotropicEmitter(position, 0f0, 100000)


# Setup medium
mean_sca_angle = 0.99f0
medium = make_cascadia_medium_properties(mean_sca_angle)

# Setup spectrum
spectrum = Monochromatic(450f0)

seed = 1

# Setup propagation
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)

# Run propagation
photons = propagate_photons(setup)


# Propagate photons from an EM Cascade
energy = Float32(1E5)
direction = SA_F32[0., 1., 0.]
p = Particle(position, direction, 0f0, energy, 0f0, PEMinus)

# Wavelength range for Cherenkov emission
wl_range = (200f0, 800f0)
source = ExtendedCherenkovEmitter(p, medium, wl_range)

spectrum = CherenkovSpectrum(wl_range, medium)
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
photons = propagate_photons(setup)


# Save output
#=
hits = make_hits_from_photons(photons, setup)
event_record = Dict(:hits=>hits, :sources=>[source], :event_id=>uuid4())

using HDF5

fid = h5open("test.hd5", "w")

fid["test"] = hits[:, [:time]]


save_event("test.hd5", event_record)
=#
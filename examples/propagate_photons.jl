using PhotonPropagation
using StaticArrays
using PhysicsTools

# Setup target
target = DetectionSphere(
    SA[0., 0., 10.],
    0.21,
    1,
    0.1,
    UInt16(1))

# convert to Float32 for fast computation on gpu
target = convert(DetectionSphere{Float32}, target)


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
hits = propagate_photons(setup)


# Propagate photons from an EM Cascade
energy = Float32(1E5)
direction = SA_F32[0., 1., 0.]
p = Particle(position, direction, 0f0, energy, 0f0, PEMinus)

# Wavelength range for Cherenkov emission
wl_range = (200f0, 800f0)
source = ExtendedCherenkovEmitter(p, medium, wl_range)

spectrum = CherenkovSpectrum(wl_range, medium)
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
hits = propagate_photons(setup)

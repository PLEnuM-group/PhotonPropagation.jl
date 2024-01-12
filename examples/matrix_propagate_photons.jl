using PhotonPropagation
using StaticArrays
using PhysicsTools
using CairoMakie
using StructArrays

mean_sca_angle = 0.95
medium = make_cascadia_medium_properties(mean_sca_angle)

# We first define a `particle` and then convert into a light source
energy = 1E5
direction = SA_F64[0., 1., 0.]
pos = SA_F64[0, 0, 0]
len = 1E4
t0 = 0.
p = Particle(pos, direction, t0, energy, len, PMuPlus)

wl_range = (300f0, 800f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)
source = FastLightsabreMuonEmitter(p, medium, spectrum)

tpos = SA_F32[0f0, 30f0, 5f0]
module_id = 1
target = POM(tpos, module_id)

wl_range = (300., 800.)
spectrum = make_cherenkov_spectrum(wl_range, medium)

tpos = SA_F64[0, 30, 5]
module_id = 1
target = POM(tpos, module_id)

buffer = StructArray{PhotonPropagationCuda.ExtendedPhotonState{Float64}}(undef, 5000000)

hits = StructArray{PhotonHit{Float64, Float64}}(undef, 50000)


@time PhotonPropagationCuda.matrix_propagate_photons(buffer, hits, 1, source, spectrum.spectral_dist, [target.shape], medium, 10)
using HDF5
using DataFrames
using PhysicsTools
using LinearAlgebra
using CairoMakie
using StatsBase
using PhotonPropagation
using StaticArrays
fid = h5open("/home/wecapstor3/capn/capn100h/snakemake/leptoninjector-extended-1.hd5")


inj = "RangedInjector2"
initial = DataFrame(fid[inj]["initial"][:])
nu = DataFrame(fid[inj]["final_1"][:])
casc = DataFrame(fid[inj]["final_2"][:])
prop = DataFrame(fid[inj]["properties"][:])


dir_1 = sph_to_cart.(nu[:, :Direction])
dir_2 = sph_to_cart.(casc[:, :Direction])
dir_init = sph_to_cart.(initial[:, :Direction])

rel_angle = acos.(dot.(dir_1, dir_2))
kin_angle = acos.(dot.(dir_2, dir_init))

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
hist!(ax, rad2deg.(rel_angle))
hist!(ax, rad2deg.(kin_angle))
fig

scatter(rel_angle, log10.(prop[:, :totalEnergy]))

hist(rad2deg.(kin_angle))


loge_bins = 2:0.5:6
bc = 0.5 .* (loge_bins[2:end] .+ loge_bins[1:end-1])
data = log10.(prop[:, :totalEnergy])

stat = Float64[]
for i in eachindex(loge_bins)
    if i == lastindex(loge_bins)
        break
    end

    data_mask = data .>= loge_bins[i] .&& data .< loge_bins[i+1]
    push!(stat, median(rel_angle[data_mask]))
end

fig, ax, _ = scatter(log10.(prop[:, :totalEnergy]), rad2deg.(rel_angle, ))
lines!(ax, bc, rad2deg.(stat))
fig

lines(bc, rad2deg.(stat))


p = Particle

mask = initial[:, :Energy] .> 1000

nu[mask, :]
casc[mask, :]

p1 = Particle(
    SA[0., 0., 0.],
    (sph_to_cart(nu[mask, :][1, :Direction])),
    0.,
    (nu[mask, :][1, :Energy]),
    0.,
    PEPlus)

p2 = Particle(
    SA[0., 0., 0.],
    (sph_to_cart(casc[mask, :][1, :Direction])),
    0.,
    (casc[mask, :][1, :Energy]),
    0.,
    PHadronShower)

medium = make_homogenous_clearice_properties(Float32)
lrad = radiation_length(medium) / material_density(medium) * 10

lparam = LongitudinalParameterisation(4f0, 0.4f0, Float32(lrad))

zs = 0:0.1:20

fig, ax, _ = lines(zs, longitudinal_profile.(zs, Ref(lparam)), label="1E3 GeV",
    ylabel="PDF", title="Longitudinal Profile", dpi=150)
lines!(ax, zs, longitudinal_profile.(p1.energy, zs, Ref(medium), Ref(p1.type)))
lines!(ax, zs, longitudinal_profile.(p2.energy, zs, Ref(medium), Ref(p2.type)))
fig

target = DOM(SA_F32[0., 0., 30.], 1)

hbc, hbg = make_hit_buffers()

spectrum = make_cherenkov_spectrum((300f0, 800f0), medium)
source = ExtendedCherenkovEmitter(p1, medium, spectrum)
setup = PhotonPropSetup([source], [target], medium, spectrum, 1)
photons = propagate_photons(setup, hbc, hbg)
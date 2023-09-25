using PhotonPropagation
using StaticArrays
using CairoMakie
using CSV
using DataFrames
using PhysicsTools
using Rotations
using LinearAlgebra
using Formatting
using CUDA

using Distributions


ppc_dir = "/home/hpc/capn/capn100h/repos/ppc/"
ppc_ice_dir = joinpath(ppc_dir, "ice/deep-homogeneous")
ppc_exe = joinpath(ppc_dir, "ocl/ppc")
ppc_geo = CSV.read(joinpath(ppc_ice_dir, "geo-f2k"), DataFrame, header=[:junk1, :junk2, :x, :y, :z, :string, :dom])


function parse_ppc_ice(model_dir)

    icemodel_dat = CSV.read(joinpath(model_dir, "icemodel.dat"), header=[:depth, :be, :adust, :delta_tau], DataFrame)
    
    alpha, kappa, A, B, D, E = open(joinpath(model_dir, "icemodel.par")) do hdl
        lines = readlines(hdl)
        lines = strip.(lines)

        return parse.(Float64, first.(split.(lines)))
    end
    
    mix, g = open(joinpath(model_dir, "cfg.txt")) do hdl
        lines = readlines(hdl)
        mix = parse(Float64, first(split(lines[4])))
        g = parse(Float64, first(split(lines[5])))
        return (mix, g)
    end



    return HomogenousIceProperties(
        radiation_length=39.652/0.9216,
        mean_scattering_angle=g,
        hg_fraction=mix,
        A_SPICE=A,
        B_SPICE=B,
        D_SPICE=D,
        E_SPICE=E,
        a_dust_400=icemodel_dat[1, :adust],
        b_dust_400=icemodel_dat[1, :be],
        alpha_sca_dust=alpha,
        kappa_abs_dust=kappa,
        deltaTSPICE=icemodel_dat[1, :delta_tau]
    )
     
end

#medium = parse_ppc_ice(ppc_ice_dir)

ppc_geo[!, :z] .= 1948.07 .+ ppc_geo[!, :z]

energy = Float32(3E4)
theta = deg2rad(90f0)
phi = deg2rad(150f0)
direction = sph_to_cart(theta, phi)
#direction = SA_F32[0, 0, 1]


dir_inv = -1 .* direction
inv_theta, inv_phi = cart_to_sph(dir_inv)



pos = SA_F32[-32.96, 80., -15]
p = Particle(pos, direction, 0f0, energy, 0f0, PEMinus)


positions = Matrix(ppc_geo[:, [:x, :y, :z]])
distances = norm.(Ref(pos) .- eachrow(positions))
closest = positions[argmin(distances), :]
min_mod = ppc_geo[argmin(distances), :]

minimum(distances)

module_position = SVector{3}(closest)
target = DOM(module_position, 1)

wl_range = (300f0, 700f0)
medium_ice = parse_ppc_ice(ppc_ice_dir)
spectrum_biased = make_biased_cherenkov_spectrum(target.acceptance.int_wl, wl_range, medium_ice)
source = ExtendedCherenkovEmitter(p, medium_ice, spectrum_biased)

seed = 1

setup = PhotonPropSetup([source], [target], medium_ice, spectrum_biased, seed)

#hbc, hbg = make_hit_buffers()
photons = propagate_photons(setup, 30)
hits = make_hits_from_photons(photons, setup, RotMatrix3(I), false)


f2k = format("""V 2000.1.2
TBEGIN ? ? ?
EM 1 1 1970 0 0 0
TR 1 0 e {} {} {} {} {} 0 {} 0
EE
TEND ? ? ?
END
""", Float64(pos[1]), Float64(pos[2]), Float64(pos[3]), Float64(rad2deg(inv_theta)), Float64(rad2deg(inv_phi)), Float64(energy))

f2k_in = joinpath(ppc_dir, "ocl/f2k_event")

open(f2k_in, "w") do hdl
    write(hdl, f2k)
end

CUDA.reclaim()
proc = pipeline(setenv(`$ppc_exe 0`, ("PPCTABLESDIR"=>ppc_ice_dir,)), stdin=f2k_in)
ppchits = read(proc, String)
ppchits = CSV.read(IOBuffer(join(split(ppchits, "\n")[5:end-4], "\n")), DataFrame, header=[:junk, :string, :dom, :time, :wavelength, :pos_theta, :pos_phi, :dir_theta, :dir_phi])

hits_mod_ppc = ppchits[ppchits[:, :string] .== min_mod[:string] .&& ppchits[:, :dom] .== min_mod[:dom], :]

bins = 50:15:450
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (ns)", ylabel="Photons / Bin")
hist!(hits_mod_ppc[:, :time], bins=bins, label="PPC")
hist!(ax, hits[:, :time],  weights=hits[:, :total_weight], bins=bins, label="PhotonPropagation.jl")
axislegend(ax)
fig


bins = 100:15:200
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Wavelength (nm)", ylabel="PDF")
hist!(ax, hits_mod_ppc[:, :wavelength], bins=bins, normalization=:pdf)
hist!(ax, hits[:, :wavelength],  weights=hits[:, :total_weight], bins=bins, normalization=:pdf)
fig


bins = -1:0.05:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Cos(theta)", ylabel="PDF")
hist!(cos.(hits_mod_ppc[:, :dir_theta]), bins=bins, normalization=:pdf)
hist!(ax, hits[:, :dir_z],  weights=hits[:, :total_weight], bins=bins, normalization=:pdf)
fig


bins = -1:0.05:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Cos(theta)", ylabel="PDF")
hist!(cos.(hits_mod_ppc[:, :dir_theta]), bins=bins, normalization=:pdf)
hist!(ax, photons[:, :dir_z],  weights=photons[:, :total_weight], bins=bins, normalization=:pdf)
hist!(ax, hits[:, :dir_z],  weights=hits[:, :total_weight], bins=bins, normalization=:pdf)
fig

photons[:, :direction] 

rotated = Ref(inv(RotMatrix3(I))) .* photons[:, :direction]  

coszeniths = dot.(rotated, Ref([0, 0, -1]))

hist(coszeniths)

ang_acc = target.acceptance.poly_ang.(coszeniths)

scatter(coszeniths, ang_acc)


rotated = Ref(inv(orientation)) .* hit_directions  
coszeniths = dot.(rotated, Ref([0, 0, -1]))

wl_acc = target.acceptance.int_wl.(hit_wavelengths)

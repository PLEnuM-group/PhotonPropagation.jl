using PhotonPropagation
using StaticArrays
using CairoMakie
using CSV
using DataFrames
using PhysicsTools
using Rotations
using LinearAlgebra
using Format
using CUDA
using Distributions

buffer_cpu, buffer_gpu = make_hit_buffers();


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


    return HomogenousIceProperties{Float32}(
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
        deltaTSPICE=icemodel_dat[1, :delta_tau],
        abs_scale=1.,
        sca_scale=1.
    )
     
end

#medium = parse_ppc_ice(ppc_ice_dir)

ppc_geo[!, :z] .= 1948.07 .+ ppc_geo[!, :z]

ppc_geo

energy = Float32(3E5)
theta = deg2rad(70f0)
phi = deg2rad(160f0)
direction = sph_to_cart(theta, phi)
pos = SA_F32[-32.96, 32.44, -230]
p = Particle(pos, direction, 0f0, energy, 0f0, PEMinus)

show(p)

dir_inv = -1 .* direction
inv_theta, inv_phi = cart_to_sph(dir_inv)






positions = Matrix(ppc_geo[:, [:x, :y, :z]])
distances = norm.(Ref(pos) .- eachrow(positions))
closest = positions[argmin(distances), :]
min_mod = ppc_geo[argmin(distances), :]

min_mod

minimum(distances)

module_position = SVector{3}(closest)
target = DOM(module_position, 1)
wls = 300:1.:700
lines(wls, target.acceptance.int_wl(wls))

target.shape.position

costh = -1:0.01:1
lines(costh, target.acceptance.poly_ang.(costh))

target.acceptance.int_wl(300)

wl_range = (300f0, 700f0)
medium_ice = parse_ppc_ice(ppc_ice_dir)
spectrum_biased = make_biased_cherenkov_spectrum(target.acceptance.int_wl, wl_range, medium_ice)
source = ExtendedCherenkovEmitter(p, medium_ice, spectrum_biased)

seed = 1

setup = PhotonPropSetup([source], [target], medium_ice, spectrum_biased, seed, spectrum_interp_steps=500)

#hbc, hbg = make_hit_buffers()
photons = propagate_photons(setup, buffer_cpu, buffer_gpu, 50, copy_output=true)
hits = make_hits_from_photons(photons, setup, RotMatrix3(I), false);


hits_pos_rel = (eachrow(Matrix(hits[:, [:pos_x, :pos_y, :pos_z]])) .- Ref(target.shape.position)) ./ target.shape.radius

f2k = format("""V 2000.1.2
TBEGIN ? ? ?
EM 1 1 1970 0 0 0
TR 1 0 e {} {} {} {} {} 0 {} 0
EE
TEND ? ? ?
END
""", Float64(pos[1]), Float64(pos[2]), Float64(pos[3]), Float64(rad2deg(inv_theta)), Float64(rad2deg(inv_phi)), Float64(energy))

f2k

f2k_in = joinpath(ppc_dir, "ocl/f2k_event")

open(f2k_in, "w") do hdl
    write(hdl, f2k)
end

CUDA.reclaim()
proc = pipeline(setenv(`$ppc_exe 0`, ("PPCTABLESDIR"=>ppc_ice_dir,)), stdin=f2k_in)
ppchits = read(proc, String)
ppchits = CSV.read(IOBuffer(join(split(ppchits, "\n")[5:end-4], "\n")), DataFrame, header=[:junk, :string, :dom, :time, :wavelength, :dir_theta, :dir_phi, :pos_theta, :pos_phi ])
hits_mod_ppc = ppchits[ppchits[:, :string] .== min_mod[:string] .&& ppchits[:, :dom] .== min_mod[:dom], :]

bins = 100:5:500
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (ns)", ylabel="Photons / Bin")
colors = Makie.wong_colors()
hist!(ax, hits_mod_ppc[:, :time], bins=bins, label="PPC", color=(colors[1], 0.7))
hist!(ax, hits[:, :time],  weights=hits[:, :total_weight], bins=bins, label="PhotonPropagation.jl",color=(colors[2], 0.7))
axislegend(ax)
fig


bins = 300:15:700
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Wavelength (nm)", ylabel="PDF")
hist!(ax, hits_mod_ppc[:, :wavelength], bins=bins, normalization=:pdf, color=(colors[1], 0.7), label="PPC")
hist!(ax, hits[:, :wavelength],  weights=hits[:, :total_weight], bins=bins, normalization=:pdf,  color=(colors[2], 0.7), label="PhotonPropagation.jl")
axislegend(ax)
fig


bins = -1:0.1:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Cos(theta)", ylabel="PDF")
hist!(cos.(hits_mod_ppc[:, :dir_theta]), bins=bins, normalization=:pdf,  color=(colors[1], 0.7), label="PPC")
hist!(ax, hits[:, :dir_z],  weights=hits[:, :total_weight], bins=bins, normalization=:pdf, color=(colors[2], 0.7), label="PhotonPropagation.jl")
#hist!(ax, photons[:, :dir_z],  weights=photons[:, :total_weight], bins=bins, normalization=:pdf, color=(colors[2], 0.7), label="PhotonPropagation.jl")
axislegend(ax)
fig


bins = -1:0.1:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Cos(theta_pos)", ylabel="PDF")
hist!(cos.(hits_mod_ppc[:, :pos_theta]), bins=bins, normalization=:pdf,  color=(colors[1], 0.7), label="PPC")
hist!(ax, .- (hits[:, :pos_z].-target.shape.position[3]) ./ target.shape.radius,  weights=hits[:, :total_weight], bins=bins, normalization=:pdf, color=(colors[2], 0.7), label="PhotonPropagation.jl")
axislegend(ax)
fig


bins = -π:0.2:2*π
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Azimuth_pos", ylabel="PDF")
hist!((hits_mod_ppc[:, :pos_phi]) .+ π, bins=bins, normalization=:pdf,  color=(colors[1], 0.7), label="PPC")
hits_sph = cart_to_sph.(hits_pos_rel)
hist!(ax, getindex.(hits_sph, 2),  weights=hits[:, :total_weight], bins=bins, normalization=:pdf, color=(colors[2], 0.7), label="PhotonPropagation.jl")
axislegend(ax)
fig





hits


bins = -1:0.05:1
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Cos(theta_pos)", ylabel="PDF")
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

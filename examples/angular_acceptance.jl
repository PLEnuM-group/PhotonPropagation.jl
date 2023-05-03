using CSV
using DataFrames
using CairoMakie
using PhysicsTools
using LinearAlgebra
using StatsBase
using PairPlots
using PhotonPropagation
using NeutrinoTelescopes
using Rotations
using PhysicsTools
using StaticArrays
using MultivariateStats
using Glob
using ProgressLogging
using BenchmarkTools
using HDF5
using Distributions
using JSON3
using PhysicalConstants.CODATA2018
using Unitful
using BSplineKit
using Polynomials
import PhotonPropagation.Detection: calc_relative_pmt_coords

begin
    coords = Matrix{Float64}(undef, 2, 16)
    # upper 
    coords[1, 1:4] .= deg2rad(90 - 25)
    coords[2, 1:4] = (range(0; step=π / 2, length=4))

    # upper 2
    coords[1, 5:8] .= deg2rad(90 - 57.5)
    coords[2, 5:8] = (range(π / 4; step=π / 2, length=4))

    # lower 2
    coords[1, 9:12] .= deg2rad(90 + 25)
    coords[2, 9:12] = [π/2, 0, 3*π/2, π]

    # lower
    coords[1, 13:16] .= deg2rad(90 + 57.5)
    coords[2, 13:16] = [π/4, 7/4*π, 5/4*π, 3/4*π]
   
    #=
    R = calc_rot_matrix(SA[0.0, 0.0, 1.0], SA[1.0, 0.0, 0.0])
    
    @views for col in eachcol(coords)
        cart = sph_to_cart(col[1], col[2])
        col[:] .= cart_to_sph((R * cart)...)
    end
    =#

end


function proc_df!(df, h_all, h_hit)
       
    #pos_glas = Matrix(df[:, ["1_x", "1_y", "1_z"]])
    pos_glas = Matrix(df[:, ["in_x", "in_y", "in_z"]])
    pos_glas[(pos_glas[:, 3] .>= 45), :] .-= permutedims([0, 0, 45])
    pos_glas[(pos_glas[:, 3] .<= -45), :] .+= permutedims([0, 0, 45])
    norm_glass = norm.(eachrow(pos_glas))
    pos_glas_normed = pos_glas ./ norm_glass
   
    df[!, :glass_norm_x] .= pos_glas_normed[:, 1]
    df[!, :glass_norm_y] .= pos_glas_normed[:, 2]
    df[!, :glass_norm_z] .= pos_glas_normed[:, 3]

    df[!, :wavelength] .= round.(ustrip(u"nm", PlanckConstant * SpeedOfLightInVacuum ./ ( df[1, :in_E]u"eV")))

    @show df[1, :wavelength]

    in_dir = Matrix(df[:, [:in_px, :in_py, :in_pz]])
    in_pos = Matrix(df[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]])

    #all_pos_dir_hit = []
    for (pmt_ix, pmt_coords_sph) in enumerate(eachcol(coords))
        copy_no = pmt_ix - 1

        pmt_vec = sph_to_cart(pmt_coords_sph)

        sel = (
            (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
            df[:, :out_Volume_CopyNo] .== copy_no .&& 
            df[:, :out_ProcessName] .== "OpAbsorption"
            )

        sel_all = Colon()
       
        pos_dir = reduce(vcat, calc_relative_pmt_coords(pmt_vec, in_pos, in_dir))
        pos_dir_hit_pmt = pos_dir[sel, :]
        pos_dir_hit = pos_dir[sel_all, :]

        h_all_i = StatsBase.fit(Histogram, (df[sel_all, :wavelength], (pos_dir_hit[:, 1]), (pos_dir_hit[:, 3]), pos_dir_hit[:, 4]), (bins_wl, bins_pos_theta, bins_dir_theta, bins_phi))
        h_hit_i = StatsBase.fit(Histogram, (df[sel, :wavelength], (pos_dir_hit_pmt[:, 1]), (pos_dir_hit_pmt[:, 3]), pos_dir_hit_pmt[:, 4]), (bins_wl, bins_pos_theta, bins_dir_theta, bins_phi))

        merge!(h_all, h_all_i)
        merge!(h_hit, h_hit_i)
        #push!(all_pos_dir_hit, pos_dir_hit)
    end
    
    #return reduce(vcat, all_pos_dir_hit)

end


bins_pos_theta = 0:0.1:π/2
bins_dir_theta = 0:0.1:π
bins_phi = 0:0.1:2*π
bins_wl = 250:20:690

h_all = StatsBase.fit(Histogram, ([], [], [], []), (bins_wl, bins_pos_theta, bins_dir_theta, bins_phi))
h_hit = StatsBase.fit(Histogram, ([], [], [], []), (bins_wl, bins_pos_theta, bins_dir_theta, bins_phi))

#sim_path = joinpath(ENV["WORK"], "geant4_pmt")

sim_path = "/home/chrhck/geant4_sims/P-OM photons 30 cm sphere/"

files = glob("*.csv", sim_path)

#df_list = []
@progress for f in files
    df = DataFrame(CSV.File(f))
    proc_df!(df[:, :], h_all, h_hit)
    #push!(df_list, df)
end
#df = reduce(vcat, df_list)


h_ratio = h_hit.weights ./ h_all.weights
h_ratio[h_all.weights .== 0] .= 0

h_ratio_avg = mean(h_ratio, dims=[2, 3, 4])[:]
stairs(bins_wl, [h_ratio_avg; h_ratio_avg[end]])


h_ratio_avg = mean(h_ratio, dims=[1, 3, 4])[:]
stairs(bins_pos_theta, [h_ratio_avg; h_ratio_avg[end]])


h_ratio_avg = mean(h_ratio, dims=[1, 2, 4])[:]
stairs(bins_dir_theta, [h_ratio_avg; h_ratio_avg[end]])

h_ratio_avg = mean(h_ratio, dims=[1, 2, 3])[:]
stairs(bins_phi, [h_ratio_avg; h_ratio_avg[end]])


h_ratio_avg = dropdims(mean(h_ratio, dims=[1, 2]), dims=(1, 2))
heatmap(bins_dir_theta, bins_phi, h_ratio_avg)

h_ratio_avg = dropdims(mean(h_ratio, dims=[1, 3]), dims=(1, 3))
heatmap(bins_pos_theta, bins_phi, h_ratio_avg)

h_ratio_avg = dropdims(mean(h_ratio, dims=[3, 4]), dims=(3, 4))
heatmap(bins_wl, bins_pos_theta, h_ratio_avg)

h_ratio_avg = dropdims(mean(h_ratio, dims=[2, 4]), dims=(2, 4))
heatmap(bins_wl, bins_dir_theta, h_ratio_avg)

h_ratio_avg = dropdims(mean(h_ratio, dims=[2, 3]), dims=(2, 3))
heatmap(bins_wl, bins_phi, h_ratio_avg)



df = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/duran_1mm.csv")))
df2 = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/duran_2mm.csv")))
df3 = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/duran_8mm.csv")))

function make_interp(df)
    sargs = sortperm(df[:, :wavelength])
    y = df[sargs, :transmissivity]
    y[y .== 0] .= 1E-7
    y = log.(y ./ 100)
    itp = interpolate(df[sargs, :wavelength], y, BSplineOrder(3))
    return itp
end


itp = make_interp(df)
itp2 = make_interp(df2)
itp3 = make_interp(df3)

wl_eval = 250:1:1000

ys = [itp.(wl_eval), itp2.(wl_eval), itp3.(wl_eval)]
ys = reduce(hcat, ys)
xs = [1, 2, 8]

polys = Polynomials.fit.(Ref(xs), eachrow(ys), 1) 

p_eval = [p(14) for p in polys]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Wavelength (nm)", ylabel="Transmissivity %")

lines!(ax, wl_eval, exp.(itp.(wl_eval)), label="1mm")
lines!(ax, wl_eval, exp.(itp2.(wl_eval)), label="2mm")
lines!(ax, wl_eval, exp.(itp3.(wl_eval)), label="8mm")
lines!(ax, wl_eval, exp.(p_eval), label="14mm (extp)")

pmt_acc = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/PMTAcc.csv"), header=["wavelength", "acceptance"]))

lines!(ax, pmt_acc[:, :wavelength], pmt_acc[:, :acceptance] , label="PMT")
Legend(fig[1, 2], ax)
fig

wl_ts = DataFrame(wavelength=wl_eval, transmissivity=exp.(p_eval))
wl_ts[!, :lambda_abs] .= -14E-3 ./ log.(wl_ts[!, :transmissivity])


CSV.write(joinpath(@__DIR__, "../assets/duran_wl_acc_14mm.csv"), wl_ts)




xs = 250:1:800
fig, ax = lines(df[sargs, :wavelength], df[sargs, :transmissivity])
lines!(ax, xs, itp.(xs))
xlims!(ax, 200, 1000)
fig

xs = 250:1:800
sargs = sortperm(df2[:, :wavelength])
fig, ax = lines(df2[sargs, :wavelength], df2[sargs, :transmissivity])
lines!(ax, xs, itp2.(xs))
xlims!(ax, 200, 1000)
fig



df = DataFrame(CSV.File(files[1]))
sel = (
    (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube"))

sel_mod = df[:, :out_VolumeName] .!= "water"


sum(sel_mod)
sum(sel)

@show unique(df[:, :out_VolumeName])
4*pi*target_radius^2 * sum(sel) / sum(sel_mod)

16*pmt_area


fname = joinpath(@__DIR__, "../assets/pmt_acc_3d.hd5")
h5open(fname, "w") do fid
    fid["acceptance"] = h_ratio
    attrs(fid)["bin_edges_x"] = JSON3.write(bins_pos_theta)
    attrs(fid)["bin_edges_y"] = JSON3.write(bins_dir_theta)
    attrs(fid)["bin_edges_z"] = JSON3.write(bins_phi)
end

fid = h5open(fname, "r")
h_ratio = fid["acceptance"][:, :, :, :]

heatmap(h_ratio[5, 21, :, :])


# Setup target
targ_pos = SA_F64[0., 5., 20.]

pmt_area = (75e-3 / 2)^2 * π
target_radius = 0.21

target = MultiPMTDetector(
    targ_pos,
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float64),
    UInt16(1)
)

target = convert(MultiPMTDetector{Float32}, target)

target_radius = 0.3
target2 = POM(
    targ_pos,
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float64),
    UInt16(1)
)

target2 = convert(POM{Float32}, target2)

# Setup source
position = SA_F32[0., 0., 0.]
energy = Float32(1E5)
direction = SA_F32[0., 1., 0.]
p = Particle(position, direction, 0f0, energy, 0f0, PEMinus)

# Setup medium
mean_sca_angle = 0.99f0
medium = make_cascadia_medium_properties(mean_sca_angle)

# Wavelength range for Cherenkov emission
wl_range = (200f0, 800f0)
source = ExtendedCherenkovEmitter(p, medium, wl_range)

spectrum = CherenkovSpectrum(wl_range, medium)

seed = 1

# Setup propagation
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
setup2 = PhotonPropSetup([source], [target2], medium, spectrum, seed)

# Run propagation
photons = propagate_photons(setup)

hits = make_hits_from_photons(photons, setup)
hits2 = make_hits_from_photons(photons, setup2)


combine(groupby(hits, :pmt_id), nrow)
combine(groupby(hits2, :pmt_id), nrow)


nsamples = 10000

function sample_uni_sphere(n)

    theta = acos.(rand(Uniform(-1, 1), n))
    phi = rand(Uniform(0, 2*π), n)

    return sph_to_cart.(theta, phi)
end

pos_rel = sample_uni_sphere(100000)
positions =  pos_rel .* Ref(target.radius) .+ Ref(target.position)
directions = sample_uni_sphere(100000)
dir_sph = reduce(hcat, cart_to_sph.(directions))
pos_rel_sph = reduce(hcat, cart_to_sph.(pos_rel))



f = Figure(resolution=(1200,1200))

for pmt_ix in 1:16

    hit = check_pmt_hit(positions, directions, target, RotMatrix3(I)) .== pmt_ix
    hit2 = check_pmt_hit(positions, directions, target2, RotMatrix3(I)) .== pmt_ix

    row, col = divrem(pmt_ix-1, 4)
    row +=1
    col +=1

    g = f[row, col] = GridLayout()
    rowgap!(g, 5)

    axtop = Axis(g[1, 1])
    axbot = Axis(g[2, 1])
    #rowsize!(g, 1, Relative(2))
    #rowsize!(g, 1, Relative(1))

    bins = 0:0.3:pi
    hist!(axtop, pos_rel_sph[1, hit], bins=bins)
    hist!(axtop, pos_rel_sph[1, hit2], bins=bins)

    bins = 0:0.3:2*pi
    hist!(axbot, pos_rel_sph[2, hit], bins=bins)
    hist!(axbot, pos_rel_sph[2, hit2], bins=bins)
end

f

hit = check_pmt_hit(positions, directions, target, RotMatrix3(I)) .== 9
hit2 = check_pmt_hit(positions, directions, target2, RotMatrix3(I)) .== 9
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
using Interpolations
using Distributions
using PhysicalConstants.CODATA2018
using Unitful
using Base.Iterators
using Random
using Cthulhu
using JSON3
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
end


function calc_coordinates!(df)
    pos_in = Matrix{Float64}(df[:, ["in_x", "in_y", "in_z"]])
    norm_in = norm.(eachrow(pos_in))
    pos_in_normed = pos_in ./ norm_in
   
    df[!, :in_norm_x] .= pos_in_normed[:, 1]
    df[!, :in_norm_y] .= pos_in_normed[:, 2]
    df[!, :in_norm_z] .= pos_in_normed[:, 3]

    in_pos_sph = reduce(hcat, cart_to_sph.(eachrow(pos_in)))

    df[!, :in_costheta] .= cos.(in_pos_sph[1, :])
    df[!, :in_phi] .= in_pos_sph[2, :]

    in_p_cart = Matrix{Float64}((df[:, [:in_px, :in_py, :in_pz]]))

    norm_p = norm.(eachrow(in_p_cart))
    in_p_cart_norm = in_p_cart ./ norm_p

    df[!, :in_p_norm_x] .= in_p_cart_norm[:, 1]
    df[!, :in_p_norm_y] .= in_p_cart_norm[:, 2]
    df[!, :in_p_norm_z] .= in_p_cart_norm[:, 3]

    in_p_sph = reduce(hcat, cart_to_sph.(eachrow(in_p_cart)))
    df[!, :p_costheta] .= cos.(in_p_sph[1, :])
    df[!, :p_phi] .= in_p_sph[2, :]
    return df
end



function Detection.calc_relative_pmt_coords(df, pmt_coords_cart)

    in_pos_cart_norm = Matrix{Float64}(df[!, [:in_norm_x, :in_norm_y, :in_norm_z]])
    in_p_cart = Matrix{Float64}(df[!, [:in_p_norm_x, :in_p_norm_y, :in_p_norm_z]])

    return Detection.calc_relative_pmt_coords(in_pos_cart_norm, in_p_cart, pmt_coords_cart)

end


function get_hit_coords(df, pmt_coords_cart, pmt_ix)
    pt, dt, dp = calc_relative_pmt_coords(df, pmt_coords_cart[:, pmt_ix])
    
    hit_pmt::BitVector = (
        (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
        df[:, :out_Volume_CopyNo] .== (pmt_ix-1) .&& 
        df[:, :out_ProcessName] .== "OpAbsorption"
    )

    return DataFrame(pt=(pt[hit_pmt]), dt=dt[hit_pmt], dp=dp[hit_pmt])
end

function filter_outlier(df)
    return df[(df[!, "pt"] .> 0.65) .&& (df[!, "dt"] .> 0.6), :]
end


function fit_pca(fnames, pmt_coords_cart)
    dfs_hi = DataFrame[]
    dfs_low = DataFrame[]

    pmt_groups = [vcat(1:4, 10:13), vcat(5:9, 13:16)]

    for fname in fnames
        df = DataFrame(CSV.File(fname))
        calc_coordinates!(df)
        
        combined = mapreduce(
            pmt_ix -> get_hit_coords(df, pmt_coords_cart, pmt_ix),
            vcat,
            pmt_groups[1])
        push!(dfs_hi, filter_outlier(combined))
        
        combined = mapreduce(
            pmt_ix -> get_hit_coords(df, pmt_coords_cart, pmt_ix),
            vcat,
            pmt_groups[2])
        push!(dfs_low, filter_outlier(combined))

    end
    
    vals_hi = Matrix(reduce(vcat, dfs_hi))'
    vals_low = Matrix(reduce(vcat, dfs_low))'
    
    M_hi = fit(PPCA, vals_hi, method=:em)
    M_low = fit(PPCA, vals_low, method=:em)
    
    return M_hi, M_low
end

function _apply_traf(pca, df, pmt_coords_cart, pmt_ixs)
    all_Y = []
    all_Y_hit = []
    for pmt_ix in pmt_ixs
        hit_pmt = (
            (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
            df[:, :out_Volume_CopyNo] .== (pmt_ix-1) .&& 
            df[:, :out_ProcessName] .== "OpAbsorption"
        )

        rel_coords = hcat(calc_relative_pmt_coords(df, pmt_coords_cart[:, pmt_ix])...)
        Y = predict(pca, rel_coords')
        Y_hit = Y[:, hit_pmt]
        push!(all_Y, Y)
        push!(all_Y_hit, Y_hit)
    end
    all_Y = reduce(hcat, all_Y)
    all_Y_hit = reduce(hcat, all_Y_hit)

    return all_Y, all_Y_hit
end


function make_acceptance_hists(files, pcas, pmt_coords_cart)
    bins_1 = -2.5:0.2:2.2
    bins_2 = -3:0.2:2
    
    pmt_groups = [vcat(1:4, 10:13), vcat(5:9, 13:16)]

    wavelengths = Float64[]
    acceptances = Float64[]

    histograms_all = Dict{Int64, Vector{Histogram}}()
    histograms_hit = Dict{Int64, Vector{Histogram}}()

    for (i, _) in enumerate(pmt_groups)
        histograms_all[i] = Vector{Histogram}(undef, length(files))
        histograms_hit[i] = Vector{Histogram}(undef, length(files))
    end
    
    for (fix, fname) in enumerate(files)
        df = DataFrame(CSV.File(fname))
        calc_coordinates!(df)
    
        for (grp_ix, pmt_grp) in enumerate(pmt_groups)
            all_Y, all_Y_hit = _apply_traf(pcas[grp_ix], df, pmt_coords_cart, pmt_grp)
            histograms_all[grp_ix][fix] = StatsBase.fit(Histogram, (all_Y[1, :], all_Y[2, :] ), (bins_1, bins_2))
            histograms_hit[grp_ix][fix] = StatsBase.fit(Histogram, (all_Y_hit[1, :], all_Y_hit[2, :] ), (bins_1, bins_2))
        end

        hit_pmt::BitVector = (
            (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
            df[:, :out_ProcessName] .== "OpAbsorption"
        )

        total_acceptance = sum(hit_pmt) / nrow(df)
        wl = round(ustrip(u"nm", PlanckConstant * SpeedOfLightInVacuum ./ ( df[1, :in_E]u"eV")))
        push!(wavelengths, wl)
        push!(acceptances, total_acceptance)
        
    end

    function _calc_ratios(h_hit, h_all, acc)
        h_hit_all = sum([h.weights ./ a for (h, a) in zip(h_hit, acc)])
        h_all_all = sum([h.weights for h in h_all])
        hr = h_hit_all ./ h_all_all
        hr[h_all_all .== 0] .= 0        
        return hr    
    end

  
    hist_ratios = Dict{Int64, Matrix{Float64}}()
    for grp_ix in keys(histograms_all)
        hist_ratios[grp_ix] = _calc_ratios(histograms_hit[grp_ix], histograms_all[grp_ix], acceptances)
    end
    
    return hist_ratios, wavelengths, acceptances
end




#sim_path = joinpath(ENV["WORK"], "geant4_pmt")
#sim_path = "/home/chrhck/geant4_sims/P-OM photons 30 cm sphere/"
sim_path = joinpath(ENV["WORK"], "geant4_pmt/30cm_sphere")
files = glob("*.csv", sim_path)
coords_cart = reduce(hcat, sph_to_cart.(eachcol(coords)))

pcas = fit_pca(files, coords_cart)
hist_ratios, wavelengths, acceptances = make_acceptance_hists(files[5:end], pcas, coords_cart)

bins_1 = -2.5:0.2:2.2
bins_2 = -3:0.2:2

fname = joinpath(@__DIR__, "../assets/pmt_acc_2d_pca.hd5")
h5open(fname, "w") do fid
    fid["acc_pmt_grp_1"] = hist_ratios[1]
    fid["acc_pmt_grp_2"] = hist_ratios[2]

    attrs(fid)["PPCA_grp_1"] = JSON3.write(pcas[1])
    attrs(fid)["PPCA_grp_2"] = JSON3.write(pcas[2])

    attrs(fid)["bin_edges_1"] = JSON3.write(bins_1)
    attrs(fid)["bin_edges_2"] = JSON3.write(bins_2)


    fid["wl_acceptance_factor_x"] = wavelengths 
    fid["wl_acceptance_factor_y"] = acceptances
end

h5open(fname, "r") do fid
    @show keys(attrs(fid))
end


# Setup target
targ_pos = SA_F64[0., 5., 20.]
pmt_area = (75e-3 / 2)^2 * π
target_radius = 0.21

shape = Spherical(Float32.(targ_pos), Float32(target_radius))

PROJECT_ROOT = pkgdir(PhotonPropagation)
df = CSV.read(joinpath(PROJECT_ROOT, "assets/PMTAcc.csv"), DataFrame, header=["wavelength", "acceptance"])
acc_pmt_wl = linear_interpolation(df[:, :wavelength], df[:, :acceptance], extrapolation_bc=0.)


target = SphericalMultiPMTDetector(
    shape,
    pmt_area,
    make_pom_pmt_coordinates(Float64),
    acc_pmt_wl,
    UInt16(1)
)

target2 = make_pone_module(targ_pos, UInt16(1))

# Setup source
position = SA_F32[0., 0., 0.]
energy = Float32(5E4)

dir_theta = deg2rad(20)
dir_phi = deg2rad(120)

direction = SVector{3, Float32}(sph_to_cart(dir_theta, dir_phi))
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

sum(photons[:, :total_weight])


combine(groupby(hits, :pmt_id), nrow)
combine(groupby(hits2, :pmt_id), nrow)


nsamples = 10000

function sample_uni_sphere(n)

    theta = acos.(rand(Uniform(-1, 1), n))
    phi = rand(Uniform(0, 2*π), n)

    return sph_to_cart.(theta, phi)
end

pos_rel = sample_uni_sphere(1000000)
positions =  pos_rel .* Ref(target.shape.radius) .+ Ref(target.shape.position)
directions = sample_uni_sphere(1000000)
dir_sph = reduce(hcat, cart_to_sph.(directions))
pos_rel_sph = reduce(hcat, cart_to_sph.(pos_rel))



f = Figure(resolution=(1200,1200))

for pmt_ix in 1:16

    hit = check_pmt_hit(positions, directions, fill(400, length(positions)), ones(length(positions)), target, RotMatrix3(I)) .== pmt_ix
    hit2 = check_pmt_hit(positions, directions, fill(400, length(positions)), ones(length(positions)), target2, RotMatrix3(I)) .== pmt_ix

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
    hist!(axtop, pos_rel_sph[1, hit], bins=bins, normalization=:pdf)
    hist!(axtop, pos_rel_sph[1, hit2], bins=bins, normalization=:pdf)

    bins = 0:0.3:2*pi
    hist!(axbot, pos_rel_sph[2, hit], bins=bins, normalization=:pdf)
    hist!(axbot, pos_rel_sph[2, hit2], bins=bins, normalization=:pdf)
end

f

hit = check_pmt_hit(positions, directions, target, RotMatrix3(I)) .== 9
hit2 = check_pmt_hit(positions, directions, target2, RotMatrix3(I)) .== 9





positions = [get_pmt_positions(target2, RotMatrix3(I))[1]]
directions = [.-get_pmt_positions(target2, RotMatrix3(I))[1]]

check_pmt_hit(positions, directions, [500.], [1], target2, RotMatrix3(I))

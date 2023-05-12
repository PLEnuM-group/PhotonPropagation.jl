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
using JSON3
using PhysicalConstants.CODATA2018
using Unitful
using EvoTrees
using CategoricalArrays
using Base.Iterators
using Random
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
    pos_in = Matrix(df[:, ["in_x", "in_y", "in_z"]])
    norm_in = norm.(eachrow(pos_in))
    pos_in_normed = pos_in ./ norm_in
   
    df[!, :in_norm_x] .= pos_in_normed[:, 1]
    df[!, :in_norm_y] .= pos_in_normed[:, 2]
    df[!, :in_norm_z] .= pos_in_normed[:, 3]

    in_pos_sph = reduce(hcat, cart_to_sph.(eachrow(pos_in)))

    df[!, :in_costheta] .= cos.(in_pos_sph[1, :])
    df[!, :in_phi] .= in_pos_sph[2, :]

    in_p_cart = Matrix((df[:, [:in_px, :in_py, :in_pz]]))

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



function calc_rel_pos(df, pmt_coords_cart)

    in_pos_cart_norm = Matrix(df[!, [:in_norm_x, :in_norm_y, :in_norm_z]])
    in_p_cart = Matrix(df[!, [:in_p_norm_x, :in_p_norm_y, :in_p_norm_z]])


    rel_costheta = dot.(eachrow(in_pos_cart_norm), Ref(pmt_coords_cart))
    in_pos_to_pmt = Ref(pmt_coords_cart) .- eachrow(in_pos_cart_norm)

    proj_pmt_in = in_pos_to_pmt .- eachrow((dot.(in_pos_to_pmt, eachrow(in_pos_cart_norm)) .* in_pos_cart_norm))
    proj_in_dir_inpo = eachrow(in_p_cart) .- eachrow((dot.(eachrow(in_p_cart), eachrow(in_pos_cart_norm)) .* in_pos_cart_norm))

    photon_dir_phi = acos.(dot.(proj_pmt_in, proj_in_dir_inpo) ./(norm.(proj_pmt_in) .* norm.(proj_in_dir_inpo)))
    photon_dir_theta  = acos.(dot.(.-eachrow(in_pos_cart_norm), eachrow(in_p_cart)))

    return rel_costheta, photon_dir_theta, photon_dir_phi

end
#function proc_df_new(df)

function proc_df(df, bins_pos_theta, bins_dir_theta, bins_phi)
       
    #pos_glas = Matrix(df[:, ["1_x", "1_y", "1_z"]])
   
    calc_glas_pos!(df)

    df[!, :wavelength] .= round.(ustrip(u"nm", PlanckConstant * SpeedOfLightInVacuum ./ ( df[1, :in_E]u"eV")))

    in_dir = Matrix(df[:, [:in_px, :in_py, :in_pz]])
    in_pos = Matrix(df[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]])

     
    h_all = StatsBase.fit(Histogram, ([], [], []), (bins_pos_theta, bins_dir_theta, bins_phi))
    h_hit = StatsBase.fit(Histogram, ([], [], []), (bins_pos_theta, bins_dir_theta, bins_phi))

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

        h_all_i = StatsBase.fit(Histogram, (pos_dir_hit[:, 1], (pos_dir_hit[:, 3]), pos_dir_hit[:, 4]), (bins_pos_theta, bins_dir_theta, bins_phi))
        h_hit_i = StatsBase.fit(Histogram, (pos_dir_hit_pmt[:, 1], (pos_dir_hit_pmt[:, 3]), pos_dir_hit_pmt[:, 4]), (bins_pos_theta, bins_dir_theta, bins_phi))

        h_all = merge(h_all, h_all_i)
        h_hit = merge(h_hit, h_hit_i)

    end
  
    return h_all, h_hit, df[1, :wavelength]

end



#sim_path = joinpath(ENV["WORK"], "geant4_pmt")
#sim_path = "/home/chrhck/geant4_sims/P-OM photons 30 cm sphere/"
sim_path = joinpath(ENV["WORK"], "geant4_pmt/30cm_sphere")
files = glob("*.csv", sim_path)

df = DataFrame(CSV.File(files[end],))
calc_coordinates!(df)


function get_hit_coords(df, pmt_ix)
    pmt_coords_sph = coords[:, pmt_ix]
    pmt_coords_cart = sph_to_cart(pmt_coords_sph)
    pt, dt, dp = calc_rel_pos(df, pmt_coords_cart)
    
    hit_pmt = (
        (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
        df[:, :out_Volume_CopyNo] .== (pmt_ix-1) .&& 
        df[:, :out_ProcessName] .== "OpAbsorption"
    )

    return DataFrame(pt=acos.(pt[hit_pmt]), dt=dt[hit_pmt], dp=dp[hit_pmt])
end

function get_hit_prob(df, pmt_ix)
    pmt_coords_sph = coords[:, pmt_ix]
    pmt_coords_cart = sph_to_cart(pmt_coords_sph)
    pt, dt, dp = calc_rel_pos(df, pmt_coords_cart)
    
    hit_pmt = (
        (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
        df[:, :out_Volume_CopyNo] .== (pmt_ix-1) .&& 
        df[:, :out_ProcessName] .== "OpAbsorption"
    )

    sel_all = Colon()

    return DataFrame(pt=acos.(pt[hit_pmt]), dt=dt[hit_pmt], dp=dp[hit_pmt])
end


pmt_groups = [vcat(1:4, 10:13), vcat(5:9, 13:16)]
per_group = []
for grp in pmt_groups
    dfs = []
    for pmt_ix in grp
        push!(dfs, get_hit_coords(df, pmt_ix))
    end
    combined = reduce(vcat, dfs)
    push!(per_group, combined)

    h_all = StatsBase.fit(Histogram, ([], [], []), (bins_pos_theta, bins_dir_theta, bins_phi))
    h_hit = StatsBase.fit(Histogram, ([], [], []), (bins_pos_theta, bins_dir_theta, bins_phi))


end

pairplot(per_group...)

t1 =  get_hit_coords(df, 1)
t2 =  get_hit_coords(df, 5)
t3 =  get_hit_coords(df, 9)
t4 =  get_hit_coords(df, 13)


pairplot(t1,  t3)#, t4)



df[!, :pos_rel_costheta] .= rel_costheta
df[!, :dir_rel_phi] .= photon_dir_phi
df[!, :dir_rel_theta] .= photon_dir_theta





#=
hit_pmt = (
    (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
    # df[:, :out_Volume_CopyNo] .== 0 .&& 
    df[:, :out_ProcessName] .== "OpAbsorption"
)
=#
hit_pmt_ix = zeros(nrow(df))
hit_pmt_ix[hit_pmt] .= df[hit_pmt, :out_Volume_CopyNo] .+ 1

df[!, :pmt_ix] .= hit_pmt_ix
df[!, :class] = categorical(hit_pmt_ix, levels=0:16)
df[!, :hit] .= hit_pmt

df_balanced = df

train_ratio = 0.8
train_indices = randperm(nrow(df_balanced))[1:Int(train_ratio * nrow(df_balanced))]

features = [:pos_rel_costheta,  :dir_rel_phi, :dir_rel_theta, :in_x, :in_y, :in_z, :in_px, :in_py, :in_pz]

train_data = df_balanced[train_indices, :]
eval_data = df_balanced[setdiff(1:nrow(df_balanced), train_indices), :]

x_train, y_train = Matrix(train_data[:, features]), train_data[:, :hit]
x_eval, y_eval = Matrix(eval_data[:, features]), eval_data[:, :hit]


hexbin(df[df[!, :hit], :pos_rel_costheta], df[df[!, :hit], :dir_rel_theta])
hexbin(df[df[!, :hit], :dir_rel_phi], df[df[!, :hit], :dir_rel_theta])



config = EvoTreeRegressor(
    loss=:logistic,
    nrounds=2000,
    eta=0.5,
    max_depth=5,
    lambda=0.4,
    rowsample = 0.8,
    nbins=64,
    T=Float64,
    device="cpu")

model = fit_evotree(config;
    x_train=x_train, y_train=y_train,# w_train=w_train,
    x_eval=x_eval, y_eval=y_eval,
    early_stopping_rounds=10,
    metric = :logloss,
    print_every_n=10)

pred_train = model(x_train)
pred_eval = model(x_eval)

hist(pred_train)

mean((pred_train .> 0.5) .== y_train)
mean((pred_eval .> 0.5) .== y_eval)

mask = y_eval

hist(pred_train[y_train])
hist(pred_eval[y_eval])


pred_train


mean((pred_train .> 0.5) .== y_train)
mean((pred_eval .> 0.5) .== y_eval)
y_train
pred_eval .> 0.5

(pred_train .> 0.5)

mean(idx_eval .== levelcode.(y_eval))
mean(idx_train .== levelcode.(y_train))


bins_pos_theta = -1:0.1:1
bins_dir_theta = -1:0.1:1
bins_phi = 0:0.2:2*π



in




calc_glas_pos!(df)
flange_thickness = 90
df[!, :g_z] = df[:, :g_z] - flange_thickness/2 * sign.(df[:, :g_z])


in_dir = Matrix(df[:, [:in_px, :in_py, :in_pz]])
in_pos = Matrix(df[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]])

pos_glas = Matrix(df[:, ["g_x", "g_y", "g_z"]])

norm_glass = norm.(eachrow(pos_glas))

norm_glass .!== 0
pos_glas_normed = pos_glas ./ norm_glass
opening_angle = deg2rad(14) # asin(75e-3 / 2 / 0.3)

for (pmt_ix, pmt_coords_sph) in enumerate(eachcol(coords))
    copy_no = pmt_ix - 1

    pmt_vec = sph_to_cart(pmt_coords_sph)

    sel = (
        (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
        df[:, :out_Volume_CopyNo] .== copy_no .&& 
        df[:, :out_ProcessName] .== "OpAbsorption"
        )

    glas_sel = (norm_glass .> 0) .&&  (acos.(dot.(eachrow(pos_glas_normed), Ref(pmt_vec))) .< opening_angle)

    @show pmt_ix sum(glas_sel .&& sel) / sum(glas_sel)
end


h_data = []
@progress for f in files
    df = DataFrame(CSV.File(f),)
    push!(h_data, proc_df(df[:, :], bins_pos_theta, bins_dir_theta, bins_phi ))
end


wl_sort = sortperm([h[3] for h in h_data])
h_data = h_data[wl_sort]

# Remove WL with low statistics
h_data = h_data[4:end]

# Total acceptance
global_acc = [sum(h[2].weights) / sum(h[1].weights) for h in h_data]
# Correction is the ratio of total hit probability relative to last wavelength
correction = global_acc ./ global_acc[end]

lines(global_acc)

h_sel_corrected = [h[2].weights / co for (h, co) in zip(h_data, correction)]
h_ratio = sum(h_sel_corrected) ./ sum([h[1].weights for h in h_data])
h_ratio[.!isfinite.(h_ratio)] .= 0


wl_acc_x = [h[3] for h in h_data]
wl_acc_y = correction

fname = joinpath(@__DIR__, "../assets/pmt_acc_3d.hd5")
h5open(fname, "w") do fid
    fid["pos_acceptance"] = h_ratio
    attrs(fid)["bin_edges_x"] = JSON3.write(bins_pos_theta)
    attrs(fid)["bin_edges_y"] = JSON3.write(bins_dir_theta)
    attrs(fid)["bin_edges_z"] = JSON3.write(bins_phi)

    fid["wl_acceptance_factor_x"] = wl_acc_x 
    fid["wl_acceptance_factor_y"] = wl_acc_y
end








h5open(fname, "r") do fid
    h_ratio = fid["pos_acceptance"][:, :, :]
    maxix = argmax(h_ratio)
    heatmap(h_ratio[end-1, :, :])
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

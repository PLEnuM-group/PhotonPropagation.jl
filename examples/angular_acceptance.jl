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
    coords[2, 13:16] = [π/4, 7/4*π, 5/4*π, 3/4*π] # (range(π / 4; step=π / 2, length=4))
end



function calc_relative_pmt_coords(df, pmt_coords)
    pmt_vec = sph_to_cart(pmt_coords[1], pmt_coords[2])

    # Rotate pmt to e_z
    R = calc_rot_matrix(pmt_vec, [0, 0, 1])

    in_dir_rot = [R * v for v in eachrow(Matrix(df[:, [:in_px, :in_py, :in_pz]]))]
    glass_pos_rot = [R * v for v in eachrow(Matrix(df[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]]))]
    
    glass_pos_rot_sph = cart_to_sph.(glass_pos_rot)
    

    # Calculate phi direction relative to glass position 
    # by rotating around e_z
    phi = [cart_to_cyl(x)[2] for x in glass_pos_rot]
    Rs = AngleAxis.(-phi, 0, 0, 1)

    in_dir_rot_rel_ez = Rs .* in_dir_rot
    in_dir_rot_rel_ez_sph = cart_to_sph.(in_dir_rot_rel_ez)
    
    return reduce(hcat, glass_pos_rot_sph), reduce(hcat, in_dir_rot_rel_ez_sph)

end

function proc_df!(df, h_all, h_hit)
       
    pos_glas = Matrix(df[:, ["1_x", "1_y", "1_z"]])
    pos_glas[(pos_glas[:, 3] .>= 45), :] .-= permutedims([0, 0, 45])
    pos_glas[(pos_glas[:, 3] .<= -45), :] .+= permutedims([0, 0, 45])
    norm_glass = norm.(eachrow(pos_glas))
    pos_glas_normed = pos_glas ./ norm_glass
   
    df[!, :glass_norm_x] .= pos_glas_normed[:, 1]
    df[!, :glass_norm_y] .= pos_glas_normed[:, 2]
    df[!, :glass_norm_z] .= pos_glas_normed[:, 3]

    
    for (pmt_ix, pmt_coords_sph) in enumerate(eachcol(coords))
        copy_no = pmt_ix - 1

        pmt_vec = sph_to_cart(pmt_coords_sph)

        sel = (
            (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
            df[:, :out_Volume_CopyNo] .== copy_no
            )

        pos_dir = vcat(calc_relative_pmt_coords(df, pmt_vec)...)
        pos_dir_hit = pos_dir[:, sel]
        h_all_i = fit(Histogram, (cos.(pos_dir[1, :]), cos.(pos_dir[3, :]), pos_dir[4, :]), (bins_costheta, bins_costheta, bins_phi))
        h_hit_i = fit(Histogram, (cos.(pos_dir_hit[1, :]), cos.(pos_dir_hit[3, :]), pos_dir_hit[4, :]), (bins_costheta, bins_costheta, bins_phi))

        merge!(h_all, h_all_i)
        merge!(h_hit, h_hit_i)

    end
    
end


bins_costheta = -1:0.05:1
bins_phi = 0:0.1:π

h_all = fit(Histogram, ([], [], []), (bins_costheta, bins_costheta, bins_phi))
h_hit = fit(Histogram, ([], [], []), (bins_costheta, bins_costheta, bins_phi))

files = glob("*.csv", "/home/chrhck/")

@progress for f in glob("*.csv", "/home/chrhck/")
    df = DataFrame(CSV.File(f))
    proc_df!(df[:, :], h_all, h_hit)
end

h_ratio = h_hit.weights ./ h_all.weights
h_ratio[h_all.weights .== 0] .= 0


heatmap(h_ratio[10, :, :])


function transform_to_tangent(x, theta, phi)
    A = [sin(theta)*cos(phi) sin(theta)*sin(phi) cos(theta);
         cos(theta)*cos(phi) cos(theta)*sin(phi) -sin(theta);
         sin(theta)*sin(phi) sin(theta)*cos(phi) 0]
    return A*x
end



all_x = []
all_sph = []
all_sph_tang = []
fig = Figure(resolution=(1500, 500))
for (pmt_ix, group) in enumerate(grouped)
    pmt_vec = sph_to_cart(coords[1, pmt_ix], coords[2, pmt_ix])
    R = calc_rot_matrix(pmt_vec, [0, 0, 1])

    function _f(x)
        return cart_to_sph(R * x)
    end

    glass_pos_rot_sph = mapreduce(_f, hcat, eachrow(Matrix(group[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]])))
    in_dir_rot_sph = mapreduce(_f , hcat, eachrow(Matrix(group[:, [:in_px, :in_py, :in_pz]])))
    
    glass_pos_rot = reduce(hcat, [R * v for v in eachrow(Matrix(group[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]]))])
    in_dir_rot = reduce(hcat, [R * v for v in eachrow(Matrix(group[:, [:in_px, :in_py, :in_pz]]))])

    #glass_pos_cyl = reduce(hcat, cart_to_cyl.(eachcol(glass_pos_rot)))

    phi = reduce(hcat, cart_to_cyl.(eachcol(glass_pos_rot)))[2, :]
    Rs = AngleAxis.(-phi, 0, 0, 1)

    in_dir_rot_rel_ez = reduce(hcat, Rs .* eachcol(in_dir_rot))
    in_dir_rot_rel_ez_sph = mapreduce(collect, hcat, cart_to_sph.(eachcol(in_dir_rot_rel_ez)))

    #in_dir_tang = reduce(hcat, transform_to_tangent.(eachcol(in_dir_rot), glass_pos_rot_sph[1, :], glass_pos_rot_sph[2, :]))

    x_sph = vcat(glass_pos_rot_sph, in_dir_rot_sph)
    x = vcat(glass_pos_rot, in_dir_rot)
    x_sph_tang = vcat(glass_pos_rot_sph, in_dir_rot_rel_ez_sph)
    push!(all_x, x)
    push!(all_sph, x_sph)
    push!(all_sph_tang, x_sph_tang)

end
fig
X = reduce(hcat, all_x)
X_sph = reduce(hcat, all_sph)
X_sph_tang = reduce(hcat, all_sph_tang)
#X_sph[1, :] = cos.(X_sph[1, :])
#X_sph[3, :] = cos.(X_sph[3, :])


X_sph_tang

df = DataFrame(X_sph', [:pos_theta, :pos_phi, :dir_theta, :dir_phi])

pairplot(df)

#pairplot(DataFrame(X_sph_tang', [:pos_theta, :pos_phi, :dir_x, :dir_y, :dir_z]))

df = DataFrame(X_sph_tang', [:pos_theta, :pos_phi, :dir_theta, :dir_phi])
df[:, :cos_postheta] = cos.(df[:, :pos_theta])
df[:, :cos_dirtheta] = cos.(df[:, :dir_theta])
df[:, :dir_phiabs] = abs.(df[:, :dir_phi] .-π)

fig = Figure()
pairplot(fig[1, 1], df[:, [:cos_postheta, :cos_dirtheta, :dir_phiabs]] => (PairPlots.HexBin(colormap=:magma, colorrange=(0, 10000)), ) )

Colorbar(fig[1, 2], colormap=:magma, colorrange=(0, 10000))
fig

fieldnames(typeof(fig.content[1]))
    #X_sph = Matrix(grouped[1][:, [:glass_theta, :glass_phi, :in_theta, :in_phi]])

fig.content[1].elements

X_tr = X_sph_tang[:, 1:2:end,]
X_te = X_sph_tang[:, 2:2:end,]


M = fit(PCA, X_tr; maxoutdim=4)
Yte = predict(M, X_te)
Xr = reconstruct(M, Yte)
X_te
pairplot(X_tr', Xr')

bins_costheta = -1:0.05:1
bins_phi = 0:0.1:π

lut = Matrix(df[:, [:cos_postheta, :cos_dirtheta, :dir_phiabs]])
h = fit(Histogram, tuple(eachcol(lut)...), (bins_costheta, bins_costheta, bins_phi))

Ah.weights

eigvals(M)

pmt_pos_geant = Vector{Vector{Float64}}()
for (groupkey, group) in pairs()
    @show groupkey
    out_pos = Matrix(group[:, ["out_x", "out_y", "out_z"]])
    out_pos_rot = reduce(hcat, [R * r for r in eachrow(out_pos)])
    avg_pos = vec(sum(out_pos_rot, dims=2)  / size(out_pos_rot, 2))
    
    push!(pmt_pos_geant, collect(avg_pos ./ norm(avg_pos)))
end


pmt_pos_geant = reduce(hcat, pmt_pos_geant)
rad2deg.(mapreduce(collect, hcat, cart_to_sph.(eachcol(pmt_pos_geant))))

pmt_pos_sim = reduce(hcat, [sph_to_cart(x...) for x in eachcol(make_pom_pmt_coordinates(Float64))])

fig = Figure()
ax = Axis3(fig[1, 1], aspect=(1, 1, 1))
scatter!(ax, pmt_pos_geant)
scatter!(ax, pmt_pos_sim)
fig


h_hit = histogram(cos.(hit_pc[:, :]), bins)
h_all = histogram(cos.(hit_pc[1, :]), bins)




pairplot(hit_pc[:, ["dir_theta", "dir_phi", "1_x", "1_y", "1_z"]])



fig, ax = hist(cos.(directions[1, :]))
hist((directions[2, :]))





function histogram(x, bins)
    bin_ixs = searchsortedfirst.(Ref(bins), x) .-1
    return counts(bin_ixs, length(bins)-1)
end


heatmap(cos.(directions[1, :]), out_pos[:, 1])

bins = -1:0.05:1
bin_ixs = searchsortedfirst.(Ref(bins), cos.(directions[1, :])) .-1

h_hit = histogram(cos.(directions[1, :]), bins)
h_all = histogram(cos.(directions_all[1, :]), bins)

h_ratio = h_hit ./h_all

[h_ratio; 0]

stairs(bins, [h_ratio; h_ratio[end]], step=:post)



counts(bin_ixs)
length(bins)

Matrix(df[sel, [:in_x, :in_y, :in_z, :in_px, :in_py, :in_pz]])


p1 = Matrix(df[sel, [:in_x, :in_y, :in_z,]])
pend = Matrix(df[sel, [:out_x, :out_y, :out_z]])
step1 = Matrix(df[sel, [:out_x, :out_y, :out_z]])
dir =  Matrix(df[sel, [:1_x, :1_y, :1_z]])

(pend .- p1) ./ dir
lines(hcat(p1[2, :], step1[2, :], pend[2, :]))

hcat(p1[1, :], step1[1, :], pend[1:10, :])
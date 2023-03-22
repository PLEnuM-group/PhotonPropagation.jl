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


df = DataFrame(CSV.File("/home/chrhck/out00.csv"))
df

directions_all = mapreduce(collect, hcat, cart_to_sph.(eachrow(Matrix(df[:, [:in_px, :in_py, :in_pz]]))))

norm_in = norm.(eachrow(Matrix(df[:, [:in_x, :in_y, :in_z]])))
in_sph = mapreduce(collect, hcat, cart_to_sph.(eachrow(Matrix(df[:, [:in_x, :in_y, :in_z]]) ./ norm_in)))
pos_glas = Matrix(df[:, ["1_x", "1_y", "1_z"]])

pos_glas[(pos_glas[:, 3] .>= 45), :] .-= permutedims([0, 0, 45])
pos_glas[(pos_glas[:, 3] .<= -45), :] .+= permutedims([0, 0, 45])

norm_glass = norm.(eachrow(pos_glas))
pos_glas_normed = pos_glas ./ norm_glass
glass_sph = mapreduce(collect, hcat, cart_to_sph.(eachrow(pos_glas_normed)))

df[!, :dir_theta] .= directions_all[1, :]
df[!, :dir_phi] .= directions_all[2, :]

df[!, :in_theta] .= in_sph[1, :]
df[!, :in_phi] .= in_sph[2, :]

df[!, :glass_theta] .= glass_sph[1, :]
df[!, :glass_phi] .= glass_sph[2, :]

df[!, :glass_norm_x] .= pos_glas_normed[:, 1]
df[!, :glass_norm_y] .= pos_glas_normed[:, 2]
df[!, :glass_norm_z] .= pos_glas_normed[:, 3]


sel = (
    (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube")
    )
hit_pc = df[sel, :]

grouped = groupby(hit_pc, :out_Volume_CopyNo)


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



fig = Figure()
for (i, group) in enumerate(grouped)

    p = Matrix(group[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]])
    q = sph_to_cart(coords[1, i], coords[2, i])
    rel_angle = (dot.(eachrow(p),Ref(q)))
    row, col = divrem(i-1, 4)
    #ax = Axis(fig[row+1, col+1], yscale=log10, limits=(-1.1, 1.1, 1E-1, 1E4))
    #hist!(ax, rel_angle, bins = -1:0.1:1, fillto=1E-1)

    function _f(x)
        return collect(cart_to_sph(x))
    end

    R = calc_rot_matrix(q, [0, 0, 1])
    rot_pos = mapreduce(_f, hcat, [R*x for x in eachrow(p)])

    ax = Axis(fig[row+1, col+1], yscale=log10, limits=(0, 2*π, 1E-1, 1E4))
    hist!(ax, (rot_pos[2, :]), bins = 0:0.1:2*π, fillto=1E-1)
    vlines!(ax, [cart_to_sph(R*q)[2] ]) # coords[2, i]


    #hist!(ax, (rot_pos[2, :]), bins = 0:0.1:2*π, fillto=1E-1)


end
fig

fig = Figure()
for (i, group) in enumerate(grouped)

    p = Matrix(group[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]])
    q = sph_to_cart(coords[1, i], coords[2, i])
    rel_pos_angle = (dot.(eachrow(p),Ref(q)))

    p = Matrix(group[:, [:in_px, :in_py, :in_pz]])
    rel_dir_angle = (dot.(eachrow(p),Ref(q)))
    
    row, col = divrem(i-1, 4)
    ax = Axis(fig[row+1, col+1])
    scatter!(ax, rel_pos_angle, rel_dir_angle)
end
fig



all_x = []
all_sph = []
fig = Figure(resolution=(1500, 500))
for (pmt_ix, group) in enumerate(grouped)
    pmt_vec = sph_to_cart(coords[1, pmt_ix], coords[2, pmt_ix])
    R = calc_rot_matrix(pmt_vec, [0, 0, 1])

    function _f(x)
        return collect(cart_to_sph(R * x))
    end

    glass_pos_rot_sph = mapreduce(_f, hcat, eachrow(Matrix(group[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]])))
    in_dir_rot_sph = mapreduce(_f , hcat, eachrow(Matrix(group[:, [:in_px, :in_py, :in_pz]])))
    
    glass_pos_rot = reduce(hcat, [R * v for v in eachrow(Matrix(group[:, [:glass_norm_x, :glass_norm_y, :glass_norm_z]]))])
    in_dir_rot = reduce(hcat, [R * v for v in eachrow(Matrix(group[:, [:in_px, :in_py, :in_pz]]))])
    x_sph = vcat(glass_pos_rot_sph, in_dir_rot_sph)
    x = vcat(glass_pos_rot, in_dir_rot)
    push!(all_x, x)
    push!(all_sph, x_sph)

end
fig
X = reduce(hcat, all_x)
X_sph = reduce(hcat, all_sph)
#X_sph[1, :] = cos.(X_sph[1, :])
#X_sph[3, :] = cos.(X_sph[3, :])


pairplot(DataFrame(X_sph', [:pos_theta, :pos_phi, :dir_theta, :dir_phi]))

    #X_sph = Matrix(grouped[1][:, [:glass_theta, :glass_phi, :in_theta, :in_phi]])

X_tr = X_sph[:, 1:2:end,]
X_te = X_sph[:, 2:2:end,]


M = fit(KernelPCA, X_tr; maxoutdim=4)
Yte = predict(M, X_te)
Xr = reconstruct(M, Yte)
X_te
pairplot(X_tr', Xr')

bins_theta = 0:0.1:π
bins_phi = 0:0.1:2*π

h = fit(Histogram, tuple(eachrow(X_sph)...), (bins_theta, bins_phi, bins_theta, bins_phi))

h.weights

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
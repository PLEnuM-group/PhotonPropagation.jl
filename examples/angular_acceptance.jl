using CSV
using DataFrames
using CairoMakie
using PhysicsTools
using LinearAlgebra
using StatsBase
using PairPlots

df = DataFrame(CSV.File("/home/chrhck/out00.csv"))
df

directions_all = mapreduce(collect, hcat, cart_to_sph.(eachrow(Matrix(df[:, [:in_px, :in_py, :in_pz]]))))

norm_in = norm.(eachrow(Matrix(df[:, [:in_x, :in_y, :in_z]])))
in_sph = mapreduce(collect, hcat, cart_to_sph.(eachrow(Matrix(df[:, [:in_x, :in_y, :in_z]]) ./ norm_in)))

df[!, :dir_theta] .= directions_all[1, :]
df[!, :dir_phi] .= directions_all[2, :]

df[!, :in_theta] .= in_sph[1, :]
df[!, :in_phi] .= in_sph[2, :]


sel = (
    (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube")
    .&& df[:, :out_Volume_CopyNo] .== 3
    )
hit_pc = df[sel, :]

pairplot(hit_pc[:, ["dir_theta", "dir_phi", "1_x", "1_y", "1_z"]])


out_pos = Matrix(hit_pc[:, ["out_x", "out_y", "out_z", "1_x"]])

fig, ax = scatter(out_pos)






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
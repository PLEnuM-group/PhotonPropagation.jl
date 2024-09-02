using PhotonPropagation
using BenchmarkTools
using CairoMakie
using StaticArrays
using CUDA
using Dates

distance = 40.0f0
medium = CascadiaMediumProperties()
n_pmts = 16
pmt_area = Float64((75e-3 / 2)^2 * π)
target_radius = 0.21f0

suite = BenchmarkGroup()
n_photons = exp10.(7:0.5:11)

target = HomogeneousDetector(
    Spherical(@SVector[0.0f0, 0f0, distance], target_radius,),
    pmt_area, UInt16(1))

spectrum = make_cherenkov_spectrum((300.0f0, 800.0f0), medium)

hbc, hbg = make_hit_buffers();

suite = BenchmarkGroup()
for nph in n_photons
    source = PointlikeIsotropicEmitter(SA[0.0f0, 0.0f0, 0.0f0], 0.0f0, Int64(ceil(nph)))
    setup = PhotonPropSetup([source], [target], medium, spectrum, 1)
    suite[nph] = @benchmarkable (CUDA.@sync $propagate_photons($setup, $hbc, $hbg)) samples=3 evals=1 seconds = 60
end
#println("Tuning starts $(now())")
#tune!(suite)
#println("Tuning ends $(now())")
println("Benchmarking starts $(now())")
results = run(suite, seconds=20, verbose = true)
println("Benchmarking ends $(now())")


medr = median(results)

collect(keys(medr))

fig, ax, p = scatter(Float64.(collect(keys(medr))), getproperty.(values(medr), (:time,)) ./ (keys(medr)),
    axis=(; xscale=log10, yscale=log10,  title=CUDA.name(CUDA.device()),
    xlabel="Number of Photons", ylabel="Time per Photon (ns)"),
   )
ylims!(ax,1E-1, 1E5)
fig

figpath =  joinpath(@__DIR__, "../figures/")

if !ispath(figpath)
    mkdir(figpath)
end

save(joinpath(figpath, "photon_benchmark_$(now())_$(CUDA.name(CUDA.device())).png"), fig)
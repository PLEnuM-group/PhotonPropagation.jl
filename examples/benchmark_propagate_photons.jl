using NeutrinoTelescopes
using Logging
using BenchmarkTools
using BenchmarkPlots, StatsPlots
using Plots
using StaticArrays
using CUDA
using StructArrays

distance = 30.0f0
medium = make_cascadia_medium_properties(0.99f0)
n_pmts = 16
pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0

suite = BenchmarkGroup()
n_photons = exp10.(5:0.5:10)

target = DetectionSphere(@SVector[0.0f0, 0f0, distance], target_radius, n_pmts, pmt_area, UInt16(1))

spectrum = CherenkovSpectrum((300.0f0, 800.0f0), 30, medium)

suite = BenchmarkGroup()
for nph in n_photons
    source = PointlikeIsotropicEmitter(SA[0.0f0, 0.0f0, 0.0f0], 0.0f0, Int64(ceil(nph)))
    suite[nph] = CUDA.@sync @benchmarkable $propagate_photons($source, $target, $medium, $spectrum)
end

tune!(suite)
results = run(suite, seconds=20)

plot(results)

medr = median(results)

p = scatter(collect(keys(medr)), getproperty.(values(medr), (:time,)) ./ (keys(medr)),
    xscale=:log10, yscale=:log10, ylim=(1E-1, 1E5),
    xlabel="Number of Photons", ylabel="Time per Photon (ns)",
    label="", dpi=150, title=CUDA.name(CUDA.device()))

savefig(p, joinpath(@__DIR__, "../figures/photon_benchmark.png"),)

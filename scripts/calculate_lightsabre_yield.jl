using CairoMakie
using PhotonPropagation
using ProposalInterface
using StatsBase
using CherenkovMediumBase
using CurveFit
using StaticArrays
using PhysicsTools
using DataFrames

medium = make_homogenous_clearice_properties(Float64)
wl_range = (300., 800.)
spectrum = make_cherenkov_spectrum(wl_range, medium)


spectrum.spectral_dist.normalization


rel_additional_track_length(1.33, 1E3)
p = Particle(SA[0, 0, 0], SA[1, 0, 0], 0, 1E3, 1, PMuPlus)
total_lightyield(p, medium, spectrum)


photons_per_m = frank_tamm_norm((300., 800.), wl -> phase_refractive_index(medium, wl))

λ, κ = PhotonPropagation.LightYield.rel_additional_track_length_params(1.33)


data = []
log_energies = 1.:0.5:8.5
lys = Float64[]
tlen = 10000.
prop = ProposalInterface.make_propagator(PMuMinus)
for le in log_energies
    particle = Particle(SA_F64[0, 0, 0], SA_F64[0, 0, 1], 0., 10^le, tlen, PMuMinus)
    for i in 1:100
        p, secondaries = propagate_muon(particle, propagator=prop)
        if length(secondaries) > 0
            ly_secondaries = sum(total_lightyield.(secondaries, Ref(medium), Ref(spectrum)))
        else
            ly_secondaries = 0.
        end
        push!(lys, ly_secondaries)
    end
    ly_secondaries_mean = mean(lys) / p.length
    ly_secondaries_median = median(lys) / p.length

    ly_secondaries_mean_old = mean(lys) / tlen
    ly_secondaries_median_old = median(lys) / tlen

    push!(data, (log_energy=le, mean_ly=ly_secondaries_mean, median_ly=ly_secondaries_median,
    mean_ly_old=ly_secondaries_mean_old, median_ly_old=ly_secondaries_median_old))
end

data = DataFrame(data)
fig, ax, lin = lines(data[:, :log_energy], log10.(data[:, :mean_ly]))
lines!(ax, data[:, :log_energy], log10.(data[:, :median_ly]), color=:red)
lines!(ax, data[:, :log_energy], log10.(data[:, :mean_ly_old]), color=:green)
lines!(ax, data[:, :log_energy], log10.(data[:, :median_ly_old]), color=:orange)
fig


coeffs = poly_fit(data[:, :log_energy], log10.(data[:, :mean_ly]), 5)
lines!(ax, log_energies, Polynomial(coeffs).(log_energies))
fig



fig, ax, lin = lines(data[:, :log_energy], (data[:, :mean_ly]))
lines!(ax, log_energies, 10 .^Polynomial(coeffs).(log_energies))
fig




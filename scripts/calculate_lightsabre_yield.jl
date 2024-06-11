using CairoMakie
using PhotonPropagation
using ProposalInterface
using StatsBase
using CurveFit
using StaticArrays
using PhysicsTools
using DataFrames
medium = make_cascadia_medium_properties(0.95)
wl_range = (300., 800.)
spectrum = make_cherenkov_spectrum(wl_range, medium)


data = []
log_energies = 1.:0.5:7
lys = Float64[]
tlen = 10000.
for le in log_energies
    particle = Particle(SA_F64[0, 0, 0], SA_F64[0, 0, 1], 0., 10^le, tlen, PMuMinus)
    for i in 1:10
        p, secondaries = propagate_muon(particle)
        if length(secondaries) > 0
            ly_secondaries = sum(total_lightyield.(secondaries, Ref(medium), Ref(spectrum)))
        else
            ly_secondaries = 0.
        end
        push!(lys, ly_secondaries)
    end
    ly_secondaries = mean(lys) / tlen
    push!(data, (log_energy=le, mean_ly=ly_secondaries))
end

data = DataFrame(data)
fig, ax, lin = lines(data[:, :log_energy], log10.(data[:, :mean_ly]))
coeffs = poly_fit(data[:, :log_energy], log10.(data[:, :mean_ly]), 5)
lines!(ax, log_energies, Polynomial(coeffs).(log_energies))
fig



fig, ax, lin = lines(data[:, :log_energy], (data[:, :mean_ly]))
lines!(ax, log_energies, 10 .^Polynomial(coeffs).(log_energies))
fig


coeffs
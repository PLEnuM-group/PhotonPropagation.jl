using CairoMakie
using PhotonPropagation
using StaticArrays
using CSV
using DataFrames
using PhysicsTools
using Random
using StatsBase
using Polynomials



log_energies = 2:0.1:10
zs = (0:0.1:20.0)# m
medium = make_cascadia_medium_properties(0.95)
wls = 300:0.1:800


cos_thetas = [scattering_function(medium) for _ in 1:100000]
fig, ax, _ = hist(cos_thetas, axis=(xlabel="cos(theta)", ylabel="PDF", yscale=log10), normalization=:pdf, )
vlines!(ax, mean(cos_thetas), color=:black, linewidth=3, linestyle=:dash)
fig


water_abs = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/water_absorption_wiki.csv");
    header=[:x, :y], delim=";", decimal=',', type=Float64))

fig = Figure()
ax = Axis(fig[1, 1], limits=(300, 800, 0, 300))
ax2 = Axis(fig[1, 1], yaxisposition = :right, ylabel="Absorption length (m)", limits=(300, 800, 0, 100))
lines!(ax, wls, d.spectrum, label="Frank-Tamm")

lines!(ax, wls, d2.spectrum, label="Frank-Tamm * QE")
lines!(ax2, water_abs[:, :x], 1 ./ water_abs[:, :y], color=:red, label="Absorption")
axislegend(ax)
fig
save(fig, joinpath(@__DIR__, "../figures/ch_spectrum.png"))


# Plot longitudinal profile
fig, ax, _ = lines(zs, longitudinal_profile.(Ref(1E3), zs, Ref(medium), Ref(PEMinus)), label="1E3 GeV",
    ylabel="PDF", title="Longitudinal Profile", dpi=150)
lines!(ax, zs, longitudinal_profile.(Ref(1E6), zs, Ref(medium), Ref(PEMinus)), label="1E6 GeV",
    xlabel="Distance along axis (m)")
axislegend(ax)
fig
savefig(p, joinpath(@__DIR__, "../figures/long_profile_comp.png"))


# Show fractional contribution for a segment of shower depth
frac_contrib = fractional_contrib_long(1E5, zs, medium, PEMinus)


plot(zs, frac_contrib, linetype=:steppost, label="", ylabel="Fractional light yield")

ftamm_norm = frank_tamm_norm((200.0, 800.0), wl -> refractive_index(wl, medium))
light_yield = cascade_cherenkov_track_length.(1E5, PEMinus)

plot(zs, frac_contrib .* light_yield, linetype=:steppost, label="", ylabel="Light yield per segment")


# Calculate Cherenkov track length as function of energy
tlens = cascade_cherenkov_track_length.((10 .^ log_energies), PEMinus)
lines(log_energies, tlens,
    axis=(; yscale=log10, xlabel="Log10(E/GeV)", ylabel="Cherenkov track length"))

total_lys = frank_tamm_norm((200.0, 800.0), wl ->phase_refractive_index(wl, medium)) * tlens

p = lines(log_energies, total_lys,
    axis=(; yscale=log10, ylabel="Number of photons", xlabel="log10(Energy/GeV)"),
    label="", dpi=150)
savefig(p, joinpath(@__DIR__, "../figures/photons_per_energy.png"))


#cascade_cherenkov_track_length(1E9, PEMinus)  / cascade_cherenkov_track_length(1E5, PEMinus)

# Calculate light yield for muons

lines(log_energies, rel_additional_track_length.(phase_refractive_index(800.0, medium), 10 .^ log_energies) .* frank_tamm(450.0, phase_refractive_index(450.0, medium)) .* 1E9 .* 1E2 / 10)


lambdas = 200:1.0:800

lines(lambdas, phase_refractive_index.(lambdas, Ref(medium)))
lines(lambdas, dispersion.(lambdas, Ref(medium)))
lines(group_velocity.(lambdas, Ref(medium)))




dom = DOM(SA[0., 0., 0.], 1)


d = make_cherenkov_spectrum((300., 800.), medium)
d2 = make_biased_cherenkov_spectrum(dom.acceptance.int_wl, (300., 800.), medium)

fig, ax, _ = hist(rand(d.spectral_dist, 100000), normalization=:pdf)
hist!(ax, rand(d2.spectral_dist, 100000), normalization=:pdf)
fig


p = Particle(SA[0., 0., 0.], SA[0., 0., 1.], 0., 1E5, 0., PEMinus)
em = ExtendedCherenkovEmitter(p, medium, d2)

tlen = cascade_cherenkov_track_length(1E5, PEMinus)


em.photons / (frank_tamm_norm((300.0, 800.0), wl ->phase_refractive_index(wl, medium)) * tlen)


medium = make_homogenous_clearice_properties()
tlens = cascade_cherenkov_track_length.((10 .^ log_energies), PEMinus)
lines(log_energies, tlens,
    axis=(; yscale=log10, xlabel="Log10(E/GeV)", ylabel="Cherenkov track length"))

total_lys = frank_tamm_norm((200.0, 800.0), wl ->phase_refractive_index(wl, medium)) * tlens

fig, ax, p = lines(log_energies, total_lys,
    axis=(; yscale=log10, ylabel="Number of photons", xlabel="log10(Energy/GeV)"),
    label="", dpi=150)

scaling_func(log_e, a, b) = @. (clamp(log_e-a, 0, typemax(Float64)))*b


lines!(ax, log_energies, total_lys .* (1 .+scaling_func(log_energies, 5, 0.1)))


fig


log_energies = 2:0.1:7
data = []
for le in log_energies

    sum_hadr_loss = Float64[]
    sum_loss = Float64[]
    for _ in 1:100
        particle = Particle(SA_F64[0, 0, 0], SA_F64[0, 0, 1], 0., 10^le, 1500., PMuMinus)
        final_state, secondaries = propagate_muon(particle)

        hadronic_losses = [s.energy for s in secondaries if s.type == PHadronShower]
        all_losses = [s.energy for s in secondaries]
        #scaled_losses = hadronic_losses .* (1 .+ scaling_func.(log10.(hadronic_losses), 2, 0.3))

        if length(hadronic_losses) > 0
            push!(sum_hadr_loss, sum(hadronic_losses))
        else
            push!(sum_hadr_loss, 0.)
        end
        push!(sum_loss, sum(all_losses))
    end
    push!(data, (le=le, hadr_loss=mean(sum_hadr_loss), total_loss = mean(sum_loss)))
end


data = DataFrame(data)

data[!, :hadr_frac] .= data[:, :hadr_loss] ./ data[:, :total_loss]
fig, ax, _ = lines(data[:, :le], (data[:, :hadr_frac]))

p =  Polynomials.fit(data[:, :le], (data[:, :hadr_frac]), 1)

xs = 2:0.1:7
lines!(ax, xs, p.(xs))

fig

    #scaled_hadr_loss_sum = sum_hadr_loss .* (1 .+ scaling_func.(log10.(sum_hadr_loss), 2, 0.3))




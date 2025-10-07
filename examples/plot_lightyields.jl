using CairoMakie
using PhotonPropagation
using StaticArrays
using CSV
using DataFrames
using PhysicsTools
using Random
using StatsBase
using Polynomials
using KM3NeTMediumProperties
using Format
using AbstractMediumProperties
using LinearAlgebra
using Unitful
using PhysicalConstants.CODATA2018

medium = CascadiaMediumProperties()

for e in 10 .^ (2:0.5:7)
    p = Particle(SA[0., 0., 0.], SA[0., 0., 1.], 0., e, 0., PEMinus)
    em = FastLightsabreMuonEmitter(p, medium, make_cherenkov_spectrum((300., 800.), medium))
    println("E: $(e), photons: $(em.photons), photons/m: $(em.photons / p.length)")
end







function plot_cherenkov_spectrum(wl_min, wl_max, medium)
    wavelength = wl_min:1.:wl_max
    norm = frank_tamm_norm((wl_min, wl_max), wl -> phase_refractive_index(wl, medium))
    fig = Figure()
    ax = Axis(fig[1, 1], title=format("Cherenkov Spectrum. Norm: {:.2f} /cm", norm/100), xlabel="Wavelength (nm)", ylabel="Photons / m^2")
    lines!(ax, wavelength, frank_tamm.(wavelength, phase_refractive_index.(wavelength, Ref(medium))))
    return fig
end


function plot_track_length(log_e_min, log_e_max; ptype=PEMinus)

    reference_data = [
        1.0064783277286045 524.734426888343
        3.0025645204983777 1583.9810802614982
        6.89300086758925 3689.39155822141
        10.073938191981963 5232.552377993151
        30.326632452208113 15617.705540146851
        68.99270732841501 36789.91425395293
        99.92285988927631 52178.60037754378
        292.7533809658026 155743.71053119222
        696.8307741465146 366857.99337362836
        1000.1197535338733 526221.2982091925
        2930.092728152024 1588504.625396774
        6911.70496158875 3699804.68230323
        9742.244333967774 5247553.723195897
    ]


    energies = 10 .^(log_e_min:0.1:log_e_max)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title="Additional track length",
        xlabel="Energy (GeV)",
        ylabel="Additional track length (cm)",
        yscale=log10,
        xscale=log10,)
    lines!(ax, energies, cascade_cherenkov_track_length.(energies, ptype))
    scatter!(ax, reference_data[:, 1], reference_data[:, 2] ./ 100, label="Reference Data")
    axislegend()
    return fig
end

function plot_total_lightyield(log_e_min, log_e_max, wl_min, wl_max, medium; ptype=PEMinus)
    energies = 10 .^(log_e_min:0.1:log_e_max)
    spectrum = make_cherenkov_spectrum((wl_min, wl_max), medium)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        title="Total light yield",
        xlabel="Energy (GeV)",
        ylabel="Total light yield (photons)",
        yscale=log10,
        xscale=log10,)
    lines!(ax, energies, total_lightyield.(Ref(Cascade()), energies, Ref(ptype), Ref(spectrum)))

    norm = frank_tamm_norm((wl_min, wl_max), wl -> phase_refractive_index(wl, medium))
    lines!(ax, energies, cascade_cherenkov_track_length.(energies, ptype) .* norm, label="Cherenkov track length")

    #=
    ps = Particle.(Ref(SA[0., 0., 0.]), Ref(SA[0., 0., 1.]), 0., energies, 0., ptype)
    ems = ExtendedCherenkovEmitter.(ps, Ref(medium), Ref(spectrum))
    nphs = [em.photons for em in ems]
    scatter!(ax, energies, nphs, label="Photons emitted")
    =#

    jpp_const = 4.7319
    lines!(ax, energies, jpp_const .* energies .* norm, label="JPP constant")
    axislegend(position=:lt)

    return fig
end


function plot_generated_photon_properties(medium, wl_min, wl_max; ptype=PEMinus)
    spectrum = make_cherenkov_spectrum((wl_min, wl_max), medium)

    energy = 1E5
    cascade_dir = [1., 0., 0.]
    p = Particle(SA[0., 0., 0.], cascade_dir, 0., energy, 0., ptype)
    #emitter = ExtendedCherenkovEmitterCustomSmear(p, medium, spectrum, cherenkov_beta_a=1.1, cherenkov_beta_b=0.07)

    emitter = ExtendedCherenkovEmitter(p, medium, spectrum)

    photons = [PhotonPropagationCuda.initialize_photon_state(emitter, medium, spectrum.spectral_dist) for _ in 1:100000]

    fig = Figure(size=(800, 800))
    ax = Axis(
        fig[1, 1],
        title="Emitted Spectrum",
        xlabel="Wavelength (nm)",
        ylabel="Photons",
        yscale=log10,
        )

    wl_bins = wl_min:10:wl_max
    hist!(ax, [p.wavelength for p in photons], bins=wl_bins)

    norms = frank_tamm_norm.([(wl_bins[i], wl_bins[i+1]) for i in 1:(length(wl_bins)-1)], wl -> phase_refractive_index(wl, medium))
    total_norm = frank_tamm_norm((wl_min, wl_max), wl -> phase_refractive_index(wl, medium))
    tlen = cascade_cherenkov_track_length.(energy, ptype)
    println(emitter.photons ./ (total_norm * tlen))
    centers = (wl_bins[1:end-1] .+ wl_bins[2:end]) ./ 2
    scatter!(ax, centers, norms ./ total_norm .* length(photons), color=:red, label="Expected")
    
    ax2 = Axis(fig[1, 2], title="Angle to cascade direction", xlabel="cos(theta)", ylabel="PDF")

    hist!(ax2, [dot(p.direction, cascade_dir) for p in photons], bins=100, normalization=:pdf)
    vlines!(ax2, cherenkov_angle(400., medium))

    ax3 = Axis(fig[2, 1], title="Phi Angle", xlabel="phi", ylabel="PDF")
    hist!(ax3, [cart_to_sph(p.direction)[2] for p in photons], normalization=:pdf)

    ax4 = Axis(fig[2, 2], title="Position along shower", xlabel="z (m)", ylabel="PDF")
    hist!(ax4, [p.position[1] for p in photons], normalization=:pdf)

    return fig
end


function plot_angles_after_scatter(medium, wl_min, wl_max; ptype=PEPlus)

    spectrum = make_cherenkov_spectrum((wl_min, wl_max), medium)

    energy = 1E5
    cascade_dir = [1., 0., 0.]
    p = Particle(SA[0., 0., 0.], cascade_dir, 0., energy, 0., ptype)
    emitter = ExtendedCherenkovEmitter(p, medium, spectrum)
    photons = [PhotonPropagationCuda.initialize_photon_state(emitter, medium, spectrum.spectral_dist) for _ in 1:10000]
    new_directions = [PhotonPropagationCuda.update_direction(ph.direction, medium) for ph in photons]
    fig = Figure()
    ax = Axis(fig[1, 1], title="Angle to cascade direction", xlabel="cos(theta)", ylabel="PDF")

    hist!(ax, [dot(nd, cascade_dir) for nd in new_directions], normalization=:pdf)
    vlines!(ax, cherenkov_angle(400., medium))

    return fig
end


medium = KM3NeTMediumArca(1., 1., 0.17)

function lambda_to_energy(lambda)
    return ustrip(u"eV", SpeedOfLightInVacuum / (lambda*1u"nm") * PlanckConstant)
end

function energy_to_lambda(E)
    return ustrip(u"nm",  SpeedOfLightInVacuum * PlanckConstant / (E*1u"eV"))
end


function sample_cherenkov_e_geant(wl_min, wl_max, medium)

    Pmin = lambda_to_energy(wl_max)
    Pmax = lambda_to_energy(wl_min)
    dp = Pmax - Pmin

    eta = rand()
    sampledEnergy = Pmin + eta * dp
    ref_ix = phase_refractive_index(energy_to_lambda(sampledEnergy), medium)
    nMax = phase_refractive_index(wl_min, medium)
    maxCos = 1 / nMax
    BetaInverse = 1
    maxSin2 = (1.0 - maxCos) * (1.0 + maxCos)
    sampled_lambda = 0.
    while true
        eta = rand()
        sampledEnergy = Pmin + eta * dp
        sampled_lambda = energy_to_lambda(sampledEnergy)
        ref_ix = phase_refractive_index(sampled_lambda, medium)
        cosTheta = BetaInverse / ref_ix;
        sin2Theta = (1.0 - cosTheta) * (1.0 + cosTheta);

        eta = rand()

        if eta * maxSin2 < sin2Theta
            break
        end

    end
    return sampled_lambda
end

function plot_cherenkov_spectrum_compare_geant(wl_min, wl_max, medium)
    wavelength = wl_min:1.:wl_max
    norm = frank_tamm_norm((wl_min, wl_max), wl -> phase_refractive_index(wl, medium))
    fig = Figure()
    ax = Axis(fig[1, 1], title=format("Cherenkov Spectrum. Norm: {:.2f} /cm", norm/100), xlabel="Wavelength (nm)", ylabel="PDF")
    

    lambdas = [sample_cherenkov_e_geant(wl_min, wl_max, medium) for _ in 1:10000]
    hist!(ax, lambdas, normalization=:pdf, label="Geant4")
    lines!(ax, wavelength, frank_tamm.(wavelength, phase_refractive_index.(wavelength, Ref(medium))) ./ norm *1E9, label="Frank-Tamm", color=Makie.wong_colors()[2])
    axislegend()
    return fig
end





#plot_cherenkov_spectrum_compare_geant(300., 800., medium)

sph_to_cart(2*π+0.1, 0)
sph_to_cart(2*π-0.1, 0)


histories = trace_photon_paths(medium, 300., 800., 50, 30)
plot_photon_history(histories, bounds=(-200, 200))

plot_cherenkov_spectrum(300., 500., medium)
plot_track_length(1., 7.)
plot_total_lightyield(1, 7., 300., 800., medium)
plot_generated_photon_properties(medium, 300., 800.)

plot_angles_after_scatter(medium, 300., 800.)

cascade_cherenkov_track_length.(1E4, PEMinus) / 1E4

histories[1].positions

log_energies = 2:0.1:10
zs = (0:0.1:20.0)# m
medium = make_cascadia_medium_properties(0.95)
wls = 300:0.1:800


cos_thetas = [scattering_function(medium) for _ in 1:100000]
fig, ax, _ = hist(cos_thetas, axis=(xlabel="cos(theta)", ylabel="PDF", yscale=log10), normalization=:pdf, )
vlines!(ax, mean(cos_thetas), color=:black, linewidth=3, linestyle=:dash)
fig


sample_cherenkov_track_direction(Float64)

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




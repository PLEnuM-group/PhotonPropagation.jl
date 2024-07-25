using PhotonPropagation
using CairoMakie
using PhysicsTools
using KM3NeTMediumProperties
using AbstractMediumProperties
using QuadGK
using Polynomials



function plot_medium(medium, fig, axes)
    wavelengths = 350:1.:700
   
    ax1, ax2, ax3, ax4 = axes

    lines!(ax1, wavelengths, absorption_length.(wavelengths, Ref(medium)))
    lines!(ax2, wavelengths, scattering_length.(wavelengths, Ref(medium)))
    lines!(ax3, wavelengths, phase_refractive_index.(wavelengths, Ref(medium)))
   
    #ax4 = Axis(fig[2,2], xlabel="Wavelength (nm)", ylabel="Group Refractive Index")
    lines!(ax4, wavelengths, group_refractive_index.(wavelengths, Ref(medium)))
    #hist!(ax4, [scattering_function(medium) for _ in 1:100000], normalization=:pdf)
    return fig, axes
end



function plot_medium(medium)
    fig = Figure()
    ax1 = Axis(fig[1,1], xlabel="Wavelength (nm)", ylabel="Absorption length (m)")
    ax2 = Axis(fig[1,2], xlabel="Wavelength (nm)", ylabel="Scattering length (m)")
    ax3 = Axis(fig[2,1], xlabel="Wavelength (nm)", ylabel="Phase Refractive Index")
    ax4 = Axis(fig[2,2], xlabel="Wavelength", ylabel="Group Refractive Index", )

    return plot_medium(medium, fig, [ax1, ax2, ax3, ax4])
end

medium = KM3NeTMediumArca(1., 1., 0.17)



fig, axes = plot_medium(medium)
medium2 = CascadiaMediumProperties(0.95, 1., 1.)
plot_medium(medium2, fig, axes)
fig


cos_thetas = [scattering_function(medium) for _ in 1:1000000]
cos_thetas_hg = [hg_scattering_func(0.925) for _ in 1:1000000]
cos_thetas_es_17 = [KM3NeTMediumProperties.mixed_hg_es_scattering_func(mean_scattering_angle(medium), 1-0.17, (KM3NeTMediumProperties.KM3NeT_ES_POLY_COEFFS)) for _ in 1:1000000]


cos_theta = -1:0.01:1
es_poly = fit(Polynomial, es_cumulative.(cos_theta, KM3NeTMediumProperties.KM3NeT_ES_B), cos_theta, 3)
etas = 0:0.01:1
fig, ax, l = lines(etas, es_poly.(etas))
lines!(ax, es_cumulative.(cos_theta, KM3NeTMediumProperties.KM3NeT_ES_B), cos_theta)
fig

fig = Figure(size=(1000, 500))
ax = Axis(fig[1, 1], yscale=log10, xlabel="cos(theta)", ylabel="Counts", title="Pure HG")
ax2 = Axis(fig[1, 2], yscale=log10, xlabel="cos(theta)", ylabel="Counts", title="75% HG, 25% ES")
ax3 = Axis(fig[1, 3], yscale=log10, xlabel="cos(theta)", ylabel="Counts", title="83% HG, 17% ES")
bins = -1:0.01:1
hist!(ax, cos_thetas_hg, bins, fillto=1E3,  )
hist!(ax2, cos_thetas, bins, fillto=1E3)
hist!(ax3, cos_thetas_es_17, bins, fillto=1E3, )
#hist!(ax, cos_thetas_es, bins, fillto=1E3)
fig







wavelengths = 350:1.:700

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Wavelength (nm)", ylabel="Attenuation length (m), calculated")

lines!(ax, wavelengths, 1 ./ (1 ./absorption_length.(wavelengths, Ref(medium)) .+ 1 ./scattering_length.(wavelengths, Ref(medium))), label="KM3NeT")
lines!(ax, wavelengths, 1 ./ (1 ./absorption_length.(wavelengths, Ref(medium2)) .+ 1 ./scattering_length.(wavelengths, Ref(medium2))), label="CASCADIA")
axislegend()
fig
medium = make_homogenous_clearice_properties()

fig, axes = plot_medium(medium)
medium2 = make_cascadia_medium_properties(0.95)
medium3 = make_cascadia_medium_properties(0.95, 1.1, 1.1)
fig, axes = plot_medium(medium2, fig, axes)
fig, axes = plot_medium(medium3, fig, axes)
fig

wls = 300:1.:800.

plot(wls, frank_tamm.(wls, 1.4))





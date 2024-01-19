using PhotonPropagation
using CairoMakie
using PhysicsTools

function plot_medium(medium, fig, axes)
    wavelengths = 350:1.:700
   
    ax1, ax2, ax3, ax4 = axes

    lines!(ax1, wavelengths, absorption_length.(wavelengths, Ref(medium)))
    lines!(ax2, wavelengths, scattering_length.(wavelengths, Ref(medium)))
    lines!(ax3, wavelengths, phase_refractive_index.(wavelengths, Ref(medium)))
   
    #ax4 = Axis(fig[2,2], xlabel="Wavelength (nm)", ylabel="Group Refractive Index")
    #lines!(ax4, wavelengths, group_refractive_index.(wavelengths, Ref(medium)))
    hist!(ax4, [scattering_function(medium) for _ in 1:100000], normalization=:pdf)
    return fig, axes
end



function plot_medium(medium)
    fig = Figure()
    ax1 = Axis(fig[1,1], xlabel="Wavelength (nm)", ylabel="Absorption length (m)")
    ax2 = Axis(fig[1,2], xlabel="Wavelength (nm)", ylabel="Scattering length (m)")
    ax3 = Axis(fig[2,1], xlabel="Wavelength (nm)", ylabel="Phase Refractive Index")
    ax4 = Axis(fig[2,2], xlabel="Scattering Function", ylabel="Density", yscale=log10)

    return plot_medium(medium, fig, [ax1, ax2, ax3, ax4])
end


medium = make_homogenous_clearice_properties()

fig, axes = plot_medium(medium)
medium2 = make_cascadia_medium_properties(0.95)
medium3 = make_cascadia_medium_properties(0.95, 1.1, 1.1)
fig, axes = plot_medium(medium2, fig, axes)
fig, axes = plot_medium(medium3, fig, axes)
fig

wls = 300:1.:800.

plot(wls, frank_tamm.(wls, 1.4))





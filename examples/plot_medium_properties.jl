using PhotonPropagation
using CairoMakie

medium = make_homogenous_clearice_properties()

wavelengths = 300:1.:700

lines(wavelengths, absorption_length.(wavelengths, Ref(medium)),
axis=(xlabel="Wavelength (nm)", ylabel="Absorption length (m)"))

lines(wavelengths, scattering_length.(wavelengths, Ref(medium)),
axis=(xlabel="Wavelength (nm)", ylabel="Scattering length (m)"))


lines(wavelengths, phase_refractive_index.(wavelengths, Ref(medium)),
axis=(xlabel="Wavelength (nm)", ylabel="Phase Refractive Index "))

lines(wavelengths, group_refractive_index.(wavelengths, Ref(medium)),
axis=(xlabel="Wavelength (nm)", ylabel="Phase Refractive Index "))
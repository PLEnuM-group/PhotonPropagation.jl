"""
    HomogenousIceProperties{T<:Real} <: MediumProperties{T}

Properties for deep (homogenuous) ice. Uses the SPICE model.
See: https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/new/paper/a.pdf

### Fields:
-radiation_length -- Radiation length (g/cm^2)
-mean_scattering_angle -- Cosine of the mean scattering angle
-A_SPICE -- A parameter of the SPICE model
-B_SPICE -- B parameter of the SPICE model
-a_dust_400 -- Absorption coefficient at 400nm
-b_dust_400 -- Scattering coefficient at 400nm
-alpha_sca_dust -- Alpha parameter of the SPICE model
-kappa_abs_dust -- Kappa parameter of the SPICE model
-deltaTAMANDA -- deltaT parameter of the SPICE model
"""
Base.@kwdef struct HomogenousIceProperties{T<:Real} <: MediumProperties{T}
    radiation_length::T # g / cm^2
    mean_scattering_angle::T
    A_SPICE::T
    B_SPICE::T
    a_dust_400::T
    b_dust_400::T
    alpha_sca_dust::T
    kappa_abs_dust::T
    deltaTSPICE::T
end

StructTypes.StructType(::Type{<:HomogenousIceProperties}) = StructTypes.Struct()

"""
    make_homogenous_clearice_properties()
Ice properties for the "homogenous clear ice model"
"""
make_homogenous_clearice_properties() = HomogenousIceProperties(
    radiation_length=0.358/0.9216,
    mean_scattering_angle=0.9,
    A_SPICE=6954.090332031250,
    B_SPICE=6617.754394531250,
    a_dust_400=0.006350013,
    b_dust_400=0.02331206666666667,
    alpha_sca_dust=0.898608505726,
    kappa_abs_dust=1.084106802940,
    deltaTSPICE=12.115215000000001
)

radiation_length(x::HomogenousIceProperties) = x.radiation_length
mean_scattering_angle(x::HomogenousIceProperties) = x.mean_scattering_angle
b_dust_400(x::HomogenousIceProperties) = x.b_dust_400
a_dust_400(x::HomogenousIceProperties) = x.a_dust_400
alpha_sca_dust(x::HomogenousIceProperties) = x.alpha_sca_dust
kappa_abs_dust(x::HomogenousIceProperties) = x.kappa_abs_dust
deltaTSPICE(x::HomogenousIceProperties) = x.deltaTSPICE
A_SPICE(x::HomogenousIceProperties) = x.A_SPICE
B_SPICE(x::HomogenousIceProperties) = x.B_SPICE

"""
    _ice_phase_refractive_index(wavelength)

Calculate the phase refractive index at wavelength (in nm)
Taken from: "Role of group and phase velocity in high-energy neutrino observatories", https://doi.org/10.1016/S0927-6505(00)00142-0
"""
function _ice_phase_refractive_index(wavelength)
    wavelength /= 1000
    return oftype(wavelength, 1.55749 − 1.57988 * wavelength + 3.99993 * wavelength^2 − 4.68271 * wavelength^3 + 2.09354 * wavelength^4)
end

"""
_ice_group_refractive_index(wavelength)

Calculate the group refractive index at wavelength (in nm)
Taken from: "Role of group and phase velocity in high-energy neutrino observatories", https://doi.org/10.1016/S0927-6505(00)00142-0
"""
function _ice_group_refractive_index(wavelength)
    np = _ice_phase_refractive_index(wavelength)
    wavelength /= 1000
    return oftype(wavelength, np * (1 + 0.227106 − 0.954648 * wavelength + 1.42568 * wavelength^2 − 0.711832 * wavelength^3))
end

phase_refractive_index(wavelength, ::HomogenousIceProperties) = _ice_phase_refractive_index(wavelength)
group_refractive_index(wavelength, ::HomogenousIceProperties) = _ice_group_refractive_index(wavelength)

function group_velocity(wavelength::T, medium::HomogenousIceProperties) where {T<:Real}
    global c_vac_m_ns
    return T(c_vac_m_ns) / group_refractive_index(wavelength, medium)
end

"""
    _absorption_coeff_dust(wavelength, a_dust_400, kappa_abs_dust)
Calculate the absorption coefficient contribution due to dust
Taken from "Measurement of South Pole ice transparency with the IceCube LED calibration system", https://doi.org/10.1016/j.nima.2013.01.054
"""
function _absorption_coeff_dust(wavelength, a_dust_400, kappa_abs_dust)
    return oftype(wavelength, a_dust_400 * (wavelength / 400)^(-kappa_abs_dust))
end

"""
    _absorption_coeff_spice(wavelength, A_SPICE, B_SPICE, a_dust_400, kappa, dT)
Calculate the absorption coefficient for the SPICE model.
Taken from "Measurement of South Pole ice transparency with the IceCube LED calibration system", https://doi.org/10.1016/j.nima.2013.01.054
"""
function _absorption_coeff_spice(wavelength, A_SPICE, B_SPICE, a_dust_400, kappa, dT)
    adust = _absorption_coeff_dust(wavelength, a_dust_400, kappa)
    a_temp = A_SPICE * exp(-B_SPICE/wavelength) * (1 + 0.01 * dT)
    return oftype(wavelength, adust + a_temp)
end

function absorption_length(wavelength, medium::HomogenousIceProperties)
    abs_coeff = _absorption_coeff_spice(
        wavelength, A_SPICE(medium), B_SPICE(medium),
        a_dust_400(medium), kappa_abs_dust(medium), deltaTSPICE(medium))
    return oftype(wavelength, 1/abs_coeff)
end

"""
_eff_sca_coeff_dust(wavelength, b_dust_400, alpha_sca_dust)
Calculate the effective scattering coefficient for the SPICE model.
Taken from "Measurement of South Pole ice transparency with the IceCube LED calibration system", https://doi.org/10.1016/j.nima.2013.01.054
"""
function _eff_sca_coeff_dust(wavelength, b_dust_400, alpha_sca_dust)
    return oftype(wavelength, b_dust_400 * (wavelength/400)^(-alpha_sca_dust))
end

function scattering_length(wavelength::Real, medium::HomogenousIceProperties)
    eff_coeff = _eff_sca_coeff_dust(wavelength, b_dust_400(medium), alpha_sca_dust(medium))
    sca_coeff = eff_coeff / ( 1 - mean_scattering_angle(medium))
    return oftype(wavelength, 1/sca_coeff)
end

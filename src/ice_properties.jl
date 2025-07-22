export HomogenousIceProperties
export make_homogenous_clearice_properties

"""
    HomogenousIceProperties{T<:Real} <: MediumProperties{T}

Properties for deep (homogenuous) ice. Uses the SPICE model.
See: https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/new/paper/a.pdf

### Fields:
-radiation_length -- Radiation length (g/cm^2)
-mean_scattering_angle -- Cosine of the mean scattering angle
-hg_fraction -- Mixture weight of the HG component
-A_SPICE -- A parameter of the SPICE model
-B_SPICE -- B parameter of the SPICE model
-D_SPICE -- B parameter of the SPICE model
-E_SPICE -- B parameter of the SPICE model
-a_dust_400 -- Absorption coefficient at 400nm
-b_dust_400 -- Scattering coefficient at 400nm
-alpha_sca_dust -- Alpha parameter of the SPICE model
-kappa_abs_dust -- Kappa parameter of the SPICE model
-deltaTAMANDA -- deltaT parameter of the SPICE model
-abs_scale -- Scaling factor for the absorption length
-sca_scale -- Scaling factor for the scattering length
"""
Base.@kwdef struct HomogenousIceProperties{T<:Real} <: MediumProperties
    radiation_length::T # g / cm^2
    mean_scattering_angle::T
    hg_fraction::T
    A_SPICE::T
    B_SPICE::T
    D_SPICE::T
    E_SPICE::T
    a_dust_400::T
    b_dust_400::T
    alpha_sca_dust::T
    kappa_abs_dust::T
    deltaTSPICE::T
    abs_scale::T
    sca_scale::T
end

StructTypes.StructType(::Type{<:HomogenousIceProperties}) = StructTypes.Struct()

"""
    make_homogenous_clearice_properties()
Ice properties for the "homogenous clear ice model"
"""
make_homogenous_clearice_properties(T, abs_scale=1., sca_scale=1.) = HomogenousIceProperties(
    radiation_length=T(39.652/0.9216),
    mean_scattering_angle=T(0.9),
    hg_fraction=T(0.45),
    A_SPICE=T(6954.090332031250),
    B_SPICE=T(6617.754394531250),
    D_SPICE=T(0.),
    E_SPICE=T(0.),
    a_dust_400=T(0.006350013),
    b_dust_400=T(0.02331206666666667),
    alpha_sca_dust=T(0.898608505726),
    kappa_abs_dust=T(1.084106802940),
    deltaTSPICE=T(12.115215000000001),
    abs_scale=T(abs_scale),
    sca_scale=T(sca_scale),
)

function Base.convert(::Type{HomogenousIceProperties{T}}, m::HomogenousIceProperties) where {T <: Real}
    return HomogenousIceProperties(
        T(m.radiation_length),
        T(m.mean_scattering_angle),
        T(m.hg_fraction),
        T(m.A_SPICE),
        T(m.B_SPICE),
        T(m.D_SPICE),
        T(m.E_SPICE),
        T(m.a_dust_400),
        T(m.b_dust_400),
        T(m.alpha_sca_dust),
        T(m.kappa_abs_dust),
        T(m.deltaTSPICE),
        T(m.abs_scale),
        T(m.sca_scale),
    )
end

mean_scattering_angle(x::HomogenousIceProperties) = x.mean_scattering_angle


CherenkovMediumBase.radiation_length(x::HomogenousIceProperties) = x.radiation_length
CherenkovMediumBase.material_density(::HomogenousIceProperties) = 0.910 * 1000
hg_fraction(x::HomogenousIceProperties) = x.hg_fraction
b_dust_400(x::HomogenousIceProperties) = x.b_dust_400
a_dust_400(x::HomogenousIceProperties) = x.a_dust_400
alpha_sca_dust(x::HomogenousIceProperties) = x.alpha_sca_dust
kappa_abs_dust(x::HomogenousIceProperties) = x.kappa_abs_dust
deltaTSPICE(x::HomogenousIceProperties) = x.deltaTSPICE
A_SPICE(x::HomogenousIceProperties) = x.A_SPICE
B_SPICE(x::HomogenousIceProperties) = x.B_SPICE
D_SPICE(x::HomogenousIceProperties) = x.D_SPICE
E_SPICE(x::HomogenousIceProperties) = x.E_SPICE

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

CherenkovMediumBase.phase_refractive_index(::HomogenousIceProperties, wavelength) = _ice_phase_refractive_index(wavelength)
CherenkovMediumBase.group_refractive_index(::HomogenousIceProperties, wavelength) = _ice_group_refractive_index(wavelength)

function CherenkovMediumBase.group_velocity(medium::HomogenousIceProperties, wavelength::T) where {T<:Real}
    global c_vac_m_ns
    return T(c_vac_m_ns) / group_refractive_index(wavelength, medium)
end

"""
    _absorption_coeff_dust(wavelength, a_dust_400, kappa_abs_dust)
Calculate the absorption coefficient contribution due to dust
Taken from "Measurement of South Pole ice transparency with the IceCube LED calibration system", https://doi.org/10.1016/j.nima.2013.01.054
"""
function _absorption_coeff_dust(wavelength, a_dust_400, kappa_abs_dust, D, E)
    #a_dust_400 = (D*a_dust_400+E)*400^-kappa_abs_dust
    return oftype(wavelength, a_dust_400 * (wavelength / 400)^(-kappa_abs_dust))
end

"""
    _absorption_coeff_spice(wavelength, A_SPICE, B_SPICE, a_dust_400, kappa, dT)
Calculate the absorption coefficient for the SPICE model.
Taken from "Measurement of South Pole ice transparency with the IceCube LED calibration system", https://doi.org/10.1016/j.nima.2013.01.054
"""
function _absorption_coeff_spice(wavelength, A_SPICE, B_SPICE, D_SPICE, E_SPICE, a_dust_400, kappa, dT)
    adust = _absorption_coeff_dust(wavelength, a_dust_400, kappa, D_SPICE, E_SPICE)
    a_temp = A_SPICE * exp(-B_SPICE/wavelength) * (1 + 0.01 * dT)
    return oftype(wavelength, adust + a_temp)
end

function CherenkovMediumBase.absorption_length(medium::HomogenousIceProperties, wavelength)
    abs_coeff = _absorption_coeff_spice(
        wavelength, A_SPICE(medium), B_SPICE(medium), D_SPICE(medium), E_SPICE(medium),
        a_dust_400(medium), kappa_abs_dust(medium), deltaTSPICE(medium))
    return oftype(wavelength, 1/abs_coeff * medium.abs_scale)
end

"""
_eff_sca_coeff_dust(wavelength, b_dust_400, alpha_sca_dust)
Calculate the effective scattering coefficient for the SPICE model.
Taken from "Measurement of South Pole ice transparency with the IceCube LED calibration system", https://doi.org/10.1016/j.nima.2013.01.054
"""
function _eff_sca_coeff_dust(wavelength, b_dust_400, alpha_sca_dust)
    return oftype(wavelength, b_dust_400 * (wavelength/400)^(-alpha_sca_dust))
end

function CherenkovMediumBase.scattering_length(medium::HomogenousIceProperties, wavelength::Real, )
    eff_coeff = _eff_sca_coeff_dust(wavelength, b_dust_400(medium), alpha_sca_dust(medium))
    sca_coeff = eff_coeff / ( 1 - mean_scattering_angle(medium))
    return oftype(wavelength, 1/sca_coeff * medium.sca_scale)
end

CherenkovMediumBase.sample_scattering_function(medium::HomogenousIceProperties) = mixed_hg_sl_scattering_func_ppc(mean_scattering_angle(medium), hg_fraction(medium))
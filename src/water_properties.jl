
# Implementation of WaterProperties
using StaticArrays

export CascadiaMediumProperties

abstract type WaterProperties{T<:Real} <: MediumProperties end

"""
    _absorption_length_straw(wavelength::Real)
Calculate the absorption length at `wavelength` (in nm).

Based on interpolation of STRAW attenuation length measurement.
"""
function _absorption_length_straw(wavelength::Real)
    T = typeof(wavelength)
    x = [T(300.0), T(365.0), T(400.0), T(450.0), T(585.0), T(800.0)]
    y = [T(10.4), T(10.4), T(14.5), T(27.7), T(7.1), T(7.1)]

    fast_linear_interp(wavelength, x, y)
end


const ABSLENGTHSTRAWFIT = let df = DataFrame(Parquet2.Dataset(joinpath(pkgdir(@__MODULE__), "assets/attenuation_length_straw_fit.parquet")))
    SA[Matrix(df[:, [:wavelength, :abs_len]])]
end

const CASCADIA_ES_B = 0.835
const CASCADIA_VOL_CONC_SMALL_PART = 7.5E-3
const CASCADIA_VOL_CONC_LARGE_PART = 7.5E-3
const CASCADIA_HG_G = 0.95
const CASCADIA_SALINITY = 34.82 # permille
const CASCADIA_TEMPERATURE = 1.8 # °C
const CASCADIA_DENSITY = DIPPR105(CASCADIA_TEMPERATURE + 273.15) # kg/m^3
const CASCADIA_PRESSURE = 265.917473476 # atm
const CASCADIA_RADIATION_LENGTH = 36.08 # g/cm^2

struct CascadiaMediumProperties{T<:Real, F, ITP} <: WaterProperties{T} 
    abs_scale::T
    sca_scale::T
    scattering_model::KopelevichScatteringModel{T, F}
    dispersion_model::QuanFryDispersion{T}
    absorption_model::InterpolatedAbsorptionModel{T, ITP}

    function CascadiaMediumProperties(
        hg_g::T,
        pure_water_fraction::T,
        abs_scale::T,
        sca_scale::T,

    ) where {T}
        scattering_function = MixedHGES(T(hg_g), T(CASCADIA_ES_B), T(1)-pure_water_fraction)#
        scattering_model = KopelevichScatteringModel(
            scattering_function,
            KM3NeT_VOL_CONC_LARGE_PART,
            KM3NeT_VOL_CONC_SMALL_PART
        )
        
        dispersion_model = QuanFryDispersion(CASCADIA_SALINITY, CASCADIA_TEMPERATURE, CASCADIA_PRESSURE)

        absorption_model = InterpolatedAbsorptionModel(ABSLENGTHSTRAWFIT[:, 1], ABSLENGTHSTRAWFIT[:, 2])

        return new{T, typeof(scattering_function), typeof(absorption_model.interpolation)}(
            abs_scale,
            sca_scale,
            scattering_model,
            dispersion_model,
            absorption_model)
    end

end


CascadiaMediumProperties() = CascadiaMediumProperties(0.95f0, 0.2f0, 1f0, 1f0)
CherenkovMediumBase.pressure(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_PRESSURE)
CherenkovMediumBase.temperature(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_TEMPERATURE)
CherenkovMediumBase.material_density(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_DENSITY)
CherenkovMediumBase.radiation_length(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_RADIATION_LENGTH)

function CherenkovMediumBase.absorption_length(medium::CascadiaMediumProperties, wavelength, )
    abs_model = get_absorption_model(medium)
    abs_len = absorption_length(abs_model, wavelength)

    return abs_len * medium.abs_scale
end

function CherenkovMediumBase.scattering_length(mediun::CascadiaMediumProperties, wavelength)
    sca_model = get_scattering_model(medium)
    sca_len = scattering_length(sca_model, wavelength)

    return sca_len * medium.sca_scale
end

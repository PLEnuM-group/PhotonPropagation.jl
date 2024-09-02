
# Implementation of WaterProperties

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

struct AbsLengthStrawFromFit
    df::DataFrame
end

const ABSLENGTHSTRAWFIT = AbsLengthStrawFromFit(
    DataFrame(Parquet2.Dataset(joinpath(pkgdir(@__MODULE__), "assets/attenuation_length_straw_fit.parquet"))))

function (f::AbsLengthStrawFromFit)(wavelength::Real)
    T = typeof(wavelength)
    x::Vector{T} = f.df[:, :wavelength]
    y::Vector{T} = f.df[:, :abs_len]
    fast_linear_interp(wavelength, x, y)
end




struct CascadiaMediumProperties{T<:Real} <: WaterProperties{T} 
    mean_scattering_angle::T
    abs_scale::T
    sca_scale::T
end


CascadiaMediumProperties() = CascadiaMediumProperties(0.92f0, 1f0, 1f0)

const CASCADIA_SALINITY = 34.82 # permille
const CASCADIA_TEMPERATURE = 1.8 # °C
const CASCADIA_DENSITY = DIPPR105(CASCADIA_TEMPERATURE + 273.15) # kg/m^3
const CASCADIA_PRESSURE = 265.917473476 # atm
const CASCADIA_RADIATION_LENGTH = 36.08 # g/cm^2
const CASCADIA_QUAN_FRY_PARAMS::Tuple = calc_quan_fry_params(CASCADIA_SALINITY, CASCADIA_TEMPERATURE, CASCADIA_PRESSURE)

vol_conc_small_part(::CascadiaMediumProperties{T}) where {T<:Real} = T(7.5E-3)
vol_conc_large_part(::CascadiaMediumProperties{T}) where {T<:Real} = T(7.5E-3)
salinity(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_SALINITY)

AbstractMediumProperties.pressure(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_PRESSURE)
AbstractMediumProperties.temperature(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_TEMPERATURE)
AbstractMediumProperties.material_density(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_DENSITY)
AbstractMediumProperties.radiation_length(::CascadiaMediumProperties{T}) where {T<:Real} = T(CASCADIA_RADIATION_LENGTH)
AbstractMediumProperties.mean_scattering_angle(x::CascadiaMediumProperties{T}) where {T<:Real} = x.mean_scattering_angle
AbstractMediumProperties.phase_refractive_index(wavelength::T, medium::CascadiaMediumProperties) where {T<:Real} = refractive_index_fry(
    wavelength,
    T.(CASCADIA_QUAN_FRY_PARAMS)
)

AbstractMediumProperties.dispersion(wavelength::T, medium::CascadiaMediumProperties) where {T<:Real} = dispersion_fry(
    wavelength,
    T.(CASCADIA_QUAN_FRY_PARAMS)
)

function AbstractMediumProperties.absorption_length(wavelength, medium::CascadiaMediumProperties)
    return oftype(wavelength, ABSLENGTHSTRAWFIT(wavelength) * medium.abs_scale)
end

@inline function AbstractMediumProperties.scattering_length(wavelength::Real, medium::CascadiaMediumProperties)
    return sca_len_part_conc(
        wavelength;
        vol_conc_small_part=vol_conc_small_part(medium),
        vol_conc_large_part=vol_conc_large_part(medium)) * medium.sca_scale
end

AbstractMediumProperties.scattering_function(medium::CascadiaMediumProperties) = hg_scattering_func(mean_scattering_angle(medium))



#=


"""
    WaterProperties{T<:Real} <: MediumProperties{T}

Properties for a water-like medium. Use unitful constructor to create a value of this type.

### Fields:
-salinity -- Salinity (permille)
-pressure -- Pressure (atm)
-temperature -- Temperature (°C)
-vol_conc_small_part -- Volumetric concentrations of small particles (ppm)
-vol_conc_large_part -- Volumetric concentrations of large particles (ppm)
-radiation_length -- Radiation length (g/cm^2)
-density -- Density (kg/m^3)
-mean_scattering_angle -- Cosine of the mean scattering angle
-abs_scale -- Scaling factor for the absorption length
-sca_scale -- Scaling factor for the scattering length
"""
struct WaterProperties{T<:Real} <: MediumProperties
    salinity::T # permille
    pressure::T # atm
    temperature::T #°C
    vol_conc_small_part::T # ppm
    vol_conc_large_part::T # ppm
    radiation_length::T # g / cm^2
    density::T # kg/m^3
    mean_scattering_angle::T
    quan_fry_params::Tuple{T,T,T,T}
    abs_scale::T
    sca_scale::T

    WaterProperties(::T, ::T, ::T, ::T, ::T, ::T, ::T, ::T, ::Tuple{T,T,T,T}, ::T, ::T) where {T} = error("Use unitful constructor")

    @doc """
            function WaterProperties(
                salinity::Unitful.Quantity{T},
                pressure::Unitful.Quantity{T},
                temperature::Unitful.Quantity{T},
                vol_conc_small_part::Unitful.Quantity{T},
                vol_conc_large_part::Unitful.Quantity{T},
                radiation_length::Unitful.Quantity{T}
            ) where {T<:Real}
        Construct a `WaterProperties` type.

        The constructor uses DIPPR105 to calculate the density at the given temperature.
        Parameters for the Quan-Fry parametrisation of the refractive index are calculated
        for the given salinity, temperature and pressure.
    """
    function WaterProperties(
        salinity::Unitful.Quantity{T},
        pressure::Unitful.Quantity{T},
        temperature::Unitful.Quantity{T},
        vol_conc_small_part::Unitful.Quantity{T},
        vol_conc_large_part::Unitful.Quantity{T},
        radiation_length::Unitful.Quantity{T},
        mean_scattering_angle::T,
        abs_scale=1,
        sca_scale=1
    ) where {T<:Real}
        salinity = ustrip(T, u"permille", salinity)
        temperature = ustrip(T, u"°C", temperature)
        pressure = ustrip(T, u"atm", pressure)
        quan_fry_params = _calc_quan_fry_params(salinity, temperature, pressure)
        density = DIPPR105(temperature + 273.15)

        new{T}(
            salinity,
            pressure,
            temperature,
            ustrip(T, u"ppm", vol_conc_small_part),
            ustrip(T, u"ppm", vol_conc_large_part),
            ustrip(T, u"g/cm^2", radiation_length),
            density,
            mean_scattering_angle,
            quan_fry_params,
            T(abs_scale),
            T(sca_scale)
        )
    end
end

StructTypes.StructType(::Type{<:WaterProperties}) = StructTypes.Struct()


"""
    make_cascadia_medium_properties(::Type{T}) where {T<:Real}
Construct `WaterProperties` with properties from Cascadia Basin of numerical type `T`.
"""
make_cascadia_medium_properties(
    mean_scattering_angle::T,
    abs_scale=1,
    sca_scale=1) where {T<:Real} = WaterProperties(
    T(34.82)u"permille",
    T(269.44088)u"bar",
    T(1.8)u"°C",
    T(0.0075)u"ppm",
    T(0.0075)u"ppm",
    T(36.08)u"g/cm^2",
    mean_scattering_angle,
    abs_scale,
    sca_scale)

salinity(x::WaterProperties) = x.salinity
pressure(x::WaterProperties) = x.pressure
temperature(x::WaterProperties) = x.temperature
material_density(x::WaterProperties) = x.density
vol_conc_small_part(x::WaterProperties) = x.vol_conc_small_part
vol_conc_large_part(x::WaterProperties) = x.vol_conc_large_part
radiation_length(x::WaterProperties) = x.radiation_length

=#





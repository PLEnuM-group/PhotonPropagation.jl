module Medium
using Unitful
using Base: @kwdef
using PhysicalConstants.CODATA2018
using Parquet
using DataFrames
using StructTypes
using PhysicsTools

export make_cascadia_medium_properties
export salinity, pressure, temperature, vol_conc_small_part, vol_conc_large_part, radiation_length, material_density
export refractive_index, scattering_length, absorption_length, dispersion, group_velocity, cherenkov_angle
export c_at_wl
export mean_scattering_angle
export MediumProperties, WaterProperties


Unitful.register(Medium)

const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)
abstract type MediumProperties{T<:Real} end

include("water_properties.jl")


# Interface for MediumProperties
salinity(::T) where {T<:MediumProperties} = error("Not implemented for $T")
pressure(::T) where {T<:MediumProperties} = error("Not implemented for $T")
temperature(::T) where {T<:MediumProperties} = error("Not implemented for $T")
material_density(::T) where {T<:MediumProperties} = error("Not implemented for $T")
vol_conc_small_part(::T) where {T<:MediumProperties} = error("Not implemented for $T")
vol_conc_large_part(::T) where {T<:MediumProperties} = error("Not implemented for $T")
radiation_length(::T) where {T<:MediumProperties} = error("Not implemented for $T")
mean_scattering_angle(::T) where {T<:MediumProperties} = error("Not implemented for $T")

"""
    scattering_length(wavelength, medium::MediumProperties) 
Return scattering length at `wavelength`.

`wavelength` is expected to be in units nm. Returned length is in units m.
"""
scattering_length(wavelength, medium::MediumProperties) = error("Not implemented for $(typeof(medium))")



"""
    refractive_index(wavelength, medium)
Return the refractive index at `wavelength`.

`wavelength` is expected to be in units nm.
"""
refractive_index(wavelength, medium::MediumProperties) = error("Not implemented for $(typeof(medium))")


"""
    refractive_index(wavelength, medium)
Return the dispersion at `wavelength`.

`wavelength` is expected to be in units nm.
"""
dispersion(wavelength, medium::MediumProperties) = error("Not implemented for $(typeof(medium))")


"""
    cherenkov_angle(wavelength, medium::MediumProperties)
Calculate the cherenkov angle (in rad) for `wavelength`.

`wavelength` is expected to be in units nm.
"""
function cherenkov_angle(wavelength, medium::MediumProperties)
    return acos(one(typeof(wavelength)) / refractive_index(wavelength, medium))
end

"""
    refractive_index(wavelength, medium)
Return the group_velocity in m/ns at `wavelength`.

`wavelength` is expected to be in units nm.
"""
function group_velocity(wavelength::T, medium::MediumProperties) where {T<:Real}
    global c_vac_m_ns
    ref_ix::T = refractive_index(wavelength, medium)
    λ_0::T = ref_ix * wavelength
    T(c_vac_m_ns) / (ref_ix - λ_0 * dispersion(wavelength, medium))
end

"""
    c_at_wl(wavelength, medium)
Return speed of light in `medium` (m/ns) at wavelength `wavelength`.

`wavelength` is expected to be in units nm.
"""
function c_at_wl(wavelength, medium::MediumProperties)
    c_vac = ustrip(u"m/ns", SpeedOfLightInVacuum)
    return c_vac / refractive_index(wavelength, medium)
end

"""
    absorption_length(wavelength, medium::MediumProperties)
Return absorption length (in m) at `wavelength`.

`wavelength` is expected to be in units nm.
"""
absorption_length(wavelength, medium::MediumProperties) = error("Not implemented for $(typeof(medium))")


end # Module

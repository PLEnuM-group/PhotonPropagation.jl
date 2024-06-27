module Medium
using Unitful
using Base: @kwdef
using PhysicalConstants.CODATA2018
using Parquet
using DataFrames
using StructTypes
using PhysicsTools
using AbstractMediumProperties

export make_cascadia_medium_properties
export make_homogenous_clearice_properties
export salinity, pressure, temperature, vol_conc_small_part, vol_conc_large_part, radiation_length, material_density
export phase_refractive_index, scattering_length, absorption_length, dispersion, group_velocity, cherenkov_angle, group_refractive_index
export mean_scattering_angle
export MediumProperties, WaterProperties, HomogenousIceProperties, SimpleMediumProperties
export scattering_function
export mixed_hg_sl_scattering_func_ppc

const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)


"""
    hg_scattering_func(g::Real)

CUDA-optimized version of Henyey-Greenstein scattering in one plane.

# Arguments
- `g::Real`: mean scattering angle

# Returns
- `typeof(g)` cosine of a scattering angle sampled from the distribution

"""
@inline function hg_scattering_func(g::T) where {T <: Real}
    """Henyey-Greenstein scattering in one plane."""
    eta = rand(T)
    costheta::T = (1 / (2 * g) * (1 + g^2 - ((1 - g^2) / (1 + g * (2 * eta - 1)))^2))
    #costheta::T = (1 / (2 * g) * (fma(g, g, 1) - (fma(-g, g, 1) / (fma(g, (fma(2, eta, -1)), 1)))^2))
    return clamp(costheta, T(-1), T(1))



end

"""
    sl_scattering_func(g::Real)
Simplified-Liu scattering angle function.
Implementation from: https://user-web.icecube.wisc.edu/~dima/work/WISC/ppc/spice/new/paper/a.pdf

# Arguments
- `g::Real`: mean scattering angle
"""
function sl_scattering_func(g::T) where {T <: Real}
    eta = rand(T)
    beta = (1-g) / (1+g)
    costheta::T = 2 * eta^beta - 1
    return clamp(costheta, T(-1), T(1))
end



"""
    mixed_hg_sl_scattering_func(g, hg_fraction)
Mixture model of HG and SL.

# Arguments
- `g::Real`: mean scattering angle
- `hg_fraction::Real`: mixture weight of the HG component
"""
function mixed_hg_sl_scattering_func(g::Real, hg_fraction::Real)
    choice = rand()
    if choice < hg_fraction
        return hg_scattering_func(g)
    end
    return sl_scattering_func(g)
end

function mixed_hg_sl_scattering_func_ppc(g::T, hg_fraction::T) where {T <:Real}
    xi = rand()
    sf = hg_fraction
    gr::T = (1-g)/(1+g)
	if(xi>sf)
	  xi=(1-xi)/(1-sf)
	  xi=2*xi-1
	  if(g!=0)
	    ga::T=(1-g*g)/(1+g*xi)
	    xi=(1+g*g-ga*ga)/(2*g)
      end
	else
	  xi/=sf
	  xi=2*xi^gr-1
    end
    return clamp(xi, T(-1), T(1))
end


include("water_properties.jl")
include("ice_properties.jl")
include("simple_medium.jl")


end # Module

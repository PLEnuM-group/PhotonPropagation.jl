export SimpleMediumProperties

Base.@kwdef struct SimpleMediumProperties{T <: Real} <: MediumProperties
    absorption_length::T
    scattering_length::T
    phase_refractive_index::T
    mean_scattering_angle::T
    group_velocity::T
end

CherenkovMediumBase.absorption_length(medium::SimpleMediumProperties, ::Real, ) = medium.absorption_length
CherenkovMediumBase.scattering_length(medium::SimpleMediumProperties, ::Real, ) = medium.scattering_length
CherenkovMediumBase.phase_refractive_index(::Real, medium::SimpleMediumProperties) = medium.phase_refractive_index
CherenkovMediumBase.sample_scattering_function(medium::SimpleMediumProperties) = hg_scattering_func(mean_scattering_angle(medium))
CherenkovMediumBase.group_velocity(medium::SimpleMediumProperties, ::Real, ) = medium.group_velocity

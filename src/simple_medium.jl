export SimpleMediumProperties

Base.@kwdef struct SimpleMediumProperties{T <: Real} <: MediumProperties
    absorption_length::T
    scattering_length::T
    phase_refractive_index::T
    mean_scattering_angle::T
    group_velocity::T
end

AbstractMediumProperties.absorption_length(::Real, medium::SimpleMediumProperties) = medium.absorption_length
AbstractMediumProperties.scattering_length(::Real, medium::SimpleMediumProperties) = medium.scattering_length
AbstractMediumProperties.phase_refractive_index(::Real, medium::SimpleMediumProperties) = medium.phase_refractive_index
AbstractMediumProperties.mean_scattering_angle(medium::SimpleMediumProperties) = medium.mean_scattering_angle
AbstractMediumProperties.scattering_function(medium::SimpleMediumProperties) = hg_scattering_func(mean_scattering_angle(medium))
AbstractMediumProperties.group_velocity(::Real, medium::SimpleMediumProperties) = medium.group_velocity

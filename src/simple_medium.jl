Base.@kwdef struct SimpleMediumProperties{T <: Real} <: MediumProperties{T}
    absorption_length::T
    scattering_length::T
    phase_refractive_index::T
    mean_scattering_angle::T
    group_velocity::T
end

absorption_length(::Real, medium::SimpleMediumProperties) = medium.absorption_length
scattering_length(::Real, medium::SimpleMediumProperties) = medium.scattering_length
phase_refractive_index(::Real, medium::SimpleMediumProperties) = medium.phase_refractive_index
mean_scattering_angle(medium::SimpleMediumProperties) = medium.mean_scattering_angle
scattering_function(medium::SimpleMediumProperties) = hg_scattering_func(mean_scattering_angle(medium))
group_velocity(::Real, medium::SimpleMediumProperties) = medium.group_velocity

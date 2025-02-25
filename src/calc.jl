
module Calc
using CherenkovMediumBase
using NeutrinoTelescopeBase
using PhysicalConstants.CODATA2018
using PhysicsTools
using Unitful
using DataFrames
using LinearAlgebra
using ..Medium
using ..Detection
using ..LightYield
using ..PhotonPropagationSetup


const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)

export calc_time_residual!, calc_tgeo
export calc_time_residual_tracks!, calc_tgeo_tracks
export shift_particle, shift_to_closest_approach
export closest_approach_distance, closest_approach_param



"""
    calc_tgeo(distance::Real, c_n::Number)

Calculate the geometric time delay for a photon traveling a given distance in a medium with a given speed of light.

# Arguments
- `distance`: The distance the photon travels, in meters.
- `c_n`: The speed of light in the medium, in meters per second.

# Returns
- The geometric time delay, in seconds.
"""
calc_tgeo(distance::Real, c_n::Number) = distance / c_n

"""
    calc_tgeo(distance::Real, medium::MediumProperties)

Calculate the geometric time delay for a photon traveling a given distance in a medium.

# Arguments
- `distance`: The distance the photon travels, in meters.
- `medium`: The properties of the medium, which should be an instance of the `MediumProperties` type.

# Returns
- The geometric time delay, in seconds.
"""
calc_tgeo(distance::Real, medium::MediumProperties) = calc_tgeo(distance, group_velocity(800.0, medium))
"""
    calc_tgeo(distance::Real, target::PhotonTarget{<:Spherical}, c_n_or_medium)

Calculate the geometric time delay for a photon traveling a given distance to a spherical target.

# Arguments
- `distance`: The distance the photon travels.
- `target`: The target the photon is traveling to. It must be a `PhotonTarget` with a `Spherical` shape.
- `c_n_or_medium`: The refractive index of the medium or an instance of `MediumProperties`.

# Returns
- The geometric time delay.
"""
function calc_tgeo(distance::Real, target::PhotonTarget{<:Spherical}, c_n_or_medium)
    return  calc_tgeo(distance - target.shape.radius, c_n_or_medium)
end

"""
    closest_approach_distance(p0, dir, pos)

Calculate the closest approach distance between a point and a line.

# Arguments
- `p0`: The origin point of the line.
- `dir`: The direction of the line.
- `pos`: The point to calculate the closest approach distance to.

# Returns
- The closest approach distance.
"""
function closest_approach_distance(p0, dir, pos)
    return norm(cross((pos .- p0), dir))
end

"""
    closest_approach_distance(particle::Particle, pos::AbstractArray)

Calculate the closest approach distance between a particle and a position.

# Arguments
- `particle`: The particle.
- `pos`: The position.

# Returns
- The closest approach distance.
"""
function closest_approach_distance(particle::Particle, pos::AbstractArray)
    if particle_shape(particle) == Track()
        return closest_approach_distance(particle.position, particle.direction, pos)
    elseif particle_shape(particle) == Cascade()
        return norm(particle.position .- pos)
    end
end

"""
    closest_approach_distance(particle, target::PhotonTarget)

Calculate the closest approach distance between a particle and a target.

# Arguments
- `particle`: The particle.
- `target`: The target.

# Returns
- The closest approach distance.
"""
function closest_approach_distance(particle, target::PhotonTarget)
    return closest_approach_distance(particle, target.shape.position)   
end

"""
    calc_tgeo_tracks(p0, dir, pos, n_ph, n_grp)

Calculate the geometric time delay for a track.

# Arguments
- `p0`: The origin point of the track.
- `dir`: The direction of the track.
- `pos`: The position to calculate the time delay at.
- `n_ph`: The phase refractive index.
- `n_grp`: The group refractive index.

# Returns
- The geometric time delay, in nanoseconds.
"""
function calc_tgeo_tracks(p0, dir, pos, n_ph, n_grp)
    dist = closest_approach_distance(p0, dir, pos)
    dpos = pos .- p0
    t_geo = 1 / c_vac_m_ns * (dot(dir, dpos) + dist * (n_grp * n_ph - 1) / sqrt((n_ph^2 - 1)))
    return t_geo
end

"""
    calc_tgeo_tracks(p0, dir, pos, medium::MediumProperties, wavelength=800.)

Calculate the geometric time delay for a track in a medium.

# Arguments
- `p0`: The origin point of the track.
- `dir`: The direction of the track.
- `pos`: The position to calculate the time delay at.
- `medium`: The properties of the medium, which should be an instance of the `MediumProperties` type.
- `wavelength`: The wavelength of the photon, in nanometers. Default is 800 nm.

# Returns
- The geometric time delay, in nanoseconds.

# Notes
- The function calculates the phase and group refractive indices based on the provided wavelength and medium properties.
- It then calls the `calc_tgeo_tracks` function with the calculated refractive indices to compute the geometric time delay.
"""
function calc_tgeo_tracks(p0, dir, pos, medium::MediumProperties, wavelength=800.)
    n_ph = phase_refractive_index(medium, wavelength)
    n_grp = c_vac_m_ns / group_velocity(medium, wavelength)
    return calc_tgeo_tracks(p0, dir, pos, n_ph, n_grp)
end

"""
    calc_tgeo(particle::Particle, target, medium)

Calculate the geometric time delay for a particle in a medium.

# Arguments
- `particle`: The particle.
- `target`: The target.
- `medium`: The medium properties.

# Returns
- The geometric time delay, in nanoseconds.
"""
function calc_tgeo(particle::Particle, target, medium)
    if particle_shape(particle) == Track()
        return calc_tgeo_tracks(particle.position, particle.direction, target.shape.position, medium)
    else
        return calc_tgeo(norm(particle.position .- target.shape.position), target, medium)
    end
end


"""
    closest_approach_param(p0, dir, pos)

Compute the closest approach parameter between a point `pos` and a line defined by a point `p0` and a direction `dir`.

# Arguments
- `p0`: The starting point of the line.
- `dir`: The direction vector of the line.
- `pos`: The point to compute the closest approach parameter to.

# Returns
The closest approach parameter `d` between the point `pos` and the line.

"""
function closest_approach_param(p0, dir, pos)

    # Vector from pos to p
    a = pos .- p0

    d = dot(a, dir)
    return d
end

"""
    closest_approach_param(particle::Particle, pos)

Compute the closest approach parameter between a particle and a position.

If the particle shape is a Track, the closest approach parameter is computed using the particle's position and direction.
If the particle shape is not a Track, the closest approach parameter is set to 0.

# Arguments
- `particle::Particle`: The particle object.
- `pos`: The position to compute the closest approach parameter to.

# Returns
- The closest approach parameter `d` between the point `pos` and the particle.
"""
function closest_approach_param(particle::Particle, pos)
    if particle_shape(particle) == Track()
        return closest_approach_param(particle.position, particle.direction, pos)
    else
        return 0.
    end
end

"""
    shift_particle(particle::Particle{T}, param) where {T <: Real}

Shift particle along its direction by `param` (in units of m)
"""
function shift_particle(particle::Particle{T}, param) where {T <: Real}
    pos_along = particle.position .+ (param .* particle.direction)

    t::T = particle.time .+ param / c_vac_m_ns
    return Particle(pos_along, particle.direction, t, particle.energy, particle.length, particle.type)
end

"""
    shift_to_closest_approach(particle::Particle, pos::AbstractVector)

Shifts the particle to its closest approach to a given position.

# Arguments
- `particle::Particle`: The particle to be shifted.
- `pos::AbstractVector`: The position to which the particle is shifted.

# Returns
- `Particle`: The shifted particle.

"""
function shift_to_closest_approach(particle::Particle{T}, pos::AbstractVector) where {T <: Real}
    d::T = closest_approach_param(particle.position, particle.direction, pos)
    return shift_particle(particle, d)
end
    


"""
    calc_time_residual_tracks!(df::AbstractDataFrame, setup::PhotonPropSetup)

Calculate the time residual for each track in the given DataFrame `df` using the provided `setup`.

# Arguments
- `df::AbstractDataFrame`: The DataFrame containing the tracks.
- `setup::PhotonPropSetup`: The setup configuration for photon propagation.

# Details
This function calculates the time residual for each track in the DataFrame `df` by subtracting the geometric time delay from the photon arrival time.
The calculated time of flight is obtained using the source position, direction, target position, and medium properties from the `setup` configuration.

"""
function calc_time_residual_tracks!(df::AbstractDataFrame, setup::PhotonPropSetup)

    targ_id_map = Dict([target.module_id => target for target in setup.targets])

    t0 = setup.sources[1].time
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        tgeo = calc_tgeo_tracks(
            setup.sources[1].position,
            setup.sources[1].direction,
            target.shape.position,
            setup.medium)

        subdf[!, :tres] = (subdf[:, :time] .- tgeo .- t0)
    end
end


function calc_time_residual_cascades!(df::AbstractDataFrame, setup::PhotonPropSetup)

    targ_id_map = Dict([target.module_id => target for target in setup.targets])

    t0 = setup.sources[1].time
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        distance = norm(setup.sources[1].position .- target.shape.position)
        tgeo = calc_tgeo(distance, target, setup.medium)

        subdf[!, :tres] = (subdf[:, :time] .- tgeo .- t0)
    end
end


function calc_time_residual!(df::AbstractDataFrame, setup::PhotonPropSetup)
    if eltype(setup.sources) <: CherenkovTrackEmitter
        return calc_time_residual_tracks!(df, setup)
    else
        return calc_time_residual_cascades!(df, setup)
    end

end
end
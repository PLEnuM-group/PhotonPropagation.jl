module PhotonPropagationCuda
using StaticArrays
using LinearAlgebra
using CUDA
using Random
using SpecialFunctions

using Unitful
using PhysicalConstants.CODATA2018
using StatsBase
using Logging
using PoissonRandom
using Rotations
using StructArrays
using PhysicsTools

export cuda_propagate_photons!, initialize_photon_arrays, process_output
export cuda_propagate_multi_target!, check_intersection
export cherenkov_ang_dist, cherenkov_ang_dist_int
export run_photon_prop_no_local_cache
export PhotonHit


using ..Medium
using ..Spectral
using ..Detection
using ..LightYield


const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)




"""
    uniform(minval::T, maxval::T) where {T}

Convenience function for sampling uniform values in [minval, maxval]
"""
@inline function uniform(minval::T, maxval::T) where {T}
    uni = rand(T)
    return minval + uni * (maxval - minval)
end

"""
    cuda_hg_scattering_func(g::Real)

CUDA-optimized version of Henyey-Greenstein scattering in one plane.

# Arguments
- `g::Real`: mean scattering angle

# Returns
- `typeof(g)` cosine of a scattering angle sampled from the distribution

"""
@inline function cuda_hg_scattering_func(g::Real)
    T = typeof(g)
    """Henyey-Greenstein scattering in one plane."""
    eta = rand(T)
    #costheta::T = (1 / (2 * g) * (1 + g^2 - ((1 - g^2) / (1 + g * (2 * eta - 1)))^2))
    costheta::T = (1 / (2 * g) * (fma(g, g, 1) - (fma(-g, g, 1) / (fma(g, (fma(2, eta, -1)), 1)))^2))
    return clamp(costheta, T(-1), T(1))
end


struct PhotonState{T,U}
    position::SVector{3,T}
    direction::SVector{3,T}
    time::U
    wavelength::T
end


"""
    initialize_direction_isotropic(T::Type)

Sample a direction isotropically

# Arguments
- `T::Type`: desired eltype of the return value

# Returns
- StaticVector{3, T}: Cartesian coordinates of the sampled direction

"""
@inline function initialize_direction_isotropic(T::Type)
    theta = acos(uniform(T(-1), T(1)))
    phi = uniform(T(0), T(2 * pi))
    sph_to_cart(theta, phi)
end


@inline function sample_cherenkov_direction(source::CherenkovEmitter{T}, medium::MediumProperties, wl::T) where {T<:Real}

    # Sample a photon direction. Assumes track is aligned with e_z
    theta_cherenkov = cherenkov_angle(wl, medium)
    phi = uniform(T(0), T(2 * pi))
    ph_dir = sph_to_cart(theta_cherenkov, phi)


    # Sample a direction of a "Cherenkov track". Assumes source direction is e_z
    track_dir::SVector{3,T} = sample_cherenkov_track_direction(T)

    # Rotate photon to track direction
    ph_dir = rot_from_ez_fast(track_dir, ph_dir)

    # Rotate track to source direction
    ph_dir = rot_from_ez_fast(source.direction, ph_dir)

    return ph_dir
end


@inline initialize_wavelength(::T) where {T<:Spectrum} = throw(ArgumentError("Cannot initialize $T"))
@inline initialize_wavelength(spectrum::Monochromatic{T}) where {T} = spectrum.wavelength
@inline initialize_wavelength(spectrum::CherenkovSpectrum{T,P}) where {T,P} = @inbounds spectrum.texture[rand(T)]


@inline function initialize_photon_state(source::PointlikeIsotropicEmitter{T}, ::MediumProperties, spectrum::Spectrum) where {T<:Real}
    wl = initialize_wavelength(spectrum)
    pos = source.position
    dir = initialize_direction_isotropic(T)
    PhotonState(pos, dir, source.time, wl)
end

@inline function initialize_photon_state(source::PointlikeTimeRangeEmitter{T,U}, ::MediumProperties, spectrum::Spectrum) where {T<:Real,U<:Real}
    wl = initialize_wavelength(spectrum)
    pos = source.position
    dir = initialize_direction_isotropic(T)
    time = uniform(source.time_range[1], source.time_range[2])
    PhotonState(pos, dir, time, wl)
end

@inline function initialize_photon_state(source::AxiconeEmitter{T}, ::MediumProperties, spectrum::Spectrum) where {T<:Real}
    wl = initialize_wavelength(spectrum)
    pos = source.position
    phi = uniform(T(0), T(2 * pi))
    theta = source.angle
    dir = rot_from_ez_fast(source.direction, sph_to_cart(theta, phi))

    PhotonState(pos, dir, source.time, wl)
end

@inline function initialize_photon_state(source::PencilEmitter{T}, ::MediumProperties, spectrum::Spectrum) where {T<:Real}
    wl = initialize_wavelength(spectrum)
    if source.beam_divergence > 0

        theta = sqrt(uniform(zero(T), source.beam_divergence^2))
        phi = uniform(zero(T), T(2 * pi))
        dir = rot_from_ez_fast(source.direction, sph_to_cart(theta, phi))
    else
        dir = source.direction
    end
    PhotonState(source.position, dir, source.time, wl)
end


@inline function initialize_photon_state(source::ExtendedCherenkovEmitter{T}, medium::MediumProperties, spectrum::Spectrum) where {T<:Real}
    #wl = initialize_wavelength(source.spectrum)
    wl = initialize_wavelength(spectrum)

    long_pos = rand_gamma(T(source.long_param.a), T(1 / source.long_param.b), Float32) * source.long_param.lrad

    pos::SVector{3,T} = source.position .+ long_pos .* source.direction
    time = source.time + long_pos / T(c_vac_m_ns)

    ph_dir = sample_cherenkov_direction(source, medium, wl)

    PhotonState(pos, ph_dir, time, wl)
end

@inline function initialize_photon_state(source::PointlikeCherenkovEmitter{T}, medium::MediumProperties, spectrum::Spectrum) where {T<:Real}
    #wl = initialize_wavelength(source.spectrum)
    wl = initialize_wavelength(spectrum)

    pos::SVector{3,T} = source.position
    time = source.time

    ph_dir = sample_cherenkov_direction(source, medium, wl)

    PhotonState(pos, ph_dir, time, wl)
end

@inline function initialize_photon_state(source::CherenkovTrackEmitter{T}, medium::MediumProperties, spectrum::Spectrum) where {T<:Real}
    wl = initialize_wavelength(spectrum)

    pos_along = uniform(T(0), T(source.length))
    pos::SVector{3,T} = source.position .+ pos_along .* source.direction
    time = source.time + pos_along / T(c_vac_m_ns)

    #TODO: For tracks this parametrisation is probably suboptimal
    ph_dir = sample_cherenkov_direction(source, medium, wl)

    PhotonState(pos, ph_dir, time, wl)
end



@inline function update_direction(this_dir::SVector{3,T}, medium::MediumProperties) where {T}
    #=
    Update the photon direction using scattering function.
    =#

    # Calculate new direction (relative to e_z)
    cos_sca_theta = cuda_hg_scattering_func(mean_scattering_angle(medium))
    sin_sca_theta = sqrt(fma(-cos_sca_theta, cos_sca_theta, 1))
    sca_phi = uniform(T(0), T(2 * pi))

    sin_sca_phi, cos_sca_phi = sincos(sca_phi)

    new_dir_1::T = cos_sca_phi * sin_sca_theta
    new_dir_2::T = sin_sca_phi * sin_sca_theta
    new_dir_3::T = cos_sca_theta

    rot_from_ez_fast(this_dir, @SVector[new_dir_1, new_dir_2, new_dir_3])

end


@inline function update_position(pos::SVector{3,T}, dir::SVector{3,T}, step_size::T) where {T}
    # update position
    #return @SVector[pos[j] + dir[j] * step_size for j in 1:3]
    #return @SVector[CUDA.fma(step_size, dir[j], pos[j]) for j in 1:3]
    return pos .+ dir .* step_size
end


@inline function check_intersection(target_shape::Spherical, pos::SVector{3,T}, dir::SVector{3,T}, step_size::T) where {T<:Real}
    target_pos = target_shape.position
    target_rsq = target_shape.radius^2

    dpos = pos .- target_pos

    #=
    # Check if intersection is even possible
    if any(((dpos .- target.radius) .> 0) .&& (dir .> 0))
        return false, NaN32
    elseif any(((dpos .+ target.radius) .< 0) .&& (dir .< 0))
        return false, NaN32
    end
    =#

    a::T = dot(dir, dpos)
    pp_norm_sq::T = sum(dpos .^ 2)

    b = fma(a, a, -pp_norm_sq + target_rsq)

    #b::Float32 = a^2 - (pp_norm_sq - target.radius^2)

    isec = b >= 0

    if !isec
        return false, NaN32
    end

    # Uncommon branch
    # Distance of of the intersection point along the line
    d = -a - sqrt(b)

    if (d > 0) & (d < step_size)
        return true, d
    else
        return false, NaN32
    end

end

@inline function check_intersection(target_shape::Rectangular, pos::SVector{3,T}, dir::SVector{3,T}, step_size::T) where {T<:Real}

    dir_normal = dir[3]

    if dir_normal == 0
        return false, NaN32
    end

    d = (target_shape.position[3] - pos[3]) / dir_normal

    if (d < 0) | (d > step_size)
        return false, NaN32
    end

    isec = ((abs((pos[1] + dir[1] * d) - target_shape.position[1]) < target_shape.length_x/2) &
            (abs((pos[2] + dir[2] * d) - target_shape.position[2]) < target_shape.length_y/2))

    if isec
        return true, d
    else
        return false, NaN32
    end

end

@inline function check_intersection(target_shape::Circular, pos::SVector{3,T}, dir::SVector{3,T}, step_size::T) where {T<:Real}

    dir_normal = dir[3]

    if dir_normal == 0
        return false, NaN32
    end

    d = (target_shape.position[3] - pos[3]) / dir_normal

    if (d < 0) | (d > step_size)
        return false, NaN32
    end

    isec = ((pos[1] + dir[1] * d - target_shape.position[1])^2 + (pos[2] + dir[2] * d - target_shape.position[2])^2) < target_shape.radius^2

    if isec
        return true, d
    else
        return false, NaN32
    end

end


struct PhotonHit{T<:Real,U<:Real}
    position::SVector{3,T}
    direction::SVector{3,T}
    initial_direction::SVector{3,T}
    wavelength::T
    time::U
    dist_travelled::T
    module_id::Int64
end


function cuda_propagate_photons!(
    out_hits::StructArray{<:PhotonHit},
    out_stack_pointer::CuDeviceVector{Int64},
    out_n_ph_simulated::CuDeviceVector{Int64},
    out_err_code::CuDeviceVector{Int32},
    seed::Int64,
    source::PhotonSource,
    spectrum::Spectrum,
    targets::CuDeviceVector{<:TargetShape},
    medium::MediumProperties{T},
    nsteps::Int32) where {T}

    block = blockIdx().x
    thread = threadIdx().x
    blockdim = blockDim().x
    griddim = gridDim().x
    # warpsize = CUDA.warpsize()
    # warp_ix = thread % warp
    n_threads_total = (griddim * blockdim)
    global_thread_index::Int32 = (block - Int32(1)) * blockdim + thread

    out_err_code[1] = 0

    Random.seed!(seed + global_thread_index)

   
    this_n_photons, remainder = divrem(source.photons, n_threads_total)

    if global_thread_index <= remainder
        this_n_photons += 1
    end

    n_photons_simulated = Int64(0)

    @inbounds for i in 1:this_n_photons
        # Check if there's enough space in the output vector
        if out_stack_pointer[1] >= length(out_hits)-1
            # not enough space
            out_err_code[1] = 1
            break
        end
        
        photon_state = initialize_photon_state(source, medium, spectrum)

        dir::SVector{3,T} = photon_state.direction
        initial_dir = copy(dir)
        wavelength::T = photon_state.wavelength
        pos::SVector{3,T} = photon_state.position

        time = photon_state.time
        dist_travelled = T(0)

        sca_len::T = scattering_length(wavelength, medium)

        c_grp::T = group_velocity(wavelength, medium)
        
        for nstep in Int32(1):nsteps

            eta = rand(T)
            step_size::T = -log(eta) * sca_len

            # Check intersection with module

            isec = false
            dist_to_target = 0.0f0
            module_ix = 0
            for (ix, target) in enumerate(targets)
                isec, d = check_intersection(target, pos, dir, step_size)

                if isec
                    module_ix = ix
                    dist_to_target = d
                    break
                end
            end

            if !isec
                pos = update_position(pos, dir, step_size)
                dist_travelled += step_size
                time += step_size / c_grp
                dir = update_direction(dir, medium)
                continue
            end

            # Intersected
            pos = update_position(pos, dir, dist_to_target)
            dist_travelled += dist_to_target
            time += dist_to_target / c_grp

            stack_idx::Int64 = CUDA.atomic_add!(pointer(out_stack_pointer, 1), Int64(1))
            CUDA.@cuassert stack_idx <= length(out_hits) "Stack overflow"

            out_hits.position[stack_idx] = pos
            out_hits.direction[stack_idx] = dir
            out_hits.initial_direction[stack_idx] = initial_dir
            out_hits.time[stack_idx] = time
            out_hits.wavelength[stack_idx] = wavelength
            out_hits.dist_travelled[stack_idx] = dist_travelled
            out_hits.module_id[stack_idx] = module_ix

            break
       
        end
        n_photons_simulated += 1

    end

    CUDA.atomic_add!(pointer(out_n_ph_simulated, 1), n_photons_simulated)
    
    
    return nothing

end

#=
function cuda_propagate_photons!(
    out_positions::CuDeviceVector{SVector{3,T}},
    out_directions::CuDeviceVector{SVector{3,T}},
    out_initial_directions::CuDeviceVector{SVector{3,T}},
    out_wavelengths::CuDeviceVector{T},
    out_dist_travelled::CuDeviceVector{T},
    out_times::CuDeviceVector{<:Real},
    out_stack_pointer::CuDeviceVector{Int64},
    out_n_ph_simulated::CuDeviceVector{Int64},
    out_err_code::CuDeviceVector{Int32},
    seed::Int64,
    source::PhotonSource,
    spectrum::Spectrum,
    target::TargetShape,
    medium::MediumProperties{T},
    nsteps::Int32) where {T}

    block = blockIdx().x
    thread = threadIdx().x
    blockdim = blockDim().x
    griddim = gridDim().x
    # warpsize = CUDA.warpsize()
    # warp_ix = thread % warp
    n_threads_total = (griddim * blockdim)
    global_thread_index::Int32 = (block - Int32(1)) * blockdim + thread

    Random.seed!(seed + global_thread_index)

    this_n_photons, remainder = divrem(source.photons, n_threads_total)

    if global_thread_index <= remainder
        this_n_photons += 1
    end

    n_photons_simulated = Int64(0)

    @inbounds for i in 1:this_n_photons

        photon_state = initialize_photon_state(source, medium, spectrum)

        dir::SVector{3,T} = photon_state.direction
        initial_dir = copy(dir)
        wavelength::T = photon_state.wavelength
        pos::SVector{3,T} = photon_state.position

        time = photon_state.time
        dist_travelled = T(0)

        sca_len::T = scattering_length(wavelength, medium)
        c_grp::T = group_velocity(wavelength, medium)


          for nstep in Int32(1):nsteps

            eta = rand(T)
            step_size::T = -log(eta) * sca_len

            # Check intersection with module

            isec = false
            dist_to_target = 0.0f0

            isec, dist_to_target = check_intersection(target, pos, dir, step_size)

            if !isec
                pos = update_position(pos, dir, step_size)
                dist_travelled += step_size
                time += step_size / c_grp
                dir = update_direction(dir, medium)
                continue
            end

            # Intersected
            pos = update_position(pos, dir, dist_to_target)
            dist_travelled += dist_to_target
            time += dist_to_target / c_grp

            stack_idx::Int64 = CUDA.atomic_add!(pointer(out_stack_pointer, 1), Int64(1))
            CUDA.@cuassert stack_idx <= length(out_positions) "Stack overflow"

            out_positions[stack_idx] = pos
            out_directions[stack_idx] = dir
            out_initial_directions[stack_idx] = initial_dir
            out_dist_travelled[stack_idx] = dist_travelled
            out_wavelengths[stack_idx] = wavelength
            out_times[stack_idx] = time
            break
        end

        n_photons_simulated += 1

    end

    CUDA.atomic_add!(pointer(out_n_ph_simulated, 1), n_photons_simulated)
    out_err_code[1] = 0

    return nothing

end
=#

function process_output(output::AbstractVector{T}, stack_pointers::AbstractVector{<:Integer}) where {T}

    output = Vector(output)

    out_size = length(output)
    stack_len = Int64(out_size / length(stack_pointers))

    stack_starts = 1:stack_len:out_size
    out_sum = sum(stack_pointers .% stack_len)

    coalesced = Vector{T}(undef, out_sum)
    ix = 1
    for i in eachindex(stack_pointers)
        sp = stack_pointers[i]
        if sp == 0
            continue
        end
        this_len = (sp - stack_starts[i]) + 1
        coalesced[ix:ix+this_len-1, :] = output[stack_starts[i]:sp]
        #println("$(stack_starts[i]:sp) to $(ix:ix+this_len-1)")
        ix += this_len

    end
    coalesced
end

function process_output(output::AbstractVector{T}, stack_pointer::Integer) where {T}
    return Vector(output[1:stack_pointer-1])
end



function initialize_photon_arrays(stack_length::Integer, blocks, type::Type)
    (
        CuVector(zeros(SVector{3,type}, stack_length * blocks)), # position
        CuVector(zeros(SVector{3,type}, stack_length * blocks)), # direction
        CuVector(zeros(SVector{3,type}, stack_length * blocks)), # initial
        CuVector(zeros(type, stack_length * blocks)), # wavelength
        CuVector(zeros(type, stack_length * blocks)), # dist travelled
        CuVector(zeros(type, stack_length * blocks)), # time
        CuVector(zeros(Int64, blocks)), # stack_idx
        CuVector(zeros(Int64, 1)) # nphotons_simulated
    )
end


function initialize_photon_arrays(stack_length::Integer, type::Type, time_type::Type)
    (
        CuVector(zeros(SVector{3,type}, stack_length)), # position
        CuVector(zeros(SVector{3,type}, stack_length)), # direction
        CuVector(zeros(SVector{3,type}, stack_length)), # initial direction
        CuVector(zeros(type, stack_length)), # wavelength
        CuVector(zeros(type, stack_length)), # dist travelled
        CuVector(zeros(time_type, stack_length)), # time
        CuVector(ones(Int64, 1)), # stack_idx
        CuVector(zeros(Int64, 1)) # nphotons_simulated
    )
end


function calc_shmem(block_size)
    block_size * 7 * sizeof(Float32) #+ 3 * sizeof(Float32)
end


function calculate_gpu_memory_usage(stack_length, blocks)
    return sizeof(SVector{3,Float32}) * 3 * stack_length * blocks +
           sizeof(Float32) * 3 * stack_length * blocks +
           sizeof(Int32) * blocks +
           sizeof(Int64)
end

function calculate_max_stack_size(total_mem, pos_type, time_type)
    one_event = sizeof(PhotonHit{pos_type,time_type})
    return Int64(fld(total_mem - 2 * (sizeof(Int64)), one_event))
end


function run_photon_prop_no_local_cache(
    sources::AbstractVector{<:PhotonSource},
    targets::AbstractVector{<:TargetShape},
    medium::MediumProperties,
    spectrum::Spectrum,
    seed::Int64;
    time_type::Type=Float32,
    n_steps=15)
    avail_mem = CUDA.totalmem(collect(CUDA.devices())[1])
    max_total_stack_len = calculate_max_stack_size(0.8 * avail_mem, Float32, time_type)

    photon_hits_cpu = StructArray{PhotonHit{Float32,time_type}}(undef, max_total_stack_len)
    photon_hits_gpu = replace_storage(CuVector, photon_hits_cpu)

    stack_idx = CuVector(ones(Int64, 1))
    n_ph_sim = CuVector(zeros(Int64, 1))
    err_code = CuVector(zeros(Int32, 1))
    targets = CuVector(targets)


    kernel = @cuda launch = false cuda_propagate_photons!(
        photon_hits_gpu, stack_idx, n_ph_sim, err_code, seed,
        sources[1], spectrum, targets, medium, Int32(n_steps))

    blocks, threads = CUDA.launch_configuration(kernel.fun)

    # Can assign photons to sources by keeping track of stack_idx for each source
    for source in sources
        CUDA.@sync begin
            kernel(
                photon_hits_gpu, stack_idx, n_ph_sim, err_code, seed,
                source, spectrum, targets, medium, Int32(n_steps); threads=threads, blocks=blocks)
        end
        err_code = Vector(err_code)[1]
        # We ran out of memory
        if err_code == 1
            @error "Photon prop ran out of memory. Photon stats may be reduced"
        end

        
    end

    stack_idx = Vector(stack_idx)[1]
    n_ph_sim = Vector(n_ph_sim)[1]
    
    #photon_hits = replace_storage(Vector, photon_hits[1:stack_idx-1])

    # Copy back to CPU without reallocating
    for (key, s) in pairs(StructArrays.components(photon_hits_gpu))
        copyto!(StructArrays.component(photon_hits_cpu, key), s)
    end
    photon_hits_cpu = photon_hits_cpu[1:stack_idx-1]

    #photon_hits = replace_storage(Vector, photon_hits)[1:stack_idx-1]

    return photon_hits_cpu, n_ph_sim
end




end # module

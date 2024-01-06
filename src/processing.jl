module Processing

using DataFrames
using Rotations
using LinearAlgebra
using StaticArrays
using StructTypes
using Distributions
using ..Medium
using ..Spectral
using ..Detection
using ..LightYield
using ..PhotonPropagationCuda


export PhotonPropSetup
export make_hits_from_photons, propagate_photons
export calc_total_weight!
export calc_number_of_steps

mutable struct PhotonPropSetup{SV<:AbstractVector{<:PhotonSource},ST<:AbstractVector{<:PhotonTarget},M<:MediumProperties,C<:SpectralDist}
    sources::SV
    targets::ST
    medium::M
    spec_dist::C
    seed::Int64
end

function PhotonPropSetup(sources::AbstractVector{<:PhotonSource}, targets::AbstractVector{<:PhotonTarget}, medium, spectrum::Spectrum{<:InterpolatedSpectralDist}, seed) 
    cuda_spectral_dist = make_cuda_spectral_dist(spectrum.spectral_dist, spectrum.wl_range)
    return PhotonPropSetup(sources, targets, medium, cuda_spectral_dist, seed)
end

function PhotonPropSetup(sources::AbstractVector{<:PhotonSource}, targets::AbstractVector{<:PhotonTarget}, medium, spectrum::Spectrum, seed) 
    return PhotonPropSetup(sources, targets, medium, spectrum.spectral_dist, seed)
end

function PhotonPropSetup(
    source::PhotonSource,
    target::PhotonTarget,
    medium,
    spectrum,
    seed)

    setup = PhotonPropSetup([source], [target], medium, spectrum, Int64(seed))
    return setup
end


function calc_total_weight!(df::AbstractDataFrame, setup::PhotonPropSetup)

    if nrow(df) == 0
        return df
    end

    abs_length = absorption_length.(df[:, :wavelength], Ref(setup.medium))
    df[!, :abs_weight] = exp.(-Float64.(df[:, :dist_travelled] ./ abs_length))
    #df[!, :ref_ix] = refractive_index.(df[:, :wavelength], Ref(setup.medium))
    df[!, :total_weight] = df[:, :base_weight] .* df[:, :abs_weight]

    return df
end


function propagate_photons(setup::PhotonPropSetup, steps=15; reserved_memory_fraction=0.7)
    hit_buffer_cpu, hit_buffer_gpu = make_hit_buffers(Float32, reserved_memory_fraction)
    return propagate_photons(setup, hit_buffer_cpu, hit_buffer_gpu, steps)
end

function propagate_photons(setup::PhotonPropSetup, hit_buffer_cpu, hit_buffer_gpu, steps=15; copy_output=false)   

    hits, n_ph_sim = run_photon_prop_no_local_cache!(
        setup.sources, [targ.shape for targ in setup.targets], setup.medium, setup.spec_dist, setup.seed,
        hit_buffer_cpu, hit_buffer_gpu, n_steps=steps,
        )


    df = DataFrame(hits, copycols=copy_output)

    mod_ids = [targ.module_id for targ in setup.targets]
    df[!, :module_id] .= mod_ids[df[:, :module_id]]
    df[!, :base_weight] .= n_ph_sim / sum([source.photons for source in setup.sources])
    calc_total_weight!(df, setup)

    return df
end



"""
    function make_hits_from_photons(
        df::AbstractDataFrame,
        setup::PhotonPropSetup,
        target_orientation::AbstractMatrix{<:Real}
    )

Convert photons to pmt_hits.
"""
function make_hits_from_photons(
    df::AbstractDataFrame,
    setup::PhotonPropSetup,
    target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I),
    apply_wl_acc=true)

    return make_hits_from_photons(df, setup.targets, target_orientation, apply_wl_acc)
end


"""
    function make_hits_from_photons(
        df::AbstractDataFrame,
        targets::AbstractArray{<:PhotonTarget},
        target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I))
    )

Convert photons to pmt_hits. Does not yet apply propagation weights.
"""
function make_hits_from_photons(
    df::AbstractDataFrame,
    targets::AbstractArray{<:PhotonTarget},
    target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I),
    apply_wl_acc=true)

    targ_id_map = Dict([target.module_id => target for target in targets])

    if "pos_x" ∉ names(df)
        transform!(df, :position => (p -> reduce(hcat, p)') => [:pos_x, :pos_y, :pos_z])
        transform!(df, :direction => (p -> reduce(hcat, p)') => [:dir_x, :dir_y, :dir_z])
    end

    hits = []
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]

        positions = vec.(eachrow(Matrix(subdf[:, [:pos_x, :pos_y, :pos_z]])))
        directions = vec.(eachrow(Matrix(subdf[:, [:dir_x, :dir_y, :dir_z]])))

        pos::Vector{SVector{3, Float64}} = positions
        dir::Vector{SVector{3, Float64}} = directions
        wl::Vector{Float64} = subdf[:, :wavelength]
        pmt_ids = check_pmt_hit(pos, dir, wl, target, target_orientation)
        mask = pmt_ids .> 0

        if apply_wl_acc
            mask .&= apply_wl_acceptance(pos, dir, wl, target, target_orientation)
        end

        h = DataFrame(copy(subdf[mask, :]))
        h[!, :pmt_id] .= pmt_ids[mask]
        push!(hits, h)

    end

    return reduce(vcat, hits)
end

"""
    calc_number_of_steps(sca_len, cutoff_distance, percentile=0.9)

Calculate the number of photon propagation steps required so that a fraction `percentile`
of all photons has travelled further than cutoff_distance.
"""
function calc_number_of_steps(sca_len, cutoff_distance, percentile=0.9)
    n_steps = ceil(Int64, cutoff_distance / sca_len)
    while true
        d = Erlang(n_steps, sca_len)
        if quantile(d, 1-percentile) > cutoff_distance
            break
        end
        n_steps += 1
    end
    return n_steps
end



end
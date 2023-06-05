module Processing

using DataFrames
using Rotations
using LinearAlgebra
using StaticArrays
using StructTypes

using ..Medium
using ..Spectral
using ..Detection
using ..LightYield
using ..PhotonPropagationCuda


export PhotonPropSetup
export make_hits_from_photons, propagate_photons
export calc_total_weight!

mutable struct PhotonPropSetup{SV<:AbstractVector{<:PhotonSource},ST<:AbstractVector{<:PhotonTarget},M<:MediumProperties,C<:Spectrum}
    sources::SV
    targets::ST
    medium::M
    spectrum::C
    seed::Int64
    
end

PhotonPropSetup(
    source::PhotonSource,
    target::PhotonTarget,
    medium::MediumProperties,
    spectrum::Spectrum,
    seed) = PhotonPropSetup([source], [target], medium, spectrum, Int64(seed))

function calc_total_weight!(df::AbstractDataFrame, setup::PhotonPropSetup)

    if nrow(df) == 0
        return df
    end

    abs_length = absorption_length.(df[:, :wavelength], Ref(setup.medium))
    df[!, :abs_weight] = convert(Vector{Float64}, exp.(-df[:, :dist_travelled] ./ abs_length))
    df[!, :ref_ix] = refractive_index.(df[:, :wavelength], Ref(setup.medium))
    df[!, :total_weight] = df[:, :base_weight] .* df[:, :abs_weight]

    return df
end

function propagate_photons(setup::PhotonPropSetup)

    hits, n_ph_sim = run_photon_prop_no_local_cache(
        setup.sources, [targ.shape for targ in setup.targets], setup.medium, setup.spectrum, setup.seed)

    df = DataFrame(hits)

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
    target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I))

    return make_hits_from_photons(df, setup.targets, target_orientation)
end


"""
    function make_hits_from_photons(
        df::AbstractDataFrame,
        targets::AbstractArray{<:PhotonTarget},
        target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I))
    )

Convert photons to pmt_hits.
"""
function make_hits_from_photons(
    df::AbstractDataFrame,
    targets::AbstractArray{<:PhotonTarget},
    target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I))

    targ_id_map = Dict([target.module_id => target for target in targets])

    hits = []
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        pos::Vector{SVector{3, Float64}} = subdf[:, :position]
        dir::Vector{SVector{3, Float64}} = subdf[:, :direction]
        wl::Vector{Float64} = subdf[:, :wavelength]
        weight::Vector{Float64} = subdf[:, :total_weight]
        pmt_ids = check_pmt_hit(pos, dir, wl, weight, target, target_orientation)
          
        mask = pmt_ids .> 0
        h = DataFrame(copy(subdf[mask, :]))
        h[!, :pmt_id] .= pmt_ids[mask]
        push!(hits, h)

    end

    return reduce(vcat, hits)
end
end
module Processing

using DataFrames

using ..Medium
using ..Spectral
using ..Detection
using ..LightYield
using ..PhotonPropagationsCudas

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



function propagate_photons(setup::PhotonPropSetup)

    hits, n_ph_sim = run_photon_prop_no_local_cache(
        setup.sources, setup.targets, setup.medium, setup.spectrum, setup.seed)

    df = DataFrame(hits)

    df[!, :base_weight] .= n_ph_sim / sum([source.photons for source in setup.sources])

    return df
end


function calc_total_weight!(df::AbstractDataFrame, setup::PhotonPropSetup)

    targ_id_map = Dict([target.module_id => target for target in setup.targets])

    if nrow(df) == 0
        return df
    end

    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        subdf[!, :area_acc] = area_acceptance.(subdf[:, :position], Ref(target))
    end

    abs_length = absorption_length.(df[:, :wavelength], Ref(setup.medium))
    df[!, :abs_weight] = convert(Vector{Float64}, exp.(-df[:, :dist_travelled] ./ abs_length))
    df[!, :wl_acc] = p_one_pmt_acc.(df[:, :wavelength])
    df[!, :ref_ix] = refractive_index.(df[:, :wavelength], Ref(setup.medium))
    df[!, :total_weight] = df[:, :base_weight] .* df[:, :area_acc] .* df[:, :wl_acc] .* df[:, :abs_weight]

    return df
end

function make_hits_from_photons(
    df::AbstractDataFrame,
    setup::PhotonPropSetup,
    target_orientation::AbstractMatrix{<:Real}=RotMatrix3(I))

    targ_id_map = Dict([target.module_id => target for target in setup.targets])

    hits = []
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        pmt_ids = check_pmt_hit(subdf[:, :position], subdf[:, :direction], target, target_orientation)
        mask = pmt_ids .> 0
        h = DataFrame(copy(subdf[mask, :]))
        h[!, :pmt_id] .= pmt_ids[mask]
        push!(hits, h)

    end

    return reduce(vcat, hits)
end
end
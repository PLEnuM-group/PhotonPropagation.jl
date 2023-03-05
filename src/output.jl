
module Output
using DataFrames
using HDF5
using Base.Iterators
using ..LightYield
using ..Detection
export hist_list_to_dataframe, targets_to_dataframe, event_info_to_dataframe, source_to_namedtuple
export save_event

#=
HDF File Structure

/hits/record_id/[HitsSchema]
/records/record_id/[RecordSchema]
/photon_sources/record_id/source_id/[SourceSchema]
=#


function hist_list_to_dataframe(hit_list, targets, target_mask)
    hits_nt = []
    n_pmt = get_pmt_count(eltype(targets))
    pmt_target_prod = product(1:n_pmt, targets[target_mask])

    for (hits, (pmt_id, target)) in zip(hit_list, pmt_target_prod)
        for hit in hits
            nt = (time=hit, pmt_id=pmt_id, module_id=Int(target.module_id))
            push!(hits_nt, nt)
        end
    end

    return DataFrame(hits_nt)

end

function targets_to_dataframe(targets)

    targets_nt = []

    for target in targets
        for (pmt_id, pmt_coords) in enumerate(eachcol(target.pmt_coordinates))
            nt = (
                module_id=Int(target.module_id),
                pmt_id=pmt_id,
                module_x=target.position[1],
                module_y=target.position[2],
                module_z=target.position[3],
                pmt_theta=pmt_coords[1],
                pmt_phi=pmt_coords[2]
            )
            push!(targets_nt, nt)
        end
    end

    return DataFrame(targets_nt)
end

function event_info_to_dataframe(particle)

    nt = (
        x=particle.position[1],
        y=particle.position[2],
        z=particle.position[3],
        dir_x=particle.direction[1],
        dir_y=particle.direction[2],
        dir_z=particle.direction[3],
        time=particle.time,
        energy=particle.energy,
        length=particle.length)

    return DataFrame([nt])
end



function source_to_namedtuple(source::ExtendedCherenkovEmitter)
    tup = (
        location_x=source.position[1],
        location_y=source.position[2],
        location_z=source.position[3],
        direction_x=source.direction[1],
        direction_y=source.direction[2],
        direction_z=source.direction[3],
        time=source.time,
        n_photons=source.photons
    )
    return tup
end


function _create_or_read_group(fid, name)
    if !haskey(fid, name)
        return create_group(fid, name)
    end
    return fid[name]
end

function save_event(path, event_record)
    event_id = string(event_record[:event_id])
    h5open(path, "cw") do fid
        ghits = _create_or_read_group(fid, "hits")
        ghits[event_id] = event_record[:hits][:, [:time, :module_id, :pmt_id]]

        gsources = _create_or_read_group(fid, "sources")
        gsources_ev = create_group(gsources, event_id)

        source_type_dict = Dict()
        for source in event_record[:sources]
            stype = typeof(source)
            if !haskey(source_type_dict, stype)
                source_type_dict[stype] = []
            end
            push!(source_type_dict[stype], source)
        end

        for (source_type, sources) in source_type_dict
            sources_df = DataFrame(source_to_namedtuple.(sources))
            gsources_ev[string(source_type)] = sources_df
        end
    end
end
end

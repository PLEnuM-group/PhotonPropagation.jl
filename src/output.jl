
module Output
using DataFrame
using Tables
export hist_list_to_dataframe, targets_to_dataframe, event_info_to_dataframe


HitsSchema = Schema(
    [:time, :pmt_id, :module_id],
    [Float64, Int64, Int64]
)

@enum SourceType begin
    ExtendedCherenkovEmitter=0
    PointLikeCherenkovEmitter=1
    BareInfiniteMuon=2
    Bioluminescence=3
end
    

SourceSchema = Schema(
    [:location_x, :location_y, :location_z,
    :orientation_x, :orientation_y, :orientation_z,
    :number_of_photons, :time, :type, :source_id
    ],
    [Float64, Float64, Float64,
     Float64, Float64, Float64,
     Int64, Float64, SourceType, Int64
    ]
)

EventRecordSchema = Schema(
    [:sources_id, ]
)




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
                pmt_theta = pmt_coords[1],
                pmt_phi = pmt_coords[2]
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
end
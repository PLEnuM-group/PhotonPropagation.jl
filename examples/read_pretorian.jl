using DataFrames
using JSON3
using StaticArrays
using PhotonPropagation
using Rotations
using LinearAlgebra
open("/home/wecapstor3/capn/capn100h/pretorian/simulation_output/g_090_a_26_s_56_simType_8_lightAngle_90_detectorsInfo.dat", "r") do hdl

    det_info = JSON3.read(readline(hdl))
    sims = readlines(hdl)

    targets = POM[]
    open(det_info[:input_detectors]) do geomfile 
        modules = JSON3.read.(readlines(geomfile))
        for mod in modules
            target = POM(SVector{3, Float64}(mod[:position]), mod[:detectornr]+1)
            push!(targets, target)
        end
    end

    for sim in sims[2:end]

        sim = Dict(JSON3.read(sim))
        @show sim

        direct_photons = DataFrame(sim[:direct])
        direct_photons[!, :total_weight] = direct_photons[!, :value]  .* det_info[:number_of_photons]

        hybrid_photons = DataFrame(sim[:hist])
        hybrid_photons[!, :total_weight] = hybrid_photons[!, :value]  .* det_info[:number_of_photons]

        photons = vcat(direct_photons, hybrid_photons)

        photons[!, :module_id] .= sim[:detectornr] +1
        photons[!, :wavelength] .= 488.
        photons[!, :position] .= copy.(photons[:, :position])
        photons[!, :direction] .= copy.(photons[:, :direction])

        push!(all_photons, photons)

    end
    photons = reduce(vcat, all_photons)
        
    make_hits_from_photons(photons, targets, RotMatrix3(I))
end
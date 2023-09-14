using StatsBase
using HDF5
using Formatting
export save_hdf!

"""
save_hdf!(
    fname::AbstractString,
    group::AbstractString,
    dataset::Matrix,
    attributes::Dict)

Save a matrix dataset to an HDF5 file with the specified filename, group, and attributes.

Datasets are appended to the group with incrementing dataset identifier, the first being `dataset_1`

### Arguments
- `fname::AbstractString`: The filename of the HDF5 file to save the dataset.
- `group::AbstractString`: The group within the HDF5 file to store the dataset.
- `dataset::Matrix`: The matrix dataset to be saved.
- `attributes::Dict`: A dictionary of attributes to be associated with the dataset.

"""
function save_hdf!(
    fname::AbstractString,
    group::AbstractString,
    dataset::Matrix,
    attributes::Dict)


    if isfile(fname)
        fid = h5open(fname, "r+")
    else
        fid = h5open(fname, "w")
    end

    if !haskey(fid, group)
        g = create_group(fid, group)
        HDF5.attrs(g)["nsims"] = 0
    else
        g = fid[group]
    end


    offset = HDF5.read_attribute(g, "nsims") + 1
    ds_name = format("dataset_{:d}", offset)

    g[ds_name] = dataset# Matrix{Float64}(res.hits[:, [:tres, :pmt_id]])
    f_attrs = HDF5.attrs(g[ds_name])
    for (k, v) in attributes
        f_attrs[k] = v
    end

    HDF5.attrs(g)["nsims"] = offset

    close(fid)
end


function make_setup(
    mode, pos, dir, energy, seed;
    g=0.99f0)

    medium = make_cascadia_medium_properties(g)
    target = POM(SA_F32[0, 0, 0], UInt16(1))
    wl_range = (300.0f0, 800.0f0)

    spectrum = CherenkovSpectrum(wl_range, medium, 30)

    if mode == :extended
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus
        )
        source = ExtendedCherenkovEmitter(particle, medium, wl_range)
    elseif mode == :bare_infinite_track
        length = 400f0
        ppos = pos .- length/2 .* dir
        

        particle = Particle(
            ppos,
            dir,
            0.0f0,
            Float32(energy),
            length,
            PMuMinus
        )

        source = CherenkovTrackEmitter(particle, medium, wl_range)    
    elseif mode == :lightsabre_muon
        length = 400f0
        ppos = pos .- length/2 .* dir
        
        particle = Particle(
            Float32.(ppos),
            Float32.(dir),
            0.0f0,
            Float32(energy),
            length,
            PMuMinus
        )

        source = LightsabreMuonEmitter(particle, medium, wl_range)

    elseif mode == :pointlike_cherenkov
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus)
        source = PointlikeChernekovEmitter(particle, medium, wl_range)
    else
        error("unknown mode $mode")
    end

    setup = PhotonPropSetup(source, target, medium, spectrum, seed)
    return setup

end
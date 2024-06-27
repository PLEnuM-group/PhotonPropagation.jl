using HDF5
using CairoMakie
using NeutrinoTelescopes
using PhotonPropagation
using Rotations
using LinearAlgebra
using DataFrames
using StaticArrays
using PhysicsTools
using Distributions
using PairPlots
using StatsBase
using NeutrinoSurrogateModelData
using PhotonSurrogateModel
using Flux

hit_buffer_cpu, hit_buffer_gpu = make_hit_buffers();

pmt_pos = get_pmt_positions(POM(SA[0., 0., 0.], 1), RotMatrix3(I))

target = POM(SA_F32[0, 0, 0], 1)
medium = make_cascadia_medium_properties(0.95f0)
spectrum = make_cherenkov_spectrum((300f0, 800f0), medium)

model = PhotonSurrogate(em_cascade_time_model(0)...)

input_buffer = create_input_buffer(model, 16, 1)
output_buffer = create_output_buffer(16, 100)
fname = "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_0.hd5"

fid = h5open(fname, "r")

ds_key = "dataset_201"
ds = DataFrame(fid["pmt_hits"][ds_key][:, :], [:time, :pmt_id, :total_weight])
ds_attrs = attrs(fid["pmt_hits"][ds_key])

pos = ds_attrs["distance"] .* sph_to_cart(ds_attrs["pos_theta"], ds_attrs["pos_phi"])
dir = sph_to_cart(ds_attrs["dir_theta"], ds_attrs["dir_phi"])
p = Particle(
    SVector{3, Float32}(pos),
    SVector{3, Float32}(dir),
    0f0,
    Float32(ds_attrs["energy"]),
    0f0,
    PEPlus
)

source = ExtendedCherenkovEmitter(p, medium, spectrum)
setup = PhotonPropSetup(source, target, medium, spectrum, 1)

photons = propagate_photons(setup, hit_buffer_cpu, hit_buffer_gpu)
hits = make_hits_from_photons(photons, setup)
calc_pe_weight!(hits, setup)


hit_list = sample_multi_particle_event!([p], [target], gpu(model), medium, feat_buffer=input_buffer, output_buffer=output_buffer, noise_rate=0)
hits_surrogate = hit_list_to_dataframe(hit_list, [target], [true])

outerjoin(
    combine(groupby(hits_surrogate, :pmt_id), nrow => :hits_surrogate),
    combine(groupby(ds, :pmt_id), :total_weight => sum => :hits_dataset),
    combine(groupby(hits, :pmt_id), :total_weight => sum => :hits_reprop), on=:pmt_id)



for ds in take(fid["pmt_hits"], 500)
    atts = attrs(ds)
    df = DataFrame(ds[:, :], [:time, :pmt_ix, :total_weight])
    #df = DataFrame(ds[:, :],  [:tres, :pos_x, :pos_y, :pos_z, :total_weight])

    nhits = sum(df[:, "total_weight"])

    direction = sph_to_cart(acos(atts["dir_costheta"]), atts["dir_phi"])
    view_angle = rad2deg.(acos(dot([0, 0, -atts["distance"]], direction) / atts["distance"]))



    push!(stats, (distance= atts["distance"], nhits=nrow(df), view_angle=view_angle))

    # per_pmt = combine(groupby(df, :pmt_ix), nrow)
    # pmt_ix_max = per_pmt[argmax(combine(groupby(df, :pmt_ix), nrow)[:, :nrow]), :pmt_ix]
    # pmt_pos_max = pmt_pos[Int(pmt_ix_max)]
end

stats = DataFrame(stats)
stats[!, :loghits] = log10.(stats[:, :nhits])
pairplot(stats[:, [:loghits, :distance, :view_angle]])


h = fit(Histogram,
    (stats[:, :view_angle], log10.(stats[:, :distance])), 
    FrequencyWeights(stats[:, :nhits]),
    (0:0.2:1, 1:0.1:2)
)



heatmap(h.edges[1], h.edges[2], h.weights, axis=(xlabel="cos(θ)", ylabel="log(distance)"))


fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, limits=(0, 200, 1, 1E5))
hexbin!(ax, distances, nhits)



fig


sort(nhits)

close(fid)
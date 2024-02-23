using HDF5
using CairoMakie
using NeutrinoTelescopes
using PhotonPropagation
using Rotations
using LinearAlgebra
using DataFrames
using StaticArrays
using PhysicsTools



fname = "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_0.hd5"
stats = []

hbc, hbg = make_hit_buffers()


fid = h5open(fname, "r")

d = fid["pmt_hits/dataset_2"]

evt_attrs = attrs(d)

pos = evt_attrs["distance"] .* sph_to_cart(evt_attrs["pos_theta"], evt_attrs["pos_phi"])

dir = sph_to_cart(evt_attrs["dir_theta"], evt_attrs["dir_phi"])

p = Particle(pos, dir, 0., evt_attrs["energy"], 0., PEPlus)


target = POM(SA[0., 0., 0.], 1)
medium = make_cascadia_medium_properties(0.95f0, evt_attrs["abs_scale"], evt_attrs["sca_scale"])
hits = propagate_particles([p], [target], 1, medium, hbc, hbg)

hits_per_pmt_mc = combine(groupby(hits, :pmt_id), :total_weight => sum)
hits_per_pmt_table = combine(groupby(DataFrame(d[:, :], [:time, :pmt_id, :total_weight]), :pmt_id), :total_weight => sum)

outerjoin(hits_per_pmt_table, hits_per_pmt_mc, on=:pmt_id, renamecols= "_table" => "_mc")
close(fid)
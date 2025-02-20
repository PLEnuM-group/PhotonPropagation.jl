
using PhotonSurrogateModel
using BSON
using Flux
using MLUtils
using StatsBase
using cuDNN
using PhotonPropagation
using LinearAlgebra
using StaticArrays
using Rotations
using Arrow
using DataFrames
using Glob
using TensorBoardLogger
using Logging
using Base.Iterators
using ParameterSchedulers
using PhysicsTools
using CairoMakie


target = POM(SA[0., 0., 0.], 1)

pmt_pos = get_pmt_positions(target, RotMatrix3(I))
ang_dists = dot.(permutedims(pmt_pos), permutedims(pmt_pos)')

files = glob("*.arrow", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_for_sr/results/em_shower/")
df = DataFrame(Arrow.Table(files))

no_gnn_model = SimpleMLPAmpModel(n_features=6, n_hidden=512, non_lin="relu", dropout=0.1)

all_data = []

for row in eachrow(df)
    #g = GNNGraph(ang_dists, ndata=(;nhits=log10.(row.hits.+1E-9)), gdata=(;x=[log10(row.dist), log10(row.energy), cos(pos_t), pos_phi, cos(row.dir_t), row.dir_phi]))
    #push!(all_graphs, g)

    x = create_model_input(no_gnn_model, row.pos, row.dir, row.energy, [0, 0, 0])
    push!(all_data, (nhits=log10.(row.hits.+1E-9), x=collect(x)))
end



device = gpu

train_data, test_data = MLUtils.splitobs(all_data, at=0.8, shuffle=true)
train_loader = DataLoader(train_data, batchsize=1024, shuffle=true, collate=true)
test_loader = DataLoader(test_data, batchsize=1024, shuffle=false, collate=true)

no_gnn_model = no_gnn_model |> device

opt_state = Flux.setup(Adam(5E-4), no_gnn_model)

sched = ParameterSchedulers.Stateful(Sequence(5E-4 => 150, 2E-4 => 50, 1E-4 => 50))

logger = TBLogger("/home/wecapstor3/capn/capn100h/tensorboard/no_gnn_surrogate", tb_increment)

for epoch in 1:250
    train_loss = 0
    Flux.adjust!(opt_state, ParameterSchedulers.next!(sched))
    Flux.trainmode!(no_gnn_model)
    for batch in train_loader
    
        batch = batch |> device
        
        tloss, grads = Flux.withgradient(no_gnn_model) do m
            Flux.mse(m(batch.x), batch.nhits)
        end

        Flux.update!(opt_state, no_gnn_model, grads[1])

        train_loss += cpu(tloss)
    end

    Flux.testmode!(no_gnn_model)
    test_loss = 0
    for batch in test_loader
        batch = batch |> gpu
        test_loss += cpu( Flux.mse(no_gnn_model(batch.x), batch.nhits))
    end

    train_loss /= length(train_loader)
    test_loss /= length(test_loader)

    with_logger(logger) do
        @info "train" loss=train_loss log_step_increment=0
        @info "test" loss=test_loss 
    end


    # Optionally, evaluate on a validation/test set
    # (similar procedure for batching and computing loss)
    @info "Epoch $epoch complete. Test loss: $test_loss"
end

BSON.bson("/home/wecapstor3/capn/capn100h/simple_mlp_amp_model.bson", model=cpu(no_gnn_model))


g = 0.95f0
pwf = 0.2f0
abs_scale = 1f0
sca_scale = 1f0
medium = CascadiaMediumProperties(g, pwf, abs_scale, sca_scale)
target = POM(SA_F32[0, 0, 0], UInt16(1))
wl_range = (300.0f0, 800.0f0)
spectrum = make_cherenkov_spectrum(wl_range, medium)

pos_theta = 1.2f0
pos_phi = 0.5f0

dist = 40f0


dir_theta = 0.7f0
dir_phi = 1.3f0
dir = sph_to_cart(dir_theta, dir_phi)
energy = 3E4



hbc, hbg = make_hit_buffers(Float32, 0.3);
pos_thetas = acos.(-1:0.1:1)
dir_phis = 0:0.2:2π
results = []


for dir_phi in dir_phis

    pos = dist .* sph_to_cart(Float32(pos_theta), pos_phi)
    dir = sph_to_cart(dir_theta, Float32(dir_phi))
    particle = Particle(
                pos,
                dir,
                0.0f0,
                Float32(energy),
                0.0f0,
                PEPlus
            )
    source = ExtendedCherenkovEmitter(particle, medium, spectrum)
    setup = PhotonPropSetup([source], [target], medium, spectrum, 1)


    photons = propagate_photons(setup, hbc, hbg)
    hits = make_hits_from_photons(photons, setup, RotMatrix3(I))
    calc_pe_weight!(hits, [target])

    

    ws = Weights(hits.total_weight)
    per_pmt_counts = counts(Int64.(hits.pmt_id), 1:16, ws)

    model_input = collect(create_model_input(no_gnn_model, particle, target))

    model_eval = cpu(no_gnn_model(model_input |> gpu))

    push!(results, (per_pmt_counts,model_eval))
end


res_pprop = [r[1][5] for r in results]
res_model = [r[2][5] for r in results]


fig = Figure()
ax = Axis(fig[1, 1])

lines!(ax, dir_phis, res_pprop)
lines!(ax, dir_phis, 10 .^ res_model .-1E-9)
fig


#7E-2, 5E-2, 1E-2
#"relu", "tanh", "sigmoid", "celu", "gelu", 
for (non_lin, lr, embed_dim, dropout) in product(["relu", "gelu"], [1E-4, 5E-4, 1E-3, 5E-3, 1E-2], [256, 512, 768, 1024], [0, 0.1, 0.1, 0.3, 0.5])
    
    hparams = Dict(
        "lr" => lr,
        "hidden_dim" => embed_dim,
        "non_lin" => non_lin,
        "dropout" => dropout
    )

    model = SimpleMLPAmpModel(n_features=6, n_hidden=hparams["hidden_dim"], non_lin=hparams["non_lin"], dropout=hparams["dropout"])
    model = model |> device
    opt_state = Flux.setup(Adam(hparams["lr"]), model)

    logger = TBLogger("/home/wecapstor3/capn/capn100h/tensorboard/no_gnn_surrogate", tb_increment)

    for epoch in 1:150
        train_loss = 0
        Flux.trainmode!(model)
        for batch in train_loader
        
            batch = batch |> device
            
            tloss, grads = Flux.withgradient(model) do m
                Flux.mse(m(batch.x), batch.nhits)
            end

            Flux.update!(opt_state, model, grads[1])

            train_loss += cpu(tloss)
        end
    
        Flux.testmode!(model)
        test_loss = 0
        for batch in test_loader
            batch = batch |> gpu
            test_loss += cpu(Flux.mse(model(batch.x), batch.nhits))
        end

        train_loss /= length(train_loader)
        test_loss /= length(test_loader)

        with_logger(logger) do
            @info "train" loss=train_loss log_step_increment=0
            @info "test" loss=test_loss 
        end


        # Optionally, evaluate on a validation/test set
        # (similar procedure for batching and computing loss)
        @info "Epoch $epoch complete. Test loss: $test_loss"
    end
    write_hparams!(logger, hparams, ["test/loss"])
end


train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.8)
train_loader = DataLoader(train_graphs, batchsize=256, shuffle=true, collate=true)
test_loader = DataLoader(test_graphs, batchsize=256, shuffle=false, collate=true)


for (lr, embed_dim) in product([1E-5, 5E-5, 1E-4, 5E-4, 1E-3, 5E-3], [32, 40, 56, 64, 80])
    hparams = Dict(
        "lr" => lr,
        "node_embedding_dim" => embed_dim,
    )


    model = JointModel(n_features=6, node_embedding_dim=hparams["node_embedding_dim"], gnn_conv_dim=8)
    model = model |> device
    opt_state = Flux.setup(Adam(hparams["lr"]), model)

    logger = TBLogger("/home/wecapstor3/capn/capn100h/tensorboard/gnn_surrogate", tb_increment)

    for epoch in 1:60
        train_loss = 0
        for batch in train_loader
        
            batch = batch |> device
            
            tloss, grads = Flux.withgradient(model) do m
                loss(m, batch)
            end

            Flux.update!(opt_state, model, grads[1])

            train_loss += cpu(tloss)
        end
    
        test_loss = 0
        for batch in test_loader
            batch = batch |> gpu
            test_loss += cpu(loss(model, batch))
        end

        train_loss /= length(train_loader)
        test_loss /= length(test_loader)

        with_logger(logger) do
            @info "train" loss=train_loss log_step_increment=0
            @info "test" loss=test_loss 
        end


        # Optionally, evaluate on a validation/test set
        # (similar procedure for batching and computing loss)
        @info "Epoch $epoch complete. Test loss: $test_loss"
    end
    write_hparams!(logger, hparams, ["test/loss"])
end


hparams = Dict(
        "lr" => 1E-3,
        "node_embedding_dim" => 64,
    )


model = GNNAmplitudeModel(n_features=6, node_embedding_dim=hparams["node_embedding_dim"])
model = model |> device
opt_state = Flux.setup(Adam(hparams["lr"]), model)
sched = ParameterSchedulers.Stateful(Sequence(1E-3 => 35, 5E-4 => 15, 1E-4 => 10))

logger = TBLogger("/home/wecapstor3/capn/capn100h/tensorboard/gnn_surrogate", tb_increment)

for epoch in 1:80
    train_loss = 0
    Flux.adjust!(opt_state, ParameterSchedulers.next!(sched))

    for batch in train_loader
    
        batch = batch |> device
        
        tloss, grads = Flux.withgradient(model) do m
            loss(m, batch)
        end

        Flux.update!(opt_state, model, grads[1])

        train_loss += cpu(tloss)
    end

    test_loss = 0
    for batch in test_loader
        batch = batch |> gpu
        test_loss += cpu(loss(model, batch))
    end

    train_loss /= length(train_loader)
    test_loss /= length(test_loader)

    with_logger(logger) do
        @info "train" loss=train_loss log_step_increment=0
        @info "test" loss=test_loss 
    end


    # Optionally, evaluate on a validation/test set
    # (similar procedure for batching and computing loss)
    @info "Epoch $epoch complete. Test loss: $test_loss"
end



model(all_graphs[650] |> gpu)
all_graphs[650].nhits




y_eval_rs = reshape(y_eval, (length(pos_cos_thetas), length(distances), ))
y_rs = reshape(y, (length(pos_cos_thetas), length(distances), ))

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, distances, y_rs[4, :])
lines!(ax, distances, y_eval_rs[4, :])
fig


fig = Figure()
ax = Axis(fig[1, 1])
pos = 10 .*  sph_to_cart(pos_theta, pos_phi)

dir2d = [dir[1], dir[2]]
dir2d ./= norm(dir2d) / 15

arrows!(ax, [pos[1]], [pos[2]], [dir2d[1]], [dir2d[2]])
arc!(Point2f(0), 0.3, -π, π)
fig
m = HealpixMap{Float64, RingOrder}(1)
dir_pix = pix2ang(m, 7)
arrows!(ax, [0], [0], [dir_pix[1]], [dir_pix[2]])
fig

fig = Figure()
ax = Axis(fig[1 ,1], yscale=log10, xscale=log10)
for i in 1:12
    pixel_counts = [map.pixels[i] for map in all_photons_maps]
    lines!(ax, distances, pixel_counts, label="$i")
end
axislegend(ax)
fig

rad2deg(dot(pmt_coords[6], pos ./norm(pos)))

30 .* sph_to_cart(pos_theta, pos_phi)


fig = Figure()
ax = Axis(fig[1 ,1], yscale=log10)
pmt_coords = get_pmt_positions(target, RotMatrix3(I))
for i in 1:16
    pmtc = pmt_coords[i]
    
    cvals = Float64[]
    for dist in distances
        pos = dist .* sph_to_cart(pos_theta, pos_phi)
        t_p_theta, t_d_theta, t_d_phi = transform_input(pmtc, pos, dir)
        push!(cvals, t_p_theta)
    end
    lines!(ax, distances, cvals)
end
fig



m.pixels


dist = 7
pos = dist .* sph_to_cart(pos_theta, pos_phi)

particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PHadronShower
        )
source = ExtendedCherenkovEmitter(particle, medium, spectrum)
setup = PhotonPropSetup(source, target, medium, spectrum, 1)
photons = propagate_photons(setup, hbc, hbg)


nside = 1
m = HealpixMap{Float64, RingOrder}(nside)
m.pixels[:] .= 0

for ph_dir in photons.direction
    ath, aph = vec2ang(ph_dir...)
    pix = ang2pix(m, ath, aph)
    if m.pixels[pix] == UNSEEN
        m.pixels[pix] = 1
    else
        m.pixels[pix] += 1
    end
end

image, mask, anymasked = mollweide(m)

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, image, colorscale=log10)
fig







all_hits = reduce(vcat, all_hits)

pmt_sel = 8
pmt_coords = get_pmt_positions(target, RotMatrix3(I))[pmt_sel]

unique(all_hits[:, :t_dir_phi])

combined = combine(groupby(all_hits[all_hits.pmt_id .== pmt_sel, :], :distance), :total_weight => sum)

lines(combined.distance, combined.total_weight_sum)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, combined.distance, combined.t_pos_theta)
lines!(ax, combined.distance, combined.t_dir_theta)
lines!(ax, combined.distance, combined.t_dir_phi)
fig


pos_sph = reduce(hcat, cart_to_sph.(photons.position ./ target.shape.radius))
fig = Figure()
ax = GeoAxis(fig[1,1])
sp = scatter!(ax, rad2deg.(pos_sph[2, :]), rad2deg.(pos_sph[1, :]) .-90, color=(:black, 0.2), markersize=1)



pos_sph = reduce(hcat, cart_to_sph.(hits.position ./ target.shape.radius))
sp = scatter!(ax, rad2deg.(pos_sph[2, :]), rad2deg.(pos_sph[1, :]) .-90, color=(:red, 0.4), markersize=4)
fig

unique(hits.pmt_id)
=#
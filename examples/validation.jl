using PhotonPropagation
using NeutrinoTelescopes
using StaticArrays
using PhysicsTools
using DataFrames
using Rotations
using LinearAlgebra
using CairoMakie
using LsqFit
using Formatting
using StatsBase
using Arrow
using JSON3
using Healpix
using BSON
using Flux
using Distributions
using JLD2

function proc_sim(photons, setup)
    hits = make_hits_from_photons(photons, setup, RotMatrix3(I))
    calc_time_residual!(hits, setup)
    n_photons = sum(photons[:, :total_weight])
    hits_per_pmt = combine(groupby(hits, :pmt_id), nrow => :nhits)

    pmt_max = nrow(hits) > 0 ? hits_per_pmt[argmax(hits_per_pmt[:, :nhits]), :pmt_id] : 0

    summary_stats = DataFrame(
        n_photons=n_photons,
        n_hits=nrow(hits),
        pmt_max=pmt_max,
    )

    pmt_ids = DataFrame(name=["pmt_$i" for i in 1:16], pmt_id=1:16)
    hits_per_pmt = leftjoin(pmt_ids, hits_per_pmt, on=:pmt_id)
    hits_per_pmt[!, :nhits]  .= coalesce.(hits_per_pmt[!, :nhits], 0)
    select!(hits_per_pmt, Not([:pmt_id]))
    summary_stats = hcat(summary_stats, permutedims(hits_per_pmt, :name))
    return summary_stats, hits
end


function scan_distance(target, medium, spectrum, source_f; n_samples=10, dir_theta=0.2, dir_phi=1.3, return_hits=false, args...)

    function _add_meta!(df, vars)
        for (key, val) in pairs(vars)
            df[!, key] .= val
        end
    end

    distances = 10:10:100
    stats = []
    all_hits = []
    for distance in distances
        position = SA_F32[0, 0, distance]
        direction = SVector{3, Float32}(sph_to_cart(dir_theta, dir_phi))
        source = source_f(position, direction)
        
        for seed in 1:n_samples
            setup = PhotonPropSetup(source, target, medium, spectrum, seed)
            photons = propagate_photons(setup)

            meta = Dict(:distance => distance, :seed => seed, :dir_theta => dir_theta, :dir_phi => dir_phi)

            summary_stats, hits = proc_sim(photons, setup)

            _add_meta!(summary_stats, meta)

            push!(stats, summary_stats)
            if return_hits && nrow(hits) > 0                
                _add_meta!(hits, meta)
                push!(all_hits, hits)
            end
        end
    end

    stats = reduce(vcat, stats)
    if return_hits
        all_hits = reduce(vcat, all_hits)
        return stats, all_hits
    end
    return stats
end


function scan_phi(target, medium, spectrum, source_f; n_samples=10, args...)
    phis = LinRange(0, 2*π, 20)
    stats = []
    distance = 20
    for phi in phis
        x = cos(phi)*distance
        y = sin(phi)*distance
        position = SA_F32[x, y, 0]
        direction = SA_F32[1, 0 , 0]
        source = source_f(position, direction)
        
        for seed in 1:n_samples
            setup = PhotonPropSetup(source, target, medium, spectrum, seed)
            photons = propagate_photons(setup)
    
            summary_stats, hits = proc_sim(photons, setup)

            summary_stats[!, :seed] .= seed
            summary_stats[!, :dir_phi] .= phi
            push!(stats, hits_per_pmt)
        end
    end
    return reduce(vcat, stats)
end



function plot_distance_scans(stats, target, att_len=27)
    # Start with isotropic emitter
    stats_mean = combine(groupby(stats, :distance), [:n_photons, :n_hits] .=> mean, [:n_photons, :n_hits] .=> sum, nrow )


    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Distance (m)", ylabel="Detected Photons", yscale=log10)

    ci_photons = reduce(hcat, poisson_confidence_interval.(stats_mean[:, :n_photons_sum]))' ./ stats_mean[:, :nrow]
    ci_hits = reduce(hcat, poisson_confidence_interval.(stats_mean[:, :n_hits_sum]))' ./ stats_mean[:, :nrow]

    colors = Makie.wong_colors()
    rangebars!(ax, stats_mean[:, :distance], ci_photons[:, 1] , ci_photons[:, 2] , color=colors[1], whiskerwidth=10  )
    rangebars!(ax, stats_mean[:, :distance], ci_hits[:, 1] , ci_hits[:, 2] , color=colors[2], whiskerwidth=10)

    scatter!(ax, stats_mean[:, :distance], stats_mean[:, :n_photons_mean], label="Sphere Hits", color=colors[1],)
    scatter!(ax, stats_mean[:, :distance], stats_mean[:, :n_hits_mean], label="PMT Hits", color=colors[2],)
    ylims!(ax, 1E-1, 2E5)
  
    #attenuation_length(wl, scattering_coeff=0) = 1 / (1/absorption_length(wl, medium) + scattering_coeff)
    photon_expec(dist, att_len=25., norm=1) = norm*1E9*target.shape.radius^2 / (4*dist^2) * exp(-dist / att_len)
    model(x, p) = photon_expec.(x, att_len, p[1])

    fit = curve_fit(model, stats_mean[:, :distance], stats_mean[:, :n_photons_mean], [1.], lower=[0.1])
    @show coef(fit) 
    lines!(ax, 10:1.:100, xs -> photon_expec(xs,att_len, coef(fit)[1]), label=format("Scaling: {:.2f} m", coef(fit)[1]))
    axislegend(ax)

    ax2 = Axis(fig[2, 1], ylabel="Ratio")

    rowsize!(fig.layout, 1, Auto(3))
    scatter!(ax2, stats_mean[:, :distance],  stats_mean[:, :n_hits_mean] ./  stats_mean[:, :n_photons_mean])
    linkxaxes!(ax, ax2)
    return fig
end

function plot_phi_scans(stats, target)
    stats_mean = combine(groupby(stats, [:dir_phi, :pmt_id]), :nhits => mean => :nhits)
    groups = groupby(stats_mean, :dir_phi)

    pmt_coords = get_pmt_positions(target, RotMatrix3(I))
    
    m = HealpixMap{Float64, RingOrder}(4)
    m[:] .= UNSEEN
    pix_id = map(x -> ang2pix(m, vec2ang(x...)...), pmt_coords)
    
    fig, ax, hm = plot(mollweide(m)[1])

    framerate = 5

    record(fig, "animate_pmt_hits_phi.mp4", groups; framerate = framerate) do group
        
        ixs0 = pix_id[1:16]
        m[ixs0] .= 0

        ixs = pix_id[Int.(group[:, :pmt_id])]

        m[ixs] .= group[:, :nhits]

        image, _, _ = mollweide(m)

        hm[3][] = image

    end
end

function poisson_confidence_interval(k, alpha=0.32)
    if k == 0
       low = 0
    else
        low = 0.5*quantile(Chisq(2*k), alpha/2)
    end
    high = 0.5*quantile(Chisq(2*k+2), 1-alpha/2)
    return [low, high]
end


function _plot_comparison!(stats, expected_amps, g)

    colors = Makie.wong_colors()

    gl = g[1, 1] = GridLayout(2, 1)
    rowsize!(gl, 1, Auto(3))

    ax = Axis(gl[1, 1], ylabel="Detected Photons", yscale=log10)
    #ci_photons = reduce(hcat, poisson_confidence_interval.(stats[:, :n_photons_sum]))' ./ stats[:, :nrow]
    ci_hits = reduce(hcat, poisson_confidence_interval.(stats[:, :n_hits_sum]))' ./ stats[:, :nrow]
    ci_surrogate = reduce(hcat, poisson_confidence_interval.(expected_amps))'

    #rangebars!(ax, stats[:, :distance], ci_photons[:, 1] , ci_photons[:, 2] , color=colors[1], whiskerwidth=10  )
    rangebars!(ax, stats[:, :distance], ci_hits[:, 1] , ci_hits[:, 2] , color=colors[1], whiskerwidth=10)
    rangebars!(ax, stats[:, :distance], ci_surrogate[:, 1] , ci_surrogate[:, 2] , color=colors[2], whiskerwidth=10)

    scatter!(ax, stats[:, :distance], stats[:, :n_hits_mean], label="PMT Hits", color=colors[1])
    scatter!(ax, stats[:, :distance], expected_amps, label="Surrogate Expectation", color=colors[2])
    #scatter!(ax, stats[:, :distance], stats[:, :n_photons_mean], label="Sphere Hits", color=colors[1])
    #scatter!(ax, stats[:, :distance], stats[:, :n_hits], color=(:red, 0.5))
    

    ylims!(ax, 1E-2, 3E4)
    axislegend(ax)

    ax2 = Axis(gl[2, 1], ylabel="Pull", xlabel="Distance (m)")
       
    n = nrow(stats)

    col = Makie.wong_colors()[3]

    band!(stats[:, :distance], -3 .*ones(n), 3 .*ones(n), color=(col, 0.2))
    band!(stats[:, :distance], -2 .*ones(n), 2 .*ones(n), color=(col, 0.2))    
    band!(stats[:, :distance], .-ones(n), ones(n), color=(col, 0.2))
    
    std_hits = vec(diff(ci_hits, dims=2) ./ 2)
    std_surrogate = vec(diff(ci_surrogate, dims=2) ./ 2)

    uncert = sqrt.(std_hits.^2 .+ std_surrogate.^2)

    pull = (stats[:, :n_hits_mean] .- expected_amps) ./ uncert
    
    scatter!(ax2, stats[:, :distance], pull, color=:black)
    linkxaxes!(ax, ax2)
    ylims!(ax2, (-3, 3))
    hlines!(ax2, [0.], linestyle=:dash, color=:black)

    return g
end

function plot_compare_distance_surrogate(stats; settings)
  

    model_path = joinpath(ENV["WORK"], "time_surrogate")
    BSON.@load joinpath(model_path, "extended/amplitude_2_FNL.bson") model tf_vec
    model = gpu(model)

    stats_mean = combine(groupby(stats, :distance), [:n_photons, :n_hits] .=> mean, [:n_photons, :n_hits] .=> sum, nrow)
    distances = stats_mean[:, :distance]

    direction = sph_to_cart(settings[:dir_theta], settings[:dir_phi])
    expected_amps = Vector{Vector{Float64}}(undef, 0)
    for distance in distances
        position = SA_F32[0, 0, distance]
        p = Particle(position, direction, 0f0, Float32(settings[:energy]), 0f0, PEMinus)
        log_expec_per_pmt, _, = get_log_amplitudes([p], [POM(SA_F64[0, 0, 0], 1)], model, tf_vec)

        push!(expected_amps, exp.(log_expec_per_pmt[:, 1, 1]))
    end

    expected_amps = reduce(hcat, expected_amps)

    fig = Figure()
    ga = fig[1, 1] = GridLayout(1, 1)
    _plot_comparison!(stats_mean, sum(expected_amps, dims=1)[:], ga)

    pmts = 1:16
    
    fig2 = Figure(resolution=(2000, 1500))
    ga = fig2[1, 1] = GridLayout(4, 4)

    
    for (i, pmt_ix) in enumerate(pmts)
        expamp = expected_amps[pmt_ix, :]

        row, col = divrem(i-1, 4)
        g = ga[row+1, col+1]
        stats_id = "pmt_$pmt_ix"
        stats_mean = combine(groupby(stats, :distance), stats_id => mean => :n_hits_mean ,  stats_id => sum => :n_hits_sum, nrow)
        _plot_comparison!(stats_mean, expamp, g)

        Label(g[1, 1, Top()], "PMT $pmt_ix", valign = :bottom,
            font = :bold,
            padding = (0, 0, 5, 0))


    end

    


    return fig, fig2
   
end


function run_scan(settings_dict)

    target = POM(SA_F32[0, 0, 0], 1)
    medium = make_cascadia_medium_properties(Float32(settings_dict[:g]))   

    if settings_dict[:source_type] == "isotropic"
        source_gen = (pos, _) -> PointlikeIsotropicEmitter(pos, 0f0, Int(settings_dict[:n_photons]))
        spectrum = Monochromatic(Float32(settings_dict[:wavelength]))
        
    elseif settings_dict[:source_type] == "cascade"
        wl_range = (300f0, 800f0)
        spectrum = CherenkovSpectrum(wl_range, medium)    
        source_gen = (pos, dir) -> ExtendedCherenkovEmitter(
            Particle(pos, dir, 0f0, Float32(settings_dict[:energy]), 0f0, PEMinus),
            medium, wl_range) 

    else
        error("unknown source type")
    end

    if settings_dict[:scan_type] == "distance"
        scan_f = scan_distance
    elseif settings_dict[:scan_type] == "phi"
        scan_f = scan_phi
    end

    return scan_f(target, medium, spectrum, source_gen; settings_dict...)

end


settings = Dict(
    :source_type=>"isotropic",
    :scan_type=>"phi",
    :g=>0.95,
    :n_samples=> 3,
    :n_photons => 1E9,
    :wavelength => 450,
)


stats = run_scan(settings)
plot_phi_scans(stats, POM(SA_F64[0, 0, 0], 1))
Arrow.write("validation_data_isotropic_phi_scan.arrow", stats, metadata=["settings" => JSON3.write(settings)])

settings = Dict(
    :source_type=>"isotropic",
    :scan_type=>"distance",
    :g=>0.95,
    :n_samples=> 3,
    :n_photons => 1E9,
    :wavelength => 450,
)

stats = run_scan(settings)

stats

plot_distance_scans(stats, POM(SA_F64[0, 0, 0], 1))
Arrow.write("validation_data_isotropic_distance_scan.arrow", stats, metadata=["settings" => JSON3.write(settings)])

settings = Dict(
    :source_type=>"cascade",
    :scan_type=>"phi",
    :g=>0.95,
    :n_samples=> 3,
    :energy => 3E4,
    :wavelength => 450,
    :dir_theta => 0.1,
    :dir_phi => 0.3
)

stats = run_scan(settings)
plot_phi_scans(stats, POM(SA_F64[0, 0, 0], 1))
Arrow.append("validation_data_cascade_phi_scan.arrow", stats,  metadata=["settings" => JSON3.write(settings)])

settings = Dict(
    :source_type=>"cascade",
    :scan_type=>"distance",
    :g=>0.95,
    :n_samples=> 5,
    :energy => 5E4,
    :wavelength => 450,
    :dir_theta => 0.1,
    :dir_phi => 0.3
)

stats = run_scan(settings)
plot_distance_scans(stats, POM(SA_F64[0, 0, 0], 1), 26.)
Arrow.write("validation_data_cascade_distance_scan.arrow", stats, metadata=["settings" => JSON3.write(settings)])
stats = Arrow.Table("validation_data_cascade_distance_scan.arrow")
settings = JSON3.read(Arrow.getmetadata(stats)["settings"], Dict{Symbol, Any})
stats = DataFrame(stats)

fig1, fig2 = plot_compare_distance_surrogate(stats, settings=settings)

fig2


settings = Dict(
    :source_type=>"cascade",
    :scan_type=>"distance",
    :g=>0.95,
    :n_samples=> 5,
    :energy => 5E4,
    :wavelength => 450,
    :dir_theta => 0.1,
    :dir_phi => 0.3,
    :return_hits => true,
)

stats, hits = run_scan(settings)

jldopen("validation.jld2", "w") do file
    file["distance/cascade/1/stats"] = stats
    file["distance/cascade/1/hits"] = hits
    file["distance/cascade/1/settings"] = settings
end


model_path = joinpath(ENV["WORK"], "time_surrogate")
model = PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_1_FNL.bson"))
model = gpu(model)

source_gen = (pos, dir) -> ExtendedCherenkovEmitter(
    Particle(pos, dir, 0f0, Float32(settings_dict[:energy]), 0f0, PEMinus),
    medium, wl_range) 

distance = 10
h = hits[hits[:, :distance] .== distance, :]

position = SA_F32[0, 0, distance]
direction = SVector{3, Float32}(sph_to_cart(settings[:dir_theta], settings[:dir_phi]))

p = Particle(position, direction, 0f0, Float32(settings[:energy]), 0f0, PEMinus)
target = POM(SA_F32[0, 0, 0], 1)
medium = make_cascadia_medium_properties(settings[:g])

compare_mc_model([p], [target], Dict("model1" => model), medium, hits; oversampling=settings[:n_samples])

flow_params = calc_flow_input(p, target, model_time[:tf_vec])

c_n = c_at_wl(800.0f0, medium)
tgeo = calc_tgeo(p, target, c_n)

times = -10:1.:400

pmt_ix = 10

llh = model_time[:model](times, repeat(flow_params[:, pmt_ix],1,  length(times)))

plot(times, llh)

llh = eval_transformed_normal_logpdf(
    times .- tgeo,
    repeat(flow_params[:, pmt_ix],1,  length(times), ),
    model.time_model.range_min,
    model.time_model.range_max)

plot(times, llh)




dir_thetas = Float32.(0:0.3:π)
dir_phi = 0f0
stats = []
n_samples = 1
for dir_theta in dir_thetas
    direction = sph_to_cart(dir_theta, dir_phi)
    position = SA_F32[0, 0, 30]
    p = Particle(position, direction, 0f0, energy, 0f0, PEMinus)
    source = ExtendedCherenkovEmitter(p, medium, wl_range)

    for seed in 1:n_samples
        setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
        photons = propagate_photons(setup)
        n_photons, n_hits, pmt_max = proc_sim(photons, setup)
        push!(stats, (n_photons=n_photons, n_hits=n_hits, pmt_max=pmt_max, dir_theta=dir_theta, seed=seed))
    end
end

stats = DataFrame(stats)

scatter(stats[:, :dir_phi], stats[:, :n_hits])






#=
fnames_casc = [joinpath(ENV["WORK"], "photon_tables/extended/hits/photon_table_extended_0_hits.hd5")]
nsel_frac = 0.9
rng = MersenneTwister(31338)

hits, features, tf_vec = read_pmt_hits(fnames_casc, nsel_frac, rng)

hits


model, loss_f = setup_model(hparams)

using MLUtils

data = (tres=Float32.(hits), label=Float32.(features))

train_data, val_data = splitobs(data, at=0.7)
train_loader, test_loader = setup_dataloaders(train_data, val_data, hparams)
opt = setup_optimizer(hparams, length(train_loader))
device=gpu
model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = train_model!(
            optimizer=opt,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss_function=loss_f,
            hparams=hparams,
            logger=nothing,
            device=device,
            use_early_stopping=true,
            checkpoint_path=nothing)

fid = h5open(fnames_casc[1])
g = fid["pmt_hits/dataset_300"]

data = DataFrame(g[:, :], [:time, :pmt_ix])


grp_attrs = attrs(g)

using CairoMakie


times_eval = -10:1:50


feature_matrix = zeros(Float64, 8, length(times_eval))

feature_matrix[1, :] .= log.(grp_attrs["distance"])
feature_matrix[2, :] .= log.(grp_attrs["energy"])

feature_matrix[3:5, :] .= permutedims(reduce(hcat, sph_to_cart.(grp_attrs["dir_theta"], grp_attrs["dir_phi"])))
feature_matrix[6:8, :] .= permutedims(reduce(hcat, sph_to_cart.(grp_attrs["pos_theta"], grp_attrs["pos_phi"])))
ExtendedCascadeModel.preproc_labels!(feature_matrix, feature_matrix, tf_vec)

pmt_ix = 5
feature_matrix = ExtendedCascadeModel.append_onehot_pmt(feature_matrix, ones(length(times_eval)).*pmt_ix)

llh = model(gpu(times_eval), gpu(feature_matrix))


sel = data[:, :pmt_ix] .== pmt_ix
fig, ax = hist(data[sel, :time], normalization=:pdf, bins=-10:5:50)


lines!(ax, times_eval, exp.(cpu(llh)))
fig

target = POM(SA_F32[0, 0, 0], 1)

pos_cart = grp_attrs["distance"] .* sph_to_cart.(grp_attrs["pos_theta"], grp_attrs["pos_phi"])
dir_cart = sph_to_cart.(grp_attrs["dir_theta"], grp_attrs["dir_phi"])
p = Particle(pos_cart, dir_cart, 0f0, Float32(grp_attrs["energy"]), 0f0, PEMinus)

calc_flow_input(p, target, tf_vec)[:, pmt_ix] .≈ feature_matrix[:, 1]

calc_flow_input(p, target, tf_vec)[:, pmt_ix] 
feature_matrix[:, 1]
=#

# Save output


#=
hits = make_hits_from_photons(photons, setup)
event_record = Dict(:hits=>hits, :sources=>[source], :event_id=>uuid4())

using HDF5

fid = h5open("test.hd5", "w")

fid["test"] = hits[:, [:time]]


save_event("test.hd5", event_record)
=#
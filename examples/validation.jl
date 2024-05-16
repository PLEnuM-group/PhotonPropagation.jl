using PhotonPropagation
using NeutrinoTelescopes
using PhotonSurrogateModel
using NeutrinoSurrogateModelData
using StaticArrays
using PhysicsTools
using DataFrames
using Rotations
using LinearAlgebra
using CairoMakie
using LsqFit
using Format
using StatsBase
using Arrow
using JSON3
using Healpix
using BSON
using Flux
using Distributions
using JLD2
using HDF5
using DataStructures

function proc_sim(photons, setup)
    hits = make_hits_from_photons(photons, setup, RotMatrix3(I))
    calc_time_residual!(hits, setup)
    calc_pe_weight!(hits, setup)
    n_photons = sum(photons[:, :total_weight])
    n_hits = sum(hits[:, :total_weight])

    hits_per_pmt = combine(groupby(hits, :pmt_id), :total_weight => sum => :nhits)

    pmt_max = nrow(hits) > 0 ? hits_per_pmt[argmax(hits_per_pmt[:, :nhits]), :pmt_id] : 0

    summary_stats = DataFrame(
        n_photons=n_photons,
        n_hits=n_hits,
        pmt_max=pmt_max,
    )

    pmt_ids = DataFrame(name=["pmt_$i" for i in 1:16], pmt_id=1:16)
    hits_per_pmt = leftjoin(pmt_ids, hits_per_pmt, on=:pmt_id)
    hits_per_pmt[!, :nhits]  .= coalesce.(hits_per_pmt[!, :nhits], 0)
    select!(hits_per_pmt, Not([:pmt_id]))
    summary_stats = hcat(summary_stats, permutedims(hits_per_pmt, :name))
    return summary_stats, hits
end

function _add_meta!(df, vars)
    for (key, val) in pairs(vars)
        df[!, key] .= val
    end
end


function scan_distance(
    target,
    medium,
    spectrum,
    source_f;
    n_samples=10,
    dir_theta=0.2,
    dir_phi=1.3,
    return_hits=false,
    args...)

    distances = 10:10:110
    stats = []
    all_hits = []
    direction = SVector{3, Float32}(sph_to_cart(dir_theta, dir_phi))

    hbc, hbg = make_hit_buffers()

    for distance in distances

        r = direction[2] > 0 ? direction[1] / direction[2] : zero(direction[1])
        position = SA_F32[distance/sqrt(1 + r^2), -r*distance/sqrt(1 + r^2), 0]
        source = source_f(position, direction)
        
        photons = nothing
        for seed in 1:n_samples

            setup = PhotonPropSetup(source, target, medium, spectrum, seed)
            base_weight = 1.
            while true
                prop_source = setup.sources[1]
                if prop_source.photons > 1E13
                    println("More than 1E13 photons, skipping")
                    return nothing
                end
                photons = propagate_photons(setup, hbc, hbg)
        
                if nrow(photons) > 100
                    break
                end
        
                setup.sources[1] = oversample_source(prop_source, 100)
                println(format("distance {:.2f} photons: {:d}", distance, setup.sources[1].photons))
                base_weight /= 100
        
            end

            photons[!, :total_weight] .*= base_weight

            meta = Dict(:distance => distance, :seed => seed, :dir_theta => dir_theta, :dir_phi => dir_phi, :pos_x => position[1],
                       :pos_y => position[2], :pos_z => position[3])

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

function prop_single(
    target,
    medium,
    spectrum,
    source_f;
    n_samples=10,
    dir_theta=0.2,
    dir_phi=1.3,
    pos_x = 6.,
    pos_y = 10.,
    pos_z = -4.,
    return_hits=false,
    args...)

    stats = []
    all_hits = []
    direction = SVector{3, Float32}(sph_to_cart(dir_theta, dir_phi))
    position = SA_F32[pos_x, pos_y, pos_z]
    source = source_f(position, direction)
    
    hbc, hbg = make_hit_buffers()

    photons = nothing
    for seed in 1:n_samples

        setup = PhotonPropSetup(source, target, medium, spectrum, seed)
        base_weight = 1.
        while true
            prop_source = setup.sources[1]
            if prop_source.photons > 1E13
                println("More than 1E13 photons, skipping")
                return nothing
            end
            photons = propagate_photons(setup, hbc, hbg)
    
            if nrow(photons) > 100
                break
            end
    
            setup.sources[1] = oversample_source(prop_source, 100)
            println(format("distance {:.2f} photons: {:d}", distance, setup.sources[1].photons))
            base_weight /= 100
    
        end

        photons[!, :total_weight] .*= base_weight

        meta = Dict(:seed => seed, :dir_theta => dir_theta, :dir_phi => dir_phi, :pos_x => position[1],
                    :pos_y => position[2], :pos_z => position[3])

        summary_stats, hits = proc_sim(photons, setup)

        _add_meta!(summary_stats, meta)

        push!(stats, summary_stats)
        if return_hits && nrow(hits) > 0                
            _add_meta!(hits, meta)
            push!(all_hits, hits)
        end
    end


    stats = reduce(vcat, stats)
    if return_hits
        all_hits = reduce(vcat, all_hits)
        return stats, all_hits
    end
    return stats
end


function scan_phi(target, medium, spectrum, source_f; n_samples=10, distance=20, return_hits=false, args...)
    phis = LinRange(0, 2*π, 20)
    stats = []
    all_hits = []
    hbc, hbg = make_hit_buffers()
    for phi in phis
        x = cos(phi)*distance
        y = sin(phi)*distance
        position = SA_F32[x, y, 0]
        direction = SA_F32[1, 0 , 0]
        source = source_f(position, direction)
        
        for seed in 1:n_samples
            setup = PhotonPropSetup(source, target, medium, spectrum, seed)
            photons = propagate_photons(setup, hbc, hbg)

            meta = Dict(:distance => distance, :seed => seed, :dir_phi => phi)
    
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


function run_scan(settings_dict)

    target = POM(SA_F32[0, 0, 0], 1)

    medium = make_cascadia_medium_properties(Float32(settings_dict[:g]), Float32(settings_dict[:abs_scale]), Float32(settings_dict[:sca_scale]))   

    if settings_dict[:source_type] == "isotropic"
        source_gen = (pos, _) -> PointlikeIsotropicEmitter(pos, 0f0, Int(settings_dict[:n_photons]))
        spectrum = make_monochromatic_spectrum(Float32(settings_dict[:wavelength]))
        
    elseif settings_dict[:source_type] == "cascade"
        wl_range = (300f0, 800f0)
        spectrum = make_cherenkov_spectrum(wl_range, medium)    
        source_gen = (pos, dir) -> ExtendedCherenkovEmitter(
            Particle(pos, dir, 0f0, Float32(settings_dict[:energy]), 0f0, PEMinus),
            medium, spectrum) 

    elseif settings_dict[:source_type] == "track"
        wl_range = (300f0, 800f0)
        spectrum = make_cherenkov_spectrum(wl_range, medium)    
        
  
        function _gen(pos, dir)
            ppos = pos .- (settings_dict[:length] / 2) .* dir
            particle = Particle(Float32.(ppos), dir, 0f0, Float32(settings_dict[:energy]), Float32(settings_dict[:length]), PMuMinus)
            return FastLightsabreMuonEmitter(particle, medium, spectrum)
        end

        source_gen = _gen
  
    else
        error("unknown source type")
    end

    if settings_dict[:scan_type] == "distance"
        scan_f = scan_distance
    elseif settings_dict[:scan_type] == "phi"
        scan_f = scan_phi
    elseif settings_dict[:scan_type] == "single" || settings_dict[:scan_type] == "single_timeuncert"
        scan_f = prop_single
    end

    return scan_f(target, medium, spectrum, source_gen; settings_dict...)

end



function plot_distance_scans(stats, target, att_len=27; settings )
    # Start with isotropic emitter
    stats_mean = combine(groupby(stats, :distance), [:n_photons, :n_hits] .=> mean, [:n_photons, :n_hits] .=> sum, nrow )


    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Distance (m)", ylabel="Detected Photons", yscale=Makie.pseudolog10, )

    ci_photons = reduce(hcat, poisson_confidence_interval.(stats_mean[:, :n_photons_sum]))' ./ stats_mean[:, :nrow]
    ci_hits = reduce(hcat, poisson_confidence_interval.(stats_mean[:, :n_hits_sum]))' ./ stats_mean[:, :nrow]

    colors = Makie.wong_colors()
    rangebars!(ax, stats_mean[:, :distance], ci_photons[:, 1] , ci_photons[:, 2] , color=colors[1], whiskerwidth=10  )
    rangebars!(ax, stats_mean[:, :distance], ci_hits[:, 1] , ci_hits[:, 2] , color=colors[2], whiskerwidth=10)

    scatter!(ax, stats_mean[:, :distance], stats_mean[:, :n_photons_mean], label="Sphere Hits", color=colors[1],)
    scatter!(ax, stats_mean[:, :distance], stats_mean[:, :n_hits_mean], label="PMT Hits", color=colors[2],)
  
    #attenuation_length(wl, scattering_coeff=0) = 1 / (1/absorption_length(wl, medium) + scattering_coeff)
    

    if settings[:source_type] == "cascade" || settings[:source_type] == "isotropic"
        photon_expec = (dist, att_len=25., norm=1) -> norm*1E9*target.shape.radius^2 / (4*dist^2) * exp(-dist / att_len)
    else        
        photon_expec = (dist, att_len=25., norm=1) -> norm*1E9/settings[:length]*target.shape.radius^2 / (2*dist) * exp(-dist / att_len)
    end
    model(x, p) = photon_expec.(x, att_len, p[1])
    mn, mx = extrema(stats_mean[:, :distance])


    fit = curve_fit(model, stats_mean[:, :distance], stats_mean[:, :n_photons_mean], [1.], lower=[0.001])
    lines!(ax, mn:1.:mx, xs -> photon_expec(xs,att_len, coef(fit)[1]), label=format("Scaling: {:.2f}", coef(fit)[1]))
    axislegend(ax)

    ax2 = Axis(fig[2, 1], ylabel="Ratio")

    rowsize!(fig.layout, 1, Auto(3))
    scatter!(ax2, stats_mean[:, :distance],  stats_mean[:, :n_hits_mean] ./  stats_mean[:, :n_photons_mean])
    linkxaxes!(ax, ax2)
    return fig
end

function plot_phi_scans(stats, target, filename)

    pmt_cols = ["pmt_$i" for i in 1:16]
    pmt_cols_mean = ["pmt_$i"*"_mean" for i in 1:16]

    stats_mean = combine(groupby(stats, :dir_phi), pmt_cols .=> mean)
    groups = groupby(stats_mean, :dir_phi)

    pmt_coords = get_pmt_positions(target, RotMatrix3(I))
    
    m = HealpixMap{Float64, RingOrder}(4)
    m[:] .= UNSEEN
    pix_id = map(x -> ang2pix(m, vec2ang(x...)...), pmt_coords)
    
    fig, ax, hm = plot(mollweide(m)[1])

    framerate = 5

    CairoMakie.record(fig, filename, groups; framerate = framerate) do group
        
        #=
        ixs0 = pix_id[1:16]
        m[ixs0] .= 0

        ixs = pix_id[Int.(group[:, :pmt_id])]

        m[ixs] .= group[:, :nhits]

        =#

        ixs = pix_id[1:16]

        nhits = Matrix(group[:, pmt_cols_mean])[:]

        m[ixs] .= nhits

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


function _plot_comparison!(stats, amp_dict, g)

    colors = Makie.wong_colors()

    gl = g[1, 1] = GridLayout(2, 1)
    rowsize!(gl, 1, Auto(3))


    ax = Axis(gl[1, 1], ylabel="Detected Photons", yscale=Makie.pseudolog10, yminorticksvisible=true, yminorticks=IntervalsBetween(10))
    ax2 = Axis(gl[2, 1], ylabel="Pull", xlabel="Distance (m)")

    #ci_photons = reduce(hcat, poisson_confidence_interval.(stats[:, :n_photons_sum]))' ./ stats[:, :nrow]
    ci_hits = reduce(hcat, poisson_confidence_interval.(stats[:, :n_hits_sum]))' ./ stats[:, :nrow]
   
    std_hits = vec(diff(ci_hits, dims=2) ./ 2)

    #rangebars!(ax, stats[:, :distance], ci_photons[:, 1] , ci_photons[:, 2] , color=colors[1], whiskerwidth=10  )
    rangebars!(ax, stats[:, :distance], ci_hits[:, 1] , ci_hits[:, 2] , color=:black, whiskerwidth=10)
    scatter!(ax, stats[:, :distance], stats[:, :n_hits_mean], label="MC Photon Propagation", color=:black)

    for (i, (mname)) in enumerate(sort(collect(keys(amp_dict))))
        expected_amps = amp_dict[mname]
        ci_surrogate = reduce(hcat, poisson_confidence_interval.(expected_amps))'    
        rangebars!(ax, stats[:, :distance], ci_surrogate[:, 1] , ci_surrogate[:, 2] , color=colors[i], whiskerwidth=10)
        scatter!(ax, stats[:, :distance], expected_amps, label="$mname", color=colors[i])

        std_surrogate = vec(diff(ci_surrogate, dims=2) ./ 2)
        uncert = sqrt.(std_hits.^2 .+ std_surrogate.^2)
        pull = (stats[:, :n_hits_mean] .- expected_amps) ./ uncert        
        scatter!(ax2, stats[:, :distance], pull, color=colors[i])

    end
    n = nrow(stats)

    col = Makie.wong_colors()[3]

    band!(stats[:, :distance], -3 .*ones(n), 3 .*ones(n), color=(col, 0.2))
    band!(stats[:, :distance], -2 .*ones(n), 2 .*ones(n), color=(col, 0.2))    
    band!(stats[:, :distance], .-ones(n), ones(n), color=(col, 0.2))
    
    

    linkxaxes!(ax, ax2)
    ylims!(ax2, (-3, 3))
    hlines!(ax2, [0.], linestyle=:dash, color=:black)

    return g
end

function plot_compare_distance_surrogate(stats, models; settings)
  
    stats_mean = combine(groupby(stats, :distance), [:n_photons, :n_hits] .=> mean, [:n_photons, :n_hits] .=> sum, nrow)
    distances = stats_mean[:, :distance]

    direction = sph_to_cart(settings[:dir_theta], settings[:dir_phi])
    amp_dict = Dict()



    for (mname) in sort(collect(keys(models)))
        model = models[mname]
        expected_amps = Vector{Vector{Float64}}(undef, 0)
        feat_buffer = create_input_buffer(model, 16, 1)
        for distance in distances
            #position = SA_F64[stats[:pos_x], stats[:pos_y], stats[:pos_z]]

            r = direction[2] > 0 ? direction[1] / direction[2] : zero(direction[1])
            position = SA_F64[distance/sqrt(1 + r^2), -r*distance/sqrt(1 + r^2), 0]

            ptype = settings[:source_type] == "cascade" ? PEMinus : PMuMinus
            length = settings[:source_type] == "cascade" ? 0. : Float64(settings[:length])

            p = Particle(position, direction, 0., settings[:energy], length, ptype)
            log_expec_per_pmt, _, = get_log_amplitudes([p], [POM(SA_F64[0, 0, 0], 1)], gpu(model), feat_buffer=feat_buffer, abs_scale=settings[:abs_scale], sca_scale=settings[:sca_scale])

            push!(expected_amps, cpu(exp.(log_expec_per_pmt[:, 1, 1])))
        end

        expected_amps = reduce(hcat, expected_amps)

        amp_dict[mname] = expected_amps
    end
    
    amp_dict_sum = Dict(key => sum(val, dims=1)[:] for (key, val) in amp_dict)
    

    fig = Figure()
    ga = fig[1, 1] = GridLayout(1, 1)
    _plot_comparison!(stats_mean, amp_dict_sum, ga)

    pmts = 1:16
    
    fig2 = Figure(resolution=(2000, 1500))
    ga = fig2[1, 1] = GridLayout(4, 4)

    
    for (i, pmt_ix) in enumerate(pmts)
        amp_dict_per_pmt = Dict(key => val[pmt_ix, :] for (key, val) in amp_dict)
       

        row, col = divrem(i-1, 4)
        g = ga[row+1, col+1]
        stats_id = "pmt_$pmt_ix"
        stats_mean = combine(groupby(stats, :distance), stats_id => mean => :n_hits_mean ,  stats_id => sum => :n_hits_sum, nrow)
        _plot_comparison!(stats_mean, amp_dict_per_pmt, g)

        Label(g[1, 1, Top()], "PMT $pmt_ix", valign = :bottom,
            font = :bold,
            padding = (0, 0, 5, 0))


    end

    fig2[1, 2] = Legend(fig2, content(fig2[1, 1][1, 1][1, 1][1, 1]))

    return fig, fig2
   
end

function plot_compare_time_dist(hits, models; settings)

    distance = sort(unique(hits[:, :distance]))[2]
    h = hits[hits[:, :distance] .== distance, :]

    direction = SVector{3, Float64}(sph_to_cart(settings[:dir_theta], settings[:dir_phi]))
    r = direction[2] > 0 ? direction[1] / direction[2] : zero(direction[1])
    position = SA_F64[distance/sqrt(1 + r^2), -r*distance/sqrt(1 + r^2), 0]
    ptype = settings[:source_type] == "cascade" ? PEMinus : PMuMinus
    length = settings[:source_type] == "cascade" ? 0. : Float64(settings[:length])

    
    p = Particle(position, direction, 0., settings[:energy], length, ptype)
    target = POM(SA_F32[0, 0, 0], 1)
    medium = make_cascadia_medium_properties(settings[:g])

    fig, fig2 = compare_mc_model([p], [target], models, medium, h; oversampling=settings[:n_samples], bin_width=0.5, abs_scale=settings[:abs_scale], sca_scale=settings[:sca_scale])
    return fig, fig2
end


function plot_compare_time_dist_single(hits, models; settings)

    direction = SVector{3, Float64}(sph_to_cart(settings[:dir_theta], settings[:dir_phi]))
    position = SA_F64[settings[:pos_x], settings[:pos_y], settings[:pos_z]]
    ptype = settings[:source_type] == "cascade" ? PEMinus : PMuMinus
    length = settings[:source_type] == "cascade" ? 0. : Float64(settings[:length])

    
    p = Particle(position, direction, 0., settings[:energy], length, ptype)
    target = POM(SA_F32[0, 0, 0], 1)
    medium = make_cascadia_medium_properties(settings[:g])

    fig, fig2 = compare_mc_model([p], [target], models, medium, hits; oversampling=settings[:n_samples], bin_width=0.5, abs_scale=settings[:abs_scale], sca_scale=settings[:sca_scale])
    return fig, fig2
end



configs = Dict(
    #="iso_phi" => Dict(
        :source_type=>"isotropic",
        :scan_type=>"phi",
        :g=>0.95,
        :n_samples=> 3,
        :n_photons => 1E9,
        :wavelength => 450,
        :return_hits => true,
        :distance => 20,  
    ),
    "iso_dist" => Dict(
        :source_type=>"isotropic",
        :scan_type=>"distance",
        :g=>0.95,
        :n_samples=> 3,
        :n_photons => 1E9,
        :wavelength => 450.,
        :return_hits => true,
        :pos_theta => 0.2,
        :pos_phi => 1.3
    ),
    =#
    "casc_phi" => Dict(
        :source_type=>"cascade",
        :scan_type=>"phi",
        :g=>0.95,
        :n_samples=> 3,
        :energy => 5E4,
        :wavelength => 450,
        :dir_theta => 1.5,
        :dir_phi => 0.3,
        :return_hits => true,
        :abs_scale => 0.95,
        :sca_scale => 1.05,
    ),

    "casc_dist" => Dict(
        :source_type=>"cascade",
        :scan_type=>"distance",
        :g=>0.95,
        :n_samples=> 3,
        :energy => 5E4,
        :dir_theta => 1.5,
        :dir_phi => 0.3,
        :pos_theta => 3.2,
        :pos_phi => 4.3,
        :return_hits => true,
        :abs_scale => 0.95,
        :sca_scale => 1.05,
    ),
     "track_dist" => Dict(
        :source_type=>"track",
        :scan_type=>"distance",
        :g=>0.95,
        :n_samples=> 5,
        :energy => 6E4,
        :length => 400,
        :dir_theta => 1.3,
        :dir_phi => 5.5,
        :return_hits => true,
        :abs_scale => 0.95,
        :sca_scale => 1.05,
    ),
    "track_single" => Dict(
        :source_type=>"track",
        :scan_type=>"single",
        :g=>0.95,
        :n_samples=> 5,
        :energy => 6E5,
        :length => 10000,
        :dir_theta => 0.3,
        :dir_phi => 2.5,
        :pos_x => 5,
        :pos_y => 6,
        :pos_z => 10,
        :return_hits => true,
        :abs_scale => 0.97,
        :sca_scale => 1.05,
    ),
    "track_single_timing_uncert" => Dict(
        :source_type=>"track",
        :scan_type=>"single_timeuncert",
        :g=>0.95,
        :n_samples=> 5,
        :energy => 6E4,
        :length => 400,
        :dir_theta => 0.3,
        :dir_phi => 2.5,
        :pos_x => 5,
        :pos_y => 6,
        :pos_z => 10,
        :return_hits => true,
        :abs_scale => 0.95,
        :sca_scale => 1.05,
    ),
    #=
    "casc_single_abs" => Dict(
        :source_type=>"casc",
        :scan_type=>"single",
        :g=>0.95,
        :n_samples=> 20,
        :energy => 6E4,
        :length => 0.,
        :dir_theta => 0.3,
        :dir_phi => 0.5,
        :pos_x => 5,
        :pos_y => 6,
        :pos_z => 10,
        :return_hits => true,
        :abs_scale => 0.9:0.1:1.1,
        :sca_scale => 1.        
    ),
    =#
)

jldopen("validation.jld2", "w") do file
end

for (key, conf) in configs
    println("Running $key")
    stats, hits = run_scan(conf)
    jldopen("validation.jld2", "a") do file
        file["$key/stats"] = stats
        file["$key/hits"] = hits
        file["$key/settings"] = conf
    end
end

PROJECT_ROOT = pkgdir(PhotonPropagation)
figure_dir = joinpath(PROJECT_ROOT, "figures")

model_path = joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate_perturb")
models_casc = Dict(
    "A1S1" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_uncert_0_2_FNL.bson")),
    "A2S1" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_2_FNL.bson"), joinpath(model_path, "extended/time_uncert_0_2_FNL.bson")),
    "A1S3" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_3_FNL.bson"), joinpath(model_path, "extended/time_uncert_0_3_FNL.bson")),

)

models_track = Dict(
    "Model A" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_0_1_FNL.bson")),
    #"A2S1" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_2_FNL.bson"), joinpath(model_path, "lightsabre/time_1_FNL.bson")),
    #"A1S2" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_2_FNL.bson")),
    "Model B" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_2_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_0_2_FNL.bson")),
    #"A3S1" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_3_FNL.bson"), joinpath(model_path, "lightsabre/time_1_FNL.bson")),
    "Model C" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_0_3_FNL.bson")),
)

models_track_2 = Dict(
    "Model A (2ns)" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_2_1_FNL.bson")),
    "Model B (2ns)" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_2_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_2_1_FNL.bson")),
    "Model C (2ns)" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_3_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_2_1_FNL.bson")),
)

models_track_3 = Dict(
    "Model A (3ns)" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_3_1_FNL.bson")),
    "Model B (3ns)" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_2_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_3_2_FNL.bson")),
    "Model C (3ns)" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_3_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_3_3_FNL.bson")),
)

all_models_track = Dict(0 => models_track, 2 => models_track_2) #2.5 => models_track_25)

extension = "png"


set_theme!()

fid = jldopen(joinpath(PROJECT_ROOT, "examples/validation.jld2"), "r")
fig1, fig2 = plot_compare_distance_surrogate(fid["casc_dist"]["stats"], models_casc, settings=fid["casc_dist"]["settings"])
fig2


jldopen(joinpath(PROJECT_ROOT, "examples/validation.jld2"), "r") do file
    #=
    fontsize_theme = Theme(
        fontsize = 30, linewidth=3,
        Axis=(xlabelsize=35, ylabelsize=35))
    set_theme!(fontsize_theme)
    =#
    @show keys(file)
    for key in keys(file)
        stats = file[key]["stats"]
        hits = file[key]["hits"]
        settings = file[key]["settings"]

        if settings[:scan_type] == "phi"
            fig = plot_phi_scans(stats, POM(SA_F64[0, 0, 0], 1), joinpath(figure_dir, "$(key)_scan.mp4"))
        elseif settings[:scan_type] == "distance"
            medium = make_cascadia_medium_properties(settings[:g], settings[:abs_scale], settings[:sca_scale])
            wl = haskey(settings, :wavelength) ? settings[:wavelength] : 450.

            fig = plot_distance_scans(stats, POM(SA_F64[0, 0, 0], 1), absorption_length(wl, medium), settings=settings)
            save(joinpath(figure_dir, "$(key)_scan.$(extension)"), fig)
        end
       
        if settings[:scan_type] == "distance" && settings[:source_type] != "isotropic"
            
            models = settings[:source_type] == "track" ? models_track : models_casc
           
            fig1, fig2 = plot_compare_distance_surrogate(stats, models, settings=settings)
            fig3, fig4 = plot_compare_time_dist(hits, models, settings=settings)

            save(joinpath(figure_dir, "$(key)_summed_comp_scan.$(extension)"), fig1)
            save(joinpath(figure_dir, "$(key)_per_pmt_comp_scan.$(extension)"), fig2)
            save(joinpath(figure_dir, "$(key)_per_pmt_comp_time.$(extension)"), fig3)        
            save(joinpath(figure_dir, "$(key)_per_pmt_max_comp_time.$(extension)"), fig4)        
        end

        if settings[:scan_type] == "single"
            models = settings[:source_type] == "track" ? models_track : models_casc
           
            fig1, fig2 = plot_compare_time_dist_single(hits, models, settings=settings)
            save(joinpath(figure_dir, "$(key)_per_pmt_comp_time.$(extension)"), fig1)
            save(joinpath(figure_dir, "$(key)_per_pmt_max_comp_time.$(extension)"), fig2)
        elseif settings[:scan_type] == "single_timeuncert"
            all_models = settings[:source_type] == "track" ? all_models_track : all_models_casc

            for (uncert, models) in all_models
                hits_smeared = copy(hits)
                hits_smeared[:, :tres] .+= randn(nrow(hits_smeared)) * uncert
                fig1, fig2 = plot_compare_time_dist_single(hits_smeared, models, settings=settings)
                save(joinpath(figure_dir, "$(key)_per_pmt_comp_time_uncert_$(uncert).$(extension)"), fig1)
                save(joinpath(figure_dir, "$(key)_per_pmt_max_comp_time_uncert$(uncert).$(extension)"), fig2)
            end
          
        end
    end
end


f = h5open(joinpath(ENV["WORK"], "photon_tables/lightsabre/photon_table_lightsabre_0.hd5"))

photons = f["photons/dataset_2"]
meta =  attrs(photons)

source_pos = JSON3.read(meta["source_pos"], SVector{3, Float64})

source_dir = sph_to_cart(acos(meta["dir_costheta"]), meta["dir_phi"])

closest_approach_distance(source_pos, source_dir, [0., 0., 0.])
meta["distance"]

close(f)



f = h5open(joinpath(ENV["WORK"], "photon_tables/lightsabre/hits/photon_table_lightsabre_0_hits.hd5"))
hits = f["pmt_hits/dataset_2250"]
meta =  attrs(hits)
source_pos = JSON3.read(meta["source_pos"], SVector{3, Float64})


p0 = meta["distance"] .* sph_to_cart(meta["pos_theta"], meta["pos_phi"])
dir = sph_to_cart(meta["dir_theta"], meta["dir_phi"])
ppos = p0 .- 200 .* dir

meta
medium = make_cascadia_medium_properties(0.95f0)
wl_range = (300f0, 800f0)
particle = Particle(Float32.(ppos), Float32.(dir), 0f0, Float32(meta["energy"]), Float32(400), PMuMinus)
source = LightsabreMuonEmitter(particle            ,
            medium, wl_range)
target = POM(SA_F32[0, 0, 0], 1)
spectrum = CherenkovSpectrum(wl_range, medium)
setup = PhotonPropSetup([source], [target], medium, spectrum, 1)
photons_prop = propagate_photons(setup)
hits_prop = make_hits_from_photons(photons_prop, setup)
hits[:, :]
model = models_track["Model A"]
log_amp, _ = get_log_amplitudes([convert(Particle{Float64}, particle)], [target], gpu(model))
sum(exp.(log_amp))
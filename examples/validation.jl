using PhotonPropagation
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


function proc_sim(photons, setup)
    hits = make_hits_from_photons(photons, setup, RotMatrix3(I))
    n_photons = sum(photons[:, :total_weight])
    hits_per_pmt = combine(groupby(hits, :pmt_id), nrow => :nhits)
    pmt_max = nrow(hits) > 0 ? hits_per_pmt[argmax(hits_per_pmt[:, :nhits]), :pmt_id] : 0
    return n_photons, nrow(hits), pmt_max, hits_per_pmt
end


function scan_distance(target, medium, spectrum, source_f, n_samples=10)
    distances = 10:10:100
    stats = []
    for distance in distances
        position = SA_F32[0, 0, distance]
        direction = SA_F32[1, 0 , 0]
        source = source_f(position, direction)
        
        for seed in 1:n_samples
            setup = PhotonPropSetup(source, target, medium, spectrum, seed)
            photons = propagate_photons(setup)
    
            n_photons, n_hits, pmt_max, _ = proc_sim(photons, setup)

            push!(stats, (n_photons=n_photons, n_hits=n_hits, pmt_max=pmt_max, distance=distance, seed=seed))
        end
    end

    stats = DataFrame(stats)
    return stats
end

function scan_phi(target, medium, spectrum, source_f, n_samples=10)
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
    
            _, _, _, hits_per_pmt = proc_sim(photons, setup)

            hits_per_pmt[!, :seed] .= seed
            hits_per_pmt[!, :dir_phi] .= phi
            push!(stats, hits_per_pmt)
        end
    end
    return reduce(vcat, stats)
end



function plot_distance_scans(stats, target, abs_len=27)
    # Start with isotropic emitter
    stats_mean = combine(groupby(stats, :distance), [:n_photons, :n_hits] .=> mean )

    @show stats_mean[:, :distance], stats_mean[:, :n_photons_mean]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Distance (m)", ylabel="Detected Photons", yscale=log10)
    scatter!(ax, stats_mean[:, :distance], stats_mean[:, :n_photons_mean], label="Sphere Hits")
    scatter!(ax, stats_mean[:, :distance], stats_mean[:, :n_hits_mean], label="PMT Hits")
    ylims!(ax, 1, 2E5)
  
    #attenuation_length(wl, scattering_coeff=0) = 1 / (1/absorption_length(wl, medium) + scattering_coeff)
    photon_expec(dist, att_len=25., norm=1) = norm*1E9*target.shape.radius^2 / (4*dist^2) * exp(-dist / att_len)
    model(x, p) = photon_expec.(x, p[1], absorption_length(stats[1, :wavelength]))

    fit = curve_fit(model, stats_mean[:, :distance], stats_mean[:, :n_photons_mean], [1], lower=[0.1])
    @show coef(fit) 
    lines!(ax, 10:1.:100, xs -> photon_expec(xs, coef(fit)[1], coef(fit)[2]), label=format("Attenuation length: {:.2f} m", coef(fit)[1]))
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


function run_scan(settings_dict)

    target = POM(SA_F32[0, 0, 0], 1)
    medium = make_cascadia_medium_properties(Float32(settings_dict["g"]))   

    if settings_dict["source_type"] == "isotropic"
        source_gen = (pos, _) -> PointlikeIsotropicEmitter(pos, 0f0, Int(settings_dict["n_photons"]))
        spectrum = Monochromatic(Float32(settings_dict["wavelength"]))
        
    elseif settings_dict["source_type"] == "cascade"
        wl_range = (300f0, 800f0)
        spectrum = CherenkovSpectrum(wl_range, medium)    
        source_gen = (pos, dir) -> ExtendedCherenkovEmitter(
            Particle(pos, dir, 0f0, Float32(settings_dict["energy"]), 0f0, PEMinus),
            medium, wl_range) 

    else
        error("unknown source type")
    end

    if settings_dict["scan_type"] == "distance"
        scan_f = scan_distance
    elseif settings_dict["scan_type"] == "phi"
        scan_f = scan_phi
    end

    return scan_f(target, medium, spectrum, source_gen, settings_dict["n_samples"])

end


settings = Dict(
    "source_type"=>"isotropic",
    "scan_type"=>"phi",
    "g"=>0.95,
    "n_samples"=> 1,
    "n_photons" => 1E9,
    "wavelength" => 450
)

stats = run_scan(settings)
metadata!(stats, "settings", JSON3.write(settings), style=:note)
plot_phi_scans(stats, POM(SA_F64[0, 0, 0], 1))
Arrow.write("validation_data_phi_scan.arrow", stats, file=false)

settings = Dict(
    "source_type"=>"isotropic",
    "scan_type"=>"distance",
    "g"=>0.95,
    "n_samples"=> 1,
    "n_photons" => 1E9,
    "wavelength" => 450
)

stats = run_scan(settings)
metadata!(stats, "settings", JSON3.write(settings), style=:note)
plot_distance_scans(stats, POM(SA_F64[0, 0, 0], 1))
Arrow.write("validation_data_distance_scan.arrow", stats, file=false)


settings = Dict(
    "source_type"=>"cascade",
    "scan_type"=>"phi",
    "g"=>0.95,
    "n_samples"=> 1,
    "energy" => 5E4,
    "wavelength" => 450
)

stats = run_scan(settings)
metadata!(stats, "settings", JSON3.write(settings), style=:note)
plot_phi_scans(stats, POM(SA_F64[0, 0, 0], 1))
Arrow.append("validation_data_phi_scan.arrow", stats)

settings = Dict(
    "source_type"=>"cascade",
    "scan_type"=>"distance",
    "g"=>0.95,
    "n_samples"=> 5,
    "energy" => 5E4,
    "wavelength" => 450
)

stats = run_scan(settings)
metadata!(stats, "settings", JSON3.write(settings), style=:note)
plot_distance_scans(stats, POM(SA_F64[0, 0, 0], 1))
Arrow.append("validation_data_distance_scan.arrow", stats)





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



# Save output


#=
hits = make_hits_from_photons(photons, setup)
event_record = Dict(:hits=>hits, :sources=>[source], :event_id=>uuid4())

using HDF5

fid = h5open("test.hd5", "w")

fid["test"] = hits[:, [:time]]


save_event("test.hd5", event_record)
=#
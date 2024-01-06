### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6d7c401a-ac83-11ee-24f7-3510dd251473
begin
    import Pkg
	Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/PLEnuM-group/julia-registry.git"))
	
    Pkg.activate(Base.current_project())
	#Pkg.activate("1.10")

	Pkg.add([
        Pkg.PackageSpec(name="PhotonPropagation"),
        Pkg.PackageSpec(name="StaticArrays"),
		Pkg.PackageSpec(name="PhysicsTools"),
		Pkg.PackageSpec(name="CairoMakie"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="HypertextLiteral"),
		Pkg.PackageSpec(name="Rotations"),
		Pkg.PackageSpec(name="LinearAlgebra"),
    ])
	Pkg.update()
    Pkg.instantiate()

	
    using PhotonPropagation
	using StaticArrays
	using PhysicsTools
	using CairoMakie
	using DataFrames
	using PlutoUI
	using HypertextLiteral
	using Rotations
	using LinearAlgebra
	
end

# ╔═╡ fa3f6880-d605-413d-aa87-412f3bdcdd70
begin
	# Allocate buffers where photons will be stored, will need this later.
	buffer_cpu, buffer_gpu = make_hit_buffers();
	
	# Define plotting functions
	
	function plot_photon_distributions(photons::AbstractVector, labels)
		
			fig = Figure()
			ax_l = Axis(fig[1, 1], xlabel="Time (ns)", ylabel="Counts", yscale=log10)
			ax_r = Axis(fig[1, 2], xlabel="Wavelength (nm)")
			# Plot arrival time distribution
			
			for (ph, lb) in zip(photons, labels)
				h = hist!(ax_l, ph[:, :time], weights=ph[:, :total_weight], bins=0:10:500, label=lb, fillto=1E-1)
				hist!(ax_r, ph[:, :wavelength], weights=ph[:, :total_weight], label=lb, )
			end
	
			ylims!(ax_l, 1E-1, 1E2)
			
			# Plot arrival spectrum
			axislegend()
			fig
		end
	
	function plot_photon_distributions(photons::AbstractDataFrame)
		return plot_photon_distributions([photons], ["Photons"])
	end
end

# ╔═╡ 86f89ecd-bbed-4589-991f-c9c04cbdce22
md"""
# Photon Propagation Demo

For photon propagation we need a few different ingredients:
* a photon source
* a detection target
* a medium

Let's set up the medium first.

## Medium setup

A medium has to implement various method related to the optical properties, as well as a few other physical properties.
Convenience constructors for water (with guesstimates of the properties at cascadia basin) as well as ice (deep homogeneous clear ice at Icecube) exist. Here we use the convenience constructor for water `make_cascadia_medium_properties`.
We can configure the mean (cos) scattering angle (used in the Henyey-Greenstein scattering function), as well as absorption and scattering length scaling. (optional)
"""

# ╔═╡ 40c6cec6-108e-4a62-8001-d654c0b5780d
medium = make_cascadia_medium_properties(0.95f0)


# ╔═╡ 022c1191-b77e-42a1-a3eb-c630725880a4
md"""
## Light source setup 

Next, we will configure our light source. Various different lightsources are implemented (see `src/lightyield.jl`).
Let's simulate the light yield of a lightsabre muon (energy losses are averaged over many muon propagations). The `FastLightsabreMuonEmitter` uses an energy dependent
parametrization of PROPOSAL simulations of muons. We also have to provide a spectrum, for which we can use the convenience function `make_cherenkov_spectrum`.
"""

# ╔═╡ 504cfcdd-b2f1-49ca-aac4-875dbae0fa29
begin
	# We first define a `particle` and then convert that into a light source
	energy = Float32(1E5)
	direction = SA_F32[0., 1., 0.]
	pos = SA_F32[0, 0, 0]
	len = Float32(1E4)
	t0 = 0f0
	p = Particle(pos, direction, t0, energy, len, PMuPlus)
	
	wl_range = (300f0, 800f0)
	spectrum = make_cherenkov_spectrum(wl_range, medium)
	source = FastLightsabreMuonEmitter(p, medium, spectrum)
	end

# ╔═╡ 82e1689d-029f-4fb5-ac49-ff7896a5b985
md"""
## Target setup

Finally, we have to provide the target. Here we'll use a P-ONE module.
"""

# ╔═╡ 50c96a7e-fc4f-421b-a586-2c22af2fe53a
begin
	tpos = SA_F32[0f0, 30f0, 30f0]
	module_id = 1
	target = POM(tpos, module_id)
	
	# Setup propagation
	seed = 1
		
	setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
	
	# Run photon propagation
	@time photons = propagate_photons(setup, buffer_cpu, buffer_gpu, copy_output=true)
end;


# ╔═╡ 86563c1d-46c6-4cc6-bbf4-4d18cfaff3e5
# Plot some distributions
plot_photon_distributions(photons)

# ╔═╡ 0fe88393-a25b-4d6f-921d-df893b20a595
md"""
## Multiple Sources

we can also propagate multiple sources at the same time. As an example, we will use PROPOSAl to propagate a muon and use the actual losses instead of the lightsabre
approximation.
"""

# ╔═╡ b496274d-6a93-4cc3-a3bb-55ebe47b7148
begin
	prop_p, losses = propagate_muon(p)
	losses_f32 = convert.(Ref(Particle{Float32}), losses)
	
	# Convert losses into emitters
	sources = ExtendedCherenkovEmitter.(losses_f32, Ref(medium), Ref(spectrum))
	
	setup_multi = PhotonPropSetup(sources, [target], medium, spectrum, seed)
	
	# Run photon propagation
	@time photons_no_approx = propagate_photons(setup_multi, buffer_cpu, buffer_gpu, copy_output=true)
end;

# ╔═╡ cf761f12-6e2f-427b-b839-cfc453a99eb0
plot_photon_distributions([photons, photons_no_approx], ["Lightsabre", "Full Losses"])

# ╔═╡ 573dba46-0957-44c8-8a92-59a6d1c9c1a2
@bind properties confirm(
	PlutoUI.combine() do Child
		@htl("""
		<h3>Changing Medium Properties</h3>
		We can also change the properties of the medium.
		<ul>
		$([
			@htl("<li>mean_sca_angle: $(Child("mean_sca_angle", PlutoUI.Slider(0f0:0.01f0:1f0, default=0.95f0, show_value=true)))"),
			@htl("<li>abs_scale: $(Child("abs_scale", PlutoUI.Slider(0.5f0:0.01f0:2f0, default=1f0, show_value=true)))"),
			@htl("<li>sca_scale: $(Child("sca_scale", PlutoUI.Slider(0.5f0:0.01f0:2f0, default=1f0, show_value=true)))")
			
		])
		</ul>
		""")
	end
)


# ╔═╡ 0612d2e1-aa19-4222-bede-84d7b7539ef0
begin
	medium_mod = make_cascadia_medium_properties(properties.mean_sca_angle, properties.abs_scale, properties.sca_scale)
	setup_mod = PhotonPropSetup([source], [target], medium_mod, spectrum, seed)
	photons_mod = propagate_photons(setup_mod, buffer_cpu, buffer_gpu, copy_output=true)
	plot_photon_distributions([photons, photons_mod], ["Baseline", "Modified"])
end

# ╔═╡ fe4af264-831c-4a74-8571-00976a5ad026
md"""
## Converting into PMT hits.

If we want to convert the photons that have intersected with our detection target to PMT hits, we have to apply the PMT acceptance. A convenience function already exists.
"""

# ╔═╡ a1cbdf5b-a0d9-4fc0-b602-b40f76799ae4
begin
	hits = make_hits_from_photons(photons, setup)
	calc_pe_weight!(hits, setup)
	expected_hits_per_pmt = combine(groupby(hits, :pmt_id), :total_weight => sum => :expected_hits)
end

# ╔═╡ 5e56acfa-a535-45c5-aced-ef8c3946f6ec
begin
	new_w = photons[:, :total_weight].* target.acceptance.pos_wl_acc_1.(photons[:, :wavelength]) ./ 0.0016
	
	fig, ax, _ = hist(photons[:, :wavelength], weights=new_w)
	hist!(ax, photons[:, :wavelength], weights=photons[:, :total_weight])
	fig
end

# ╔═╡ ef63f6d2-6ff5-43d8-890a-7ad88caf20ca
begin
	sum(expected_hits_per_pmt[:, :expected_hits]) / sum(photons[:, :total_weight])
	#target.pmt_area * get_pmt_count(target) / (π * target.shape.radius^2)
end

# ╔═╡ 8e767195-bdfd-4465-b1ee-2627e5ede972
md"""
## Other Sources and Targets


Different light sources are available:
- AxiconeEmitter
- ExtendedCherenkovEmitter (cascade)
- CherenkovTrackEmitter (track)
- PointlikeIsotropicEmitter
- PencilEmitter

As well as other targets:
- HomogeneousDetector{<:TargetShape} with shapes such as Spherical, Rectangular, Circular
- PixelatedDetector{<:TargetShape} for supporting multi-pmt detectors.
- POM
- DOM
"""

# ╔═╡ Cell order:
# ╠═6d7c401a-ac83-11ee-24f7-3510dd251473
# ╠═fa3f6880-d605-413d-aa87-412f3bdcdd70
# ╠═86f89ecd-bbed-4589-991f-c9c04cbdce22
# ╠═40c6cec6-108e-4a62-8001-d654c0b5780d
# ╟─022c1191-b77e-42a1-a3eb-c630725880a4
# ╠═504cfcdd-b2f1-49ca-aac4-875dbae0fa29
# ╟─82e1689d-029f-4fb5-ac49-ff7896a5b985
# ╠═50c96a7e-fc4f-421b-a586-2c22af2fe53a
# ╠═86563c1d-46c6-4cc6-bbf4-4d18cfaff3e5
# ╠═0fe88393-a25b-4d6f-921d-df893b20a595
# ╠═b496274d-6a93-4cc3-a3bb-55ebe47b7148
# ╠═cf761f12-6e2f-427b-b839-cfc453a99eb0
# ╟─573dba46-0957-44c8-8a92-59a6d1c9c1a2
# ╟─0612d2e1-aa19-4222-bede-84d7b7539ef0
# ╠═fe4af264-831c-4a74-8571-00976a5ad026
# ╠═a1cbdf5b-a0d9-4fc0-b602-b40f76799ae4
# ╠═5e56acfa-a535-45c5-aced-ef8c3946f6ec
# ╠═ef63f6d2-6ff5-43d8-890a-7ad88caf20ca
# ╠═8e767195-bdfd-4465-b1ee-2627e5ede972

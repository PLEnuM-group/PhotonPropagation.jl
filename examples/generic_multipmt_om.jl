using CairoMakie
using PhysicsTools
using NeutrinoTelescopeBase
using StaticArrays
using Distributions
using LinearAlgebra
using Rotations
using PhotonPropagation
using CSV
using DataFrames


function fibonacci_sphere(N::Int)
    points = Vector{Vector{Float64}}(undef, N)
    φ = π * (3 - sqrt(5))  # golden angle in radians


    for i in 0:N-1
        y = 1 - (i / (N - 1)) * 2  # y goes from 1 to -1
        radius = sqrt(1 - y * y)
        theta = φ * i

        x = cos(theta) * radius
        z = sin(theta) * radius

        points[i + 1] = [x, y, z]
    end

    return points
end

n_pmt = 20
points = fibonacci_sphere(20)
points_sph = SMatrix{2, 20, Float64}(reduce(hcat, cart_to_sph.(points)))
radius = 0.25

wl_acc = PhotonPropagation.RelQuantumEff("PhotonPropagation/assets/R6091_relqe.csv")

target = SphericalMultiPMTTarget(
    Spherical(SA[0., 0, 0], 0.25),
    (0.0762/2)^2 * π,
    points_sph,
    nothing,
    UInt16(1)
)

target.pmt_area*20 /  cross_section(target.shape, [1, 0, 0])
area_coverage = target.shape

nph = 100000
rand_th = acos.(rand(Uniform(-1, 1), nph))
rand_phi = rand(Uniform(0, 2π), nph)

rand_hit_dir = sph_to_cart.(rand_th, rand_phi)
rand_hit_pos = radius .* rand_hit_dir

pmt_hit_ids = check_pmt_hit(rand_hit_pos, rand_hit_dir, [], target, RotMatrix3(I))


make_pmt_hits()

sum(pmt_hit_ids .> 0)

apply_qe


qe_table = CSV.read("PhotonPropagation/examples/R6091_qe.csv", DataFrame)
qe_table = sort(qe_table, :wavelength)
qe_table[:, :rel_acceptance] = qe_table[:, :QE] ./ maximum(qe_table[:, :QE])

maximum(qe_table[:, :QE])

CSV.write("PhotonPropagation/assets/R6091_relqe.csv", qe_table)
qe_table



"""
    minimize_energy_on_sphere(N; max_iter=1000, lr=0.01)

Distribute N points approximately uniformly on a sphere
by minimizing pairwise Coulomb repulsion energy.
"""
function minimize_energy_on_sphere(N::Int; max_iter::Int = 1000, lr::Float64 = 0.01, anneal_rate=0.001, seed=1)
    # Initialize points randomly on sphere
    rng = MersenneTwister(seed)
    points = [normalize(randn(rng, SVector{3,Float64})) for _ in 1:N]
    energies = Float64[]

    for iter in 1:max_iter
        forces = [SVector{3, Float64}(0.0, 0.0, 0.0) for _ in 1:N]

        total_energy = 0.0
        # Compute repulsive forces
        for i in 1:N-1
            for j in i+1:N
                rij = points[i] - points[j]
                dist = norm(rij)
                force = rij / dist^3  # Coulomb repulsion
                forces[i] += force
                forces[j] -= force
                total_energy += 1.0 / dist
            end
        end

        push!(energies, total_energy)

        # Update positions with gradient descent & re-project to sphere
        for i in 1:N
            annealed_lr = lr * exp(-i*anneal_rate)
            new_point = points[i] + annealed_lr * forces[i]
            points[i] = normalize(new_point)
        end
    end

    return points, energies
end

function estimate_min_energy(N::Int)
    return 0.5 * N^2 - 1.5 * N^(3/2)
end


points, energies = minimize_energy_on_sphere(20, max_iter=10000, lr=0.01, anneal_rate=0.001, seed=2)
fig, ax, _ = lines(energies, axis=(;xscale=log10))

hlines!(ax, 2*estimate_min_energy(50))



fig
points, energies = minimize_energy_on_sphere(20, max_iter=100000, lr=0.1, anneal_rate=0)
lines(energies, axis=(;xscale=log10))


scatter(Point3f.(points))

reduce(hcat, cart_to_sph.(points))

SA[ 2.25125  0.634634  1.14901  1.65559  0.899014  1.69387  0.186803  2.51845  0.745349  1.04614  2.16353  0.962081  1.4594   2.00863  1.74027  1.47242  2.38773  2.94739   2.08877   1.40132
 6.01403  5.77419   4.92832  2.4311   4.00995   4.24677  2.99576   4.22343  0.843871  3.01157  1.74808  1.98313   5.76502  5.08536  3.43438  1.29802  2.87137  0.808321  0.726752  0.292785]
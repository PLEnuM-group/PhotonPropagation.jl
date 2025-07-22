export make_generic_multipmt_om, make_generic_multipmt_positions

function fibonacci_sphere(N::Integer)
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


function make_generic_multipmt_positions(T)

    return T.(SA[2.25125  0.634634  1.14901  1.65559  0.899014  1.69387  0.186803  2.51845  0.745349  1.04614  2.16353  0.962081  1.4594   2.00863  1.74027  1.47242  2.38773  2.94739   2.08877   1.40132
                 6.01403  5.77419   4.92832  2.4311   4.00995   4.24677  2.99576   4.22343  0.843871  3.01157  1.74808  1.98313   5.76502  5.08536  3.43438  1.29802  2.87137  0.808321  0.726752  0.292785])

end


function make_generic_multipmt_om(position::SVector{3, T}, radius::T, module_id::Integer, pmt_diameter=0.0762) where {T <: Real}

    PROJECT_ROOT = pkgdir(@__MODULE__)

    points_sph = make_generic_multipmt_positions(T)
   
    wl_acc = InterpQuantumEff(joinpath(PROJECT_ROOT, "assets/R6091_relqe.csv"))

    target = SphericalMultiPMTTarget(
        Spherical(position, radius),
        (pmt_diameter/2)^2 * π,
        points_sph,
        wl_acc,
        UInt16(module_id)
    )

    return target

end
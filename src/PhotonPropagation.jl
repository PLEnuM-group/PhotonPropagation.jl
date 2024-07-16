module PhotonPropagation

using PrecompileTools
using Reexport  


include("medium.jl")
include("spectrum.jl")
include("lightyield.jl")
include("detection.jl")
include("photon_prop_cuda.jl")
include("photon_propagation_setup.jl")
include("calc.jl")
include("processing.jl")
include("output.jl")


@reexport using .Medium
@reexport using .Spectral
@reexport using .LightYield
@reexport using .Detection
@reexport using .PhotonPropagationCuda
@reexport using .PhotonPropagationSetup
@reexport using .Calc
@reexport using .Processing
@reexport using .Output


#=
@setup_workload begin

    using PhysicsTools
    using CUDA
    using StaticArrays
    using Rotations
     
    cuda_functional = CUDA.functional()
    @show cuda_functional

    @compile_workload begin
        medium = make_cascadia_medium_properties(0.95)
        target = POM(SA_F32[0, 0, 0], UInt16(1))
        wl_range = (300.0f0, 800.0f0)
        
        pos = SA_F32[0, 0, 10]
        dir = SA_F32[0, 1, 0]
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(1E2),
            0.0f0,
            PEMinus
        )
        source = ExtendedCherenkovEmitter(particle, medium, wl_range)
        if cuda_functional
            spectrum = CherenkovSpectrum(wl_range, medium, 30)
            setup = PhotonPropSetup(source, target, medium, spectrum, 1)
            photons = propagate_photons(setup)
        end

        length = 400f0
        ppos = pos .- length/2 .* dir
        
        particle = Particle(
            Float32.(ppos),
            Float32.(dir),
            0.0f0,
            Float32(1E2),
            length,
            PMuMinus
        )

        if cuda_functional
            source = LightsabreMuonEmitter(particle, medium, wl_range)
            setup = PhotonPropSetup(source, target, medium, spectrum, 1)
            photons = propagate_photons(setup)
            orientation = rand(RotMatrix3)
            hits = make_hits_from_photons(photons, [target], orientation)
        end

        
    
    
    end

end
=#

end
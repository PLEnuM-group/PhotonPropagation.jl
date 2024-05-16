# PhotonPropagation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://plenum-group.github.io/PhotonPropagation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://plenum-group.github.io/PhotonPropagation.jl/dev/)
[![Build Status](https://github.com/plenum-group/PhotonPropagation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/plenum-group/PhotonPropagation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/plenum-group/PhotonPropagation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/plenum-group/PhotonPropagation.jl)

CUDA-accelerated Monte-Carlo simulation of photon transport in homogeneous media.

## Installation

This package is registered in the PLEnuM julia package [registry](https://github.com/PLEnuM-group/julia-registry). In order to use this registry run:
```{julia}
using Pkg
pkg"registry add https://github.com/PLEnuM-group/julia-registry"
```

Then install the package:
```{julia}
using Pkg
pkg"add PhotonPropagation"
```

## Example
```julia
using PhotonPropagation
using StaticArrays
using PhysicsTools
using CUDA


# Target Shape
module_position = SA[0., 0., 10.]
module_radius = 0.3
active_area = 16 * π * (0.0762)^2
shape = Spherical(module_position, module_radius)

# convert to Float32 for fast computation on gpu
shape = convert(Spherical{Float32}, shape)

# Setup target
target = HomogeneousDetector(shape, active_area, UInt16(1))

# Setup source
position = SA_F32[0., 0., 0.]
source = PointlikeIsotropicEmitter(position, 0f0, 100000)

# Setup medium
mean_sca_angle = 0.99f0
medium = make_cascadia_medium_properties(mean_sca_angle)

# Setup spectrum
spectrum = Monochromatic(450f0)

seed = 1

# Setup propagation
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)

# Run propagation
photons = propagate_photons(setup)
```
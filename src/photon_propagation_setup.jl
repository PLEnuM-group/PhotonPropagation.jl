module PhotonPropagationSetup
using ..Medium
using ..Spectral
using ..LightYield
using ..PhotonPropagationCuda
using CherenkovMediumBase
export PhotonPropSetup

mutable struct PhotonPropSetup{SV<:AbstractVector{<:PhotonSource},ST,M<:MediumProperties,C<:SpectralDist}
    sources::SV
    targets::ST
    medium::M
    spec_dist::C
    seed::Int64
    photon_scaling::Float64
end

function PhotonPropSetup(sources::AbstractVector{<:PhotonSource}, targets, medium, spectrum::Spectrum{<:InterpolatedSpectralDist}, seed; photon_scaling=1., spectrum_interp_steps=600) 
    cuda_spectral_dist = make_cuda_spectral_dist(spectrum.spectral_dist, spectrum.wl_range, spectrum_interp_steps)
    return PhotonPropSetup(sources, targets, medium, cuda_spectral_dist, seed, photon_scaling)
end

function PhotonPropSetup(sources::AbstractVector{<:PhotonSource}, targets, medium, spectrum::Spectrum, seed; photon_scaling=1.) 
    return PhotonPropSetup(sources, targets, medium, spectrum.spectral_dist, seed, photon_scaling)
end


end
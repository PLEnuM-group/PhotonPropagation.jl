module PhotonPropagationSetup
using ..Detection
using ..Medium
using ..Spectral
using ..LightYield

export PhotonPropSetup

mutable struct PhotonPropSetup{SV<:AbstractVector{<:PhotonSource},ST<:AbstractVector{<:PhotonTarget},M<:MediumProperties,C<:SpectralDist}
    sources::SV
    targets::ST
    medium::M
    spec_dist::C
    seed::Int64
end

function PhotonPropSetup(sources::AbstractVector{<:PhotonSource}, targets::AbstractVector{<:PhotonTarget}, medium, spectrum::Spectrum{<:InterpolatedSpectralDist}, seed) 
    cuda_spectral_dist = make_cuda_spectral_dist(spectrum.spectral_dist, spectrum.wl_range)
    return PhotonPropSetup(sources, targets, medium, cuda_spectral_dist, seed)
end

function PhotonPropSetup(sources::AbstractVector{<:PhotonSource}, targets::AbstractVector{<:PhotonTarget}, medium, spectrum::Spectrum, seed) 
    return PhotonPropSetup(sources, targets, medium, spectrum.spectral_dist, seed)
end

function PhotonPropSetup(
    source::PhotonSource,
    target::PhotonTarget,
    medium,
    spectrum,
    seed)

    setup = PhotonPropSetup([source], [target], medium, spectrum, Int64(seed))
    return setup
end


end
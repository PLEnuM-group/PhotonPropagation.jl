using PhotonPropagation
using Documenter

DocMeta.setdocmeta!(PhotonPropagation, :DocTestSetup, :(using PhotonPropagation); recursive=true)

makedocs(;
    modules=[PhotonPropagation],
    authors="Christian Haack",
    repo="https://github.com/chrhck/PhotonPropagation.jl/blob/{commit}{path}#{line}",
    sitename="PhotonPropagation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chrhck.github.io/PhotonPropagation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chrhck/PhotonPropagation.jl",
    devbranch="main",
)

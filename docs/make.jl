using Documenter
using DeepUncertainty

DocMeta.setdocmeta!(
    DeepUncertainty,
    :DocTestSetup,
    :(using DeepUncertainty);
    recursive=true,
)

makedocs(
    sitename="DeepUncertainty.jl",
    modules=[DeepUncertainty],
    repo="https://github.com/aced-differentiate/DeepUncertainty.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String["assets/flux.css"],
    ),
    pages=[
        "Home" => "index.md",
        "MC Dropout" => "layers/mclayers.md",
        "Bayesian" => "layers/bayesian.md",
    ],
)

deploydocs(
    repo="github.com/aced-differentiate/DeepUncertainty.jl",
    target="build",
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)

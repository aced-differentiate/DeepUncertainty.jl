using Documenter
using DeepUncertainty

DocMeta.setdocmeta!(
    DeepUncertainty,
    :DocTestSetup,
    :(using DeepUncertainty);
    recursive=true,
)

makedocs(
    modules=[DeepUncertainty],
    repo="https://github.com/aced-differentiate/DeepUncertainty.jl/blob/{commit}{path}#{line}",
    sitename="DeepUncertainty.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aced-differentiate.github.io/DeepUncertainty.jl",
        assets=String["assets/flux.css"]
    ),
    pages=["Home" => "index.md", 
            "MC Dropout" => "layers/mclayers.md"],
)

deploydocs(repo="github.com/aced-differentiate/DeepUncertainty.jl.git",
           target="build",
           branch="gh-pages",
           devbranch="main",
           push_preview=true)

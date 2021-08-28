using DeepUncertainty
using Documenter

DocMeta.setdocmeta!(
    DeepUncertainty,
    :DocTestSetup,
    :(using DeepUncertainty);
    recursive = true,
)

makedocs(;
    modules = [DeepUncertainty],
    authors = "DwaraknathT <dwarakasharma@gmail.com> and contributors",
    repo = "https://github.com/aced-differentiate/DeepUncertainty.jl/blob/{commit}{path}#{line}",
    sitename = "DeepUncertainty.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://DwaraknathT.github.io/DeepUncertainty.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/aced-differentiate/DeepUncertainty.jl")

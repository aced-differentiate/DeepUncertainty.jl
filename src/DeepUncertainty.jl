module DeepUncertainty

export Ensemble, ensemble_train
incluce("ensemble.jl")
using .ensemble: Ensemble, ensemble_train

end

using Base:String
using ArgParse

"""
"""
function getargs()

    s = ArgParseSettings()

    s.prog = "args.jl"  # program name (for usage & help screen)
    s.description = "CLI arguments for uncertainty estimation" # desciption (for help screen)

    @add_arg_table s begin
        "--model"
        help = "Model to use"
        "--dataset"
            help = "Dataset to use"
            arg_type = String
            default = "MNIST"
        "--batch_size"
            help = "Training and testing batch size"
            arg_type = Int 
            default = 128
        "arg1"
            help = "a positional argument"
    end

    parsed_args = parse_args(ARGS, s) # the result is a Dict{String,Any}
    return parsed_args
end

function printargs(args::Dict)
    println("Parsed args:")
    for pa in args
        println("  $(pa[1])  =>  $(pa[2])")
    end
    return nothing
end
    
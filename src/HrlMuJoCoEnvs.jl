module HrlMuJoCoEnvs

using Distributions
using Random
using UnsafeArrays
using LinearAlgebra
import UnsafeArrays: @uviews
using Distances: euclidean, sqeuclidean, cosine_dist

using LightXML
using LyceumBase, LyceumBase.Tools, LyceumMuJoCo, MuJoCo, Shapes
using StatsBase

include("../assets/AssetManager.jl")
using .AssetManager

include("worldstructure.jl")
using .WorldStructure

include("robots/robotbase.jl")
include("robots/pointmass.jl")
include("robots/ant.jl")

include("envs/envbase.jl")
include("envs/walkerbase.jl")

include("envs/antenv.jl")
include("envs/flagrunenv.jl")
include("envs/gatherenv.jl")
include("envs/mazeenv.jl")

end # module

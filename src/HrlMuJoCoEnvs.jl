module HrlMuJoCoEnvs

using Distributions
using Random
using UnsafeArrays
using LinearAlgebra
import UnsafeArrays: @uviews
using Distances: sqeuclidean, cosine_dist

using LightXML
using LyceumBase, LyceumBase.Tools, LyceumMuJoCo, MuJoCo, Shapes
using StatsBase

include("MazeStructure.jl")
using .MazeStructure

include("WalkerBase.jl")
include("Ant-v2.jl")  # must be included before maze and flagrun
include("AntMaze.jl")
include("Flagrun.jl")
include("AntGather.jl")

end # module

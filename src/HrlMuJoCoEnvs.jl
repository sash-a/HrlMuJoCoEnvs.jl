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

include("WorldStructure.jl")
using .WorldStructure

include("WalkerBase.jl")
include("Ant-v2.jl")  # must be included before maze and flagrun
include("AntMaze.jl")
include("Flagrun.jl")
include("Gather/GatherBase.jl")
include("Gather/AntGather.jl")
include("Gather/PointGather.jl")

end # module

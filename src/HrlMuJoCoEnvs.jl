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

include("robots/robotbase.jl")
include("robots/pointmass.jl")
include("robots/ant.jl")

include("envs/envbase.jl")
include("envs/walkerbase.jl")

include("WalkerBase.jl")
include("Ant-v2.jl")  # must be included before maze and flagrun
include("Flagrun.jl")

include("Maze/MazeBase.jl")
include("Maze/AntMazeEnv.jl")
include("Maze/PointMazeEnv.jl")

# include("Gather/GatherBase.jl")
# include("Gather/AntGather.jl")
# include("Gather/PointGather.jl")

include("envs/gatherenv.jl")
include("envs/mazeenv.jl")

end # module

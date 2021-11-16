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
include("Ant.jl")  # must be included before maze and flagrun
include("Flagrun.jl")

# Maze
include("Maze/MazeBase.jl")
include("Maze/AntMazeEnv.jl")
include("Maze/PointMazeEnv.jl")

# Gather
include("Gather/GatherBase.jl")
include("Gather/AntGather.jl")
include("Gather/PointGather.jl")

# Push
include("push/pushbase.jl")
include("push/antpushenv.jl")
include("push/pointpushenv.jl")

# Fall
include("fall/fallbase.jl")
include("fall/antfallenv.jl")
include("fall/pointfallenv.jl")


function make(name::String)
    if "AntMaze" == name
        AntMazeEnv
    elseif "PointMaze" == name
        PointMazeEnv
    elseif "AntGather" == name
        AntGatherEnv
    elseif "PointGather" == name
        PointGatherEnv
    elseif "AntPush" == name
        AntPushEnv
    elseif "PointPush" == name
        PointPushEnv
    elseif "AntFall" == name
        AntFallEnv
    elseif "PointFall" == name
        PointFallEnv
    else
        print("Unrecognized environment name")
        nothing
    end
end

end # module

module HrlMuJoCoEnvs

using Distributions
using Random
using UnsafeArrays
import UnsafeArrays:@uviews
using Distances:Euclidean

using LightXML
using LyceumBase, LyceumBase.Tools, LyceumMuJoCo, MuJoCo, Shapes

include("MazeStructure.jl")
using .MazeStructure

include("WalkerBase.jl")
include("Ant-v2.jl")  # must be included before maze and flagrun
include("AntMaze.jl")
include("Flagrun.jl")

end # module

module HrlMuJoCoEnvs

using LyceumMuJoCo

# include("Ant-v2.jl")
include("AntMaze.jl")
using LyceumMuJoCoViz
# using MuJoCo
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = AntMaze()
# env = AntV2()
# rand(actionspace(env))

# step!(env)
setaction!(env, zeros(8))
axes(statespace(env))
axes(statespace(getsim(env)))
getstate(env)
for t = 1:1000
    step!(env)
    setaction!(env, zeros(8))
    # states[:, t] .= getstate(env)
end

visualize(env, controller = (e -> setaction!(e, randn(8))))

end # module

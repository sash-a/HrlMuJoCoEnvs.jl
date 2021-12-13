include("../src/HrlMuJoCoEnvs.jl")
using .HrlMuJoCoEnvs

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = HrlMuJoCoEnvs.AntMazeEnv()

# getsim(env).mn
# typeof(getsim(env).mn)

# getsim(env).

# cm = getsim(env).mn[:geom_pos]
# cm[:,:goal]

reset!(env)
for i in 1:10000
    setaction!(env, rand(8))
    step!(env)
    o = getobs(env)
    # @show getreward(env)
    if isdone(env)
        println("done")
    end
end

visualize(env, controller = (e -> setaction!(e, [0, 1])))
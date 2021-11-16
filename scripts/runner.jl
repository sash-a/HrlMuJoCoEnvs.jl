include("../src/HrlMuJoCoEnvs.jl")
using .HrlMuJoCoEnvs

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = HrlMuJoCoEnvs.PointFallEnv()

for i in 1:100
    step!(env)
    o = getobs(env)
    setaction!(env, rand(2))
    # @show getreward(env)
    if isdone(env)
        println("done")
    end
end

visualize(env, controller = (e -> setaction!(e, [0, 1])))
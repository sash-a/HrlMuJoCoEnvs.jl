include("HrlMuJoCoEnvs.jl")
using .HrlMuJoCoEnvs

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = HrlMuJoCoEnvs.AntFlagrun()
# env = HrlMuJoCoEnvs.SwimmerFlagrun()

axes(actionspace(env))

for t = 1:10
    step!(env)
    setaction!(env, zeros(2) .+ 1)
    @show LyceumMuJoCo._torso_ang(env)
    # getobs(env)
end
visualize(env, controller = (e -> setaction!(e, randn(8))))
include("../src/HrlMuJoCoEnvs.jl")
using .HrlMuJoCoEnvs

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = HrlMuJoCoEnvs.AntFlagrun()
statespace(env)
model = getsim(env).mn
# model[:geom_pos]
print(model[:geom_pos][4:6])

getsim(env).m

getsim(env).mn[:geom_pos][4:6]=[5,1,1]

# for t = 1:10
#     step!(env)
#     setaction!(env, zeros(8))
#     println(getsim(env).mn[:geom_pos][7:9])
#     # println(LyceumMuJoCo._torso_height(env))
#     # @show LyceumMuJoCo._torso_ang(env)
# end
visualize(env, controller = (e -> setaction!(e, zeros(8))))
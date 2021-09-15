include("../src/HrlMuJoCoEnvs.jl")
using .HrlMuJoCoEnvs

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = HrlMuJoCoEnvs.PointGatherEnv()
# model = getsim(env).mn
# model[:geom_pos]
# i = 2
# model[:geom_pos][ngeom=Symbol("apple_$i")] = [1,1,0]
# model[:geom_pos][ngeom=:apple_2] = [3,2,1]
# getsim(env).mn[:geom_pos][ngeom=:apple_2]
# @show getsim(env).mn[:geom_pos][10:20]

# getsim(env).m

for i in 1:100
    reset!(env)
    step!(env)
    o = getobs(env)
    setaction!(env, zeros(2))
    @show getreward(env)
end

visualize(env, controller = (e -> setaction!(e, zeros(2) .+ 1)))
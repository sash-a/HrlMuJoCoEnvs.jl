include("../src/HrlMuJoCoEnvs.jl")
using .HrlMuJoCoEnvs

using MuJoCo
using LyceumMuJoCo
# using LyceumMuJoCoViz
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = HrlMuJoCoEnvs.AntEnv()
obsspace(env).robobs

env1 = HrlMuJoCoEnvs.PointMazeEnv()
reset!(env)
reset!(env1)
@show getobs(env)
@show getobs(env1)

setaction!(env, zeros(length(actionspace(env))) .+ 0.5)
setaction!(env1, zeros(length(actionspace(env1))) .+ 1)
step!(env)
step!(env1)

@show getobs(env)

HrlMuJoCoEnvs._torso_xy(env)
state=LyceumBase.allocate(statespace(env))
HrlMuJoCoEnvs._torso_xy(statespace(env)(state), env)

for i in 1:5
    step!(env)
    step!(env1)
    o = getobs(env)
    o1 = getobs(env1)

    @show o o1

    setaction!(env, zeros(length(actionspace(env))))
    setaction!(env, zeros(length(actionspace(env1))))
    
    @show getreward(env)
    @show getreward(env1)
    
    if isdone(env)
        println("done")
    end
end

visualize(env, controller = (e -> setaction!(e, rand(length(actionspace(env))))))
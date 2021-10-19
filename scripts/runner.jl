include("../src/HrlMuJoCoEnvs.jl")
using .HrlMuJoCoEnvs

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
mj_activate("/home/sasha/.mujoco/mjkey.txt")

env = HrlMuJoCoEnvs.AntMazeEnv()
getsim(env).m.nq
getsim(env).m.nv


pushenv = HrlMuJoCoEnvs.AntPushEnv(hide_block_joints=true)
mazeenv = HrlMuJoCoEnvs.AntMazeEnv()
pushsim = getsim(pushenv)
mazesim = getsim(mazeenv)

println(getsim(pushenv).d.qpos)
println(getsim(pushenv).d.qvel)

println(getobs(pushenv)[4:4 + 14])
println(getobs(pushenv)[19:end - 1])



println(getsim(mazeenv).d.qpos)
println(getsim(mazeenv).d.qvel)

# getsim(env).m
reset!(pushenv)
setaction!(pushenv, zeros(8))
step!(pushenv)

reset!(mazeenv)
setaction!(mazeenv, zeros(8))
step!(mazeenv)

for i in 1:100
    step!(env)
    o = getobs(env)
    setaction!(env, rand(8))
    # @show getreward(env)
    if isdone(env)
        println("done")
    end
end

visualize(env, controller = (e -> setaction!(e, rand(8))))
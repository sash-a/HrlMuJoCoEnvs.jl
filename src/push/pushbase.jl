abstract type AbstractPushEnv <: WalkerBase.AbstractWalkerMJEnv end

PUSH_DIST_THRESH = 5
PUSH_TARGET  = [0, 19]

function LyceumMuJoCo.step!(env::AbstractPushEnv)
    env.t += 1
    WalkerBase._step!(env)
end

function LyceumMuJoCo.reset!(env::AbstractPushEnv)
    env.t = 0
    r = WalkerBase._reset!(env)
    env.start_targ_dist = env.d_old = euclidean(_torso_xy(env), env.target)
    r
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AbstractPushEnv)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        env.d_old = euclidean(_torso_xy(env), env.target)
        1 - (env.d_old / env.start_targ_dist) + (env.d_old < PUSH_DIST_THRESH ? 2 : 0)
    end
end

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AbstractPushEnv) = false

LyceumMuJoCo.geteval(env::AbstractPushEnv) = euclidean(_torso_xy(env), PUSH_TARGET) < PUSH_DIST_THRESH ? 1 : 0
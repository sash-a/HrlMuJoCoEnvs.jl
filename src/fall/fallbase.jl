abstract type AbstractFallEnv <: WalkerBase.AbstractWalkerMJEnv end

FALL_DIST_THRESH = 5
FALL_TARGET  = [0, 27, 4.5]

function LyceumMuJoCo.step!(env::AbstractFallEnv)
    env.t += 1
    WalkerBase._step!(env)
end

function LyceumMuJoCo.reset!(env::AbstractFallEnv)
    env.start_targ_dist = env.d_old = euclidean(_torso_xyz(env), env.target)
    env.t = 0
    WalkerBase._reset!(env)
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AbstractFallEnv)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        env.d_old = euclidean(_torso_xyz(env), env.target)
        1 - (env.d_old / env.start_targ_dist) + (env.d_old < FALL_DIST_THRESH ? 2 : 0)
    end
end

LyceumMuJoCo.geteval(env::AbstractFallEnv) = euclidean(_torso_xyz(env), FALL_TARGET) < FALL_DIST_THRESH ? 1 : 0
# TODO remove the dependancy on torso_x
abstract type AbstractWalker3DMJEnv <: AbstractRoboticEnv end

@inline LyceumMuJoCo.getsim(env::AbstractWalker3DMJEnv) = env.sim
@inline LyceumMuJoCo.statespace(env::AbstractWalker3DMJEnv) = env.statespace

function LyceumMuJoCo.getstate!(state, env::AbstractWalker3DMJEnv)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        getstate!(shaped.simstate, getsim(env))
        shaped.last_torso_x = env.last_torso_x
    end
    state
end

function LyceumMuJoCo.setstate!(env::AbstractWalker3DMJEnv, state)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        setstate!(getsim(env), shaped.simstate)
        env.last_torso_x = shaped.last_torso_x
    end
    env
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AbstractWalker3DMJEnv)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        alive_bonus = 1.0
        reward = (LyceumMuJoCo._torso_x(shapedstate, env) - shapedstate.last_torso_x) / timestep(env)
        reward += alive_bonus
        reward -= controlcost(robot(env)) * sum(x->x^2, action)
        reward
    end
end

function LyceumMuJoCo.geteval(state, ::Any, ::Any, env::AbstractWalker3DMJEnv)
    checkaxes(statespace(env), state)
    @uviews state begin
        LyceumMuJoCo._torso_x(statespace(env)(state), env)
    end
end

function _reset!(env::AbstractWalker3DMJEnv)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end

LyceumMuJoCo.reset!(env::AbstractWalker3DMJEnv) = _reset!(env)

function LyceumMuJoCo.randreset!(rng::AbstractRNG, env::AbstractWalker3DMJEnv)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    perturb!(rng, env.randreset_distribution, env.sim.d.qpos)
    perturb!(rng, env.randreset_distribution, env.sim.d.qvel)
    forward!(env.sim)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end

function _step!(env::AbstractWalker3DMJEnv)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    step!(env.sim)
    env
end
LyceumMuJoCo.step!(env::AbstractWalker3DMJEnv) = _step!(env)
LyceumMuJoCo.isdone(::Any, ::Any, ::Any, env::AbstractWalker3DMJEnv) = isdone(robot(env))
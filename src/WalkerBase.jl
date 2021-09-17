module WalkerBase

using Distributions
using Random
using UnsafeArrays
using UnsafeArrays: @uviews
using LyceumBase, LyceumBase.Tools, LyceumMuJoCo, Shapes

abstract type AbstractWalkerMJEnv <: AbstractMuJoCoEnvironment end

function LyceumBase.tconstruct(T::Type{<:AbstractWalkerMJEnv}, n::Integer, modelpath::String)
    Tuple(T(s) for s in LyceumBase.tconstruct(LyceumMuJoCo.MJSim, n, modelpath, skip=4))
end

@inline LyceumMuJoCo.getsim(env::T) where T <: AbstractWalkerMJEnv = env.sim
@inline LyceumMuJoCo.statespace(env::T) where T <: AbstractWalkerMJEnv = env.statespace

function LyceumMuJoCo.getstate!(state, env::T) where T <: AbstractWalkerMJEnv
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        getstate!(shaped.simstate, env.sim)
        shaped.last_torso_x = env.last_torso_x
    end
    state
end

function LyceumMuJoCo.setstate!(env::T, state) where T <: AbstractWalkerMJEnv
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        setstate!(env.sim, shaped.simstate)
        env.last_torso_x = shaped.last_torso_x
    end
    env
end


@inline LyceumMuJoCo.obsspace(env::T) where T <: AbstractWalkerMJEnv = env.obsspace

function LyceumMuJoCo.getobs!(obs, env::T) where T <: AbstractWalkerMJEnv
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[2:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end
    obs
end


function LyceumMuJoCo.getreward(state, action, ::Any, env::T) where T <: AbstractWalkerMJEnv
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        alive_bonus = 1.0
        reward = (LyceumMuJoCo._torso_x(shapedstate, env) - shapedstate.last_torso_x) / timestep(env)
        reward += alive_bonus
        reward -= 0.5 * sum(x->x^2, action)  # this 0.5 is the ant control cost
        reward
    end
end

function LyceumMuJoCo.geteval(state, ::Any, ::Any, env::T) where T <: AbstractWalkerMJEnv
    checkaxes(statespace(env), state)
    @uviews state begin
        LyceumMuJoCo._torso_x(statespace(env)(state), env)
    end
end

function _reset!(env::T) where T <: AbstractWalkerMJEnv
    reset!(env.sim)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end

LyceumMuJoCo.reset!(env::T) where T <: AbstractWalkerMJEnv = _reset!(env)

function LyceumMuJoCo.randreset!(rng::AbstractRNG, env::T) where T <: AbstractWalkerMJEnv
    LyceumMuJoCo.reset_nofwd!(env.sim)
    perturb!(rng, env.randreset_distribution, env.sim.d.qpos)
    perturb!(rng, env.randreset_distribution, env.sim.d.qvel)
    forward!(env.sim)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end

function _step!(env::T) where T <: AbstractWalkerMJEnv
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    step!(env.sim)
    env
end
LyceumMuJoCo.step!(env::T) where T <: AbstractWalkerMJEnv = _step!(env)

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::T) where T <: AbstractWalkerMJEnv = false

@inline LyceumMuJoCo._torso_x(shapedstate::ShapedView, ::T) where T <: AbstractWalkerMJEnv = shapedstate.simstate.qpos[1]
@inline LyceumMuJoCo._torso_x(env::T) where T <: AbstractWalkerMJEnv = env.sim.d.qpos[1]

# @inline _torso_xy(shapedstate::ShapedView, ::AbstractWalkerMJEnv) = shapedstate.simstate.qpos[1:2]
# @inline _torso_xy(env::AbstractWalkerMJEnv) = env.sim.d.qpos[1:2]

# @inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AbstractWalkerMJEnv) = shapedstate.simstate.qpos[3]
# # TODO get the quat and project on xy
# @inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AbstractWalkerMJEnv) = shapedstate.simstate.qpos[4]
end # module
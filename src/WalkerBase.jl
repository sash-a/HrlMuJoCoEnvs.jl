abstract type WalkerBase <: AbstractMuJoCoEnvironment end

function LyceumBase.tconstruct(T::Type{<:WalkerBase}, n::Integer, modelpath::String)
    Tuple(T(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

@inline LyceumMuJoCo.getsim(env::T) where T <: WalkerBase = env.sim
@inline LyceumMuJoCo.statespace(env::T) where T <: WalkerBase = env.statespace

function LyceumMuJoCo.getstate!(state, env::T) where T <: WalkerBase
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        getstate!(shaped.simstate, env.sim)
        shaped.last_torso_x = env.last_torso_x
    end
    state
end

function LyceumMuJoCo.setstate!(env::T, state) where T <: WalkerBase
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        setstate!(env.sim, shaped.simstate)
        env.last_torso_x = shaped.last_torso_x
    end
    env
end


@inline LyceumMuJoCo.obsspace(env::T) where T <: WalkerBase = env.obsspace

function LyceumMuJoCo.getobs!(obs, env::T) where T <: WalkerBase
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


function LyceumMuJoCo.getreward(state, action, ::Any, env::T) where T <: WalkerBase
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        alive_bonus = 1.0
        reward = (LyceumMuJoCo._torso_x(shapedstate, env) - shapedstate.last_torso_x) / timestep(env)
        reward += alive_bonus
        reward -= 1e-3 * sum(x->x^2, action)
        reward
    end
end

function LyceumMuJoCo.geteval(state, ::Any, ::Any, env::T) where T <: WalkerBase
    checkaxes(statespace(env), state)
    @uviews state begin
        LyceumMuJoCo._torso_x(statespace(env)(state), env)
    end
end


function LyceumMuJoCo.reset!(env::T) where T <: WalkerBase
    reset!(env.sim)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end

function LyceumMuJoCo.randreset!(rng::AbstractRNG, env::T) where T <: WalkerBase
    LyceumMuJoCo.reset_nofwd!(env.sim)
    perturb!(rng, env.randreset_distribution, env.sim.d.qpos)
    perturb!(rng, env.randreset_distribution, env.sim.d.qvel)
    forward!(env.sim)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end


function LyceumMuJoCo.step!(env::T) where T <: WalkerBase
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    step!(env.sim)
    env
end

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::T) where T <: WalkerBase = false

@inline LyceumMuJoCo._torso_x(shapedstate::ShapedView, ::T) where T <: WalkerBase = shapedstate.simstate.qpos[1]
@inline LyceumMuJoCo._torso_x(env::T) where T <: WalkerBase = env.sim.d.qpos[1]

# @inline _torso_xy(shapedstate::ShapedView, ::WalkerBase) = shapedstate.simstate.qpos[1:2]
# @inline _torso_xy(env::WalkerBase) = env.sim.d.qpos[1:2]

# @inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::WalkerBase) = shapedstate.simstate.qpos[3]
# # TODO get the quat and project on xy
# @inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::WalkerBase) = shapedstate.simstate.qpos[4]
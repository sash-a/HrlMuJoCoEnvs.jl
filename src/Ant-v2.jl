using Distributions
using UnsafeArrays
import UnsafeArrays: @uviews

using LyceumBase, LyceumBase.Tools, LyceumMuJoCo, MuJoCo, Shapes

mutable struct AntV2{SIM<:MJSim, S, O} <: AbstractMuJoCoEnvironment
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64

    function AntV2(sim::MJSim)
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 2),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntV2}, n::Integer)
    modelpath = joinpath(@__DIR__, "..", "assets", "ant.xml")
    Tuple(AntV2(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntV2() = first(tconstruct(AntV2, 1))

@inline LyceumMuJoCo.getsim(env::AntV2) = env.sim

@inline LyceumMuJoCo.statespace(env::AntV2) = env.statespace
function LyceumMuJoCo.getstate!(state, env::AntV2)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        getstate!(shaped.simstate, env.sim)
        shaped.last_torso_x = env.last_torso_x
    end
    state
end
function LyceumMuJoCo.setstate!(env::AntV2, state)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        setstate!(env.sim, shaped.simstate)
        env.last_torso_x = LyceumMuJoCo._torso_x(env)
    end
    env
end

@inline LyceumMuJoCo.obsspace(env::AntV2) = env.obsspace
function LyceumMuJoCo.getobs!(obs, env::AntV2)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
        # missing contact forces
    end
    obs
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AntV2)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        alive_bonus = 1.0
        reward = (LyceumMuJoCo._torso_x(env) - env.last_torso_x) / timestep(env)
        reward += alive_bonus
        reward -= 5e-4 * sum(x->x^2, action)
        reward
    end
end

function LyceumMuJoCo.geteval(state, ::Any, ::Any, env::AntV2)
    checkaxes(statespace(env), state)
    @uviews state begin
        LyceumMuJoCo._torso_x(env)
    end
end

function LyceumMuJoCo.reset!(env::AntV2)
    reset!(env.sim)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end

function LyceumMuJoCo.step!(env::AntV2)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    step!(env.sim)
    env
end

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntV2)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = _torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.2 >= height >= 1)
        done
    end
end

# @inline LyceumMuJoCo._torso_x(shapedstate::ShapedView, ::AntV2) = shapedstate.simstate.qpos[1]
@inline LyceumMuJoCo._torso_x(env::AntV2) = env.sim.d.qpos[1]

@inline _torso_xy(shapedstate::ShapedView, ::AntV2) = shapedstate.simstate.qpos[1:2]
@inline _torso_xy(env::AntV2) = env.sim.d.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntV2) = shapedstate.simstate.qpos[3]
# TODO get the quat and project on xy
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntV2) = shapedstate.simstate.qpos[4]
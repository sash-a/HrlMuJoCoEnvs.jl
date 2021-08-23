mutable struct AntV2{SIM<:MJSim, S, O} <: WalkerBase
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

AntV2() = first(tconstruct(AntV2, 1, joinpath(@__DIR__, "..", "assets", "ant.xml")))

function LyceumMuJoCo.getobs!(obs, env::T) where T <: WalkerBase
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end
    obs
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

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntV2) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_ang(env::AntV2) = env.sim.d.qpos[4]  # TODO get the quat and project on xy

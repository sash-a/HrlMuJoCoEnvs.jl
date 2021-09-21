mutable struct AntEnv{SIM<:MJSim, S, O} <: AbstractWalker3DMJEnv
    sim::SIM
    robot::Ant
    statespace::S
    obsspace::O
    last_torso_x::Float64
    randreset_distribution::Uniform{Float64}

    function AntEnv(sim::MJSim)
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 2),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, Ant(sim), sspace, ospace, 0, Uniform(-0.1, 0.1))
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{::AntEnv}, n::Integer)
    Tuple(AntEnv(s) for s in LyceumBase.tconstruct(LyceumMuJoCo.MJSim, n, getfile(Ant), skip=4))
end

AntEnv() = first(tconstruct(WalkerEnv, 1))
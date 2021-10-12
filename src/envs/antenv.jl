mutable struct AntEnv{SIM<:MJSim, S, O} <: AbstractWalker3DMJEnv
    sim::SIM
    robot::Ant
    statespace::S
    obsspace::O
    last_torso_x::Float64
    randreset_distribution::Uniform{Float64}

    function AntEnv(sim::MJSim)
        robot = Ant(sim; cropqpos=true)
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(robobs=obsspace(robot))

        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, robot, sspace, ospace, 0, Uniform(-0.1, 0.1))
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntEnv}, n::Integer)
    Tuple(AntEnv(s) for s in LyceumBase.tconstruct(LyceumMuJoCo.MJSim, n, getfile(Ant), skip=4))
end

AntEnv() = first(tconstruct(AntEnv, 1))

robot(env::AntEnv) = env.robot
LyceumMuJoCo.obsspace(env::AntEnv) = env.obsspace

_set_prev_pos!(env::AntEnv, pos::Float64) = env.last_torso_x = pos
_set_prev_pos!(shapedstate, ::AntEnv, pos::Float64) = shapedstate.last_torso_x = pos
get_prev_pos(env::AntEnv) = env.last_torso_x
get_prev_pos(shapedstate, ::AntEnv) = shapedstate.last_torso_x

getpos(env::AntEnv) = LyceumMuJoCo._torso_x(env)
getpos(shapedstate, env::AntEnv) = LyceumMuJoCo._torso_x(shapedstate, env)

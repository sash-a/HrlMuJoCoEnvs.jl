abstract type AbstractRoboticEnv <: AbstractMuJoCoEnvironment end

robot(::AbstractRoboticEnv) = nothing
@inline LyceumMuJoCo.obsspace(env::AbstractRoboticEnv) = obsspace(robot(env))
LyceumMuJoCo.getobs!(obs, env::AbstractRoboticEnv) = getobs(getsim(env), robot(env))


function LyceumBase.tconstruct(Env::Type{AbstractRoboticEnv}, n::Integer; skip=4)
    Tuple(Env(s) for s in LyceumBase.tconstruct(MJSim, n, getfile(robot(Env)), skip=skip))
end

@inline LyceumMuJoCo._torso_x(env::AbstractRoboticEnv) = first(_torso_xy(getsim(env), robot(env)))
@inline LyceumMuJoCo._torso_x(shapedstate::ShapedView, env::AbstractRoboticEnv) = first(_torso_xy(shapedstate, robot(env)))

@inline _torso_xy(env::AbstractRoboticEnv) = _torso_xy(getsim(env), robot(env))
@inline _torso_xy(shapedstate::ShapedView, env::AbstractRoboticEnv) = _torso_xy(shapedstate, robot(env))

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, env::AbstractRoboticEnv) = LyceumMuJoCo._torso_height(shapedstate, robot(env))
@inline LyceumMuJoCo._torso_height(env::AbstractRoboticEnv) = LyceumMuJoCo._torso_height(getsim(env), robot(env))

@inline LyceumMuJoCo._torso_ang(env::AbstractRoboticEnv) =  LyceumMuJoCo._torso_ang(getsim(env), robot(env))
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, env::AbstractRoboticEnv) = LyceumMuJoCo._torso_ang(shapedstate, robot(env))
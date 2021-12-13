mutable struct PointPushEnv{SIM<:MJSim, S, O} <: AbstractPushEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    target::Vector{Number}
    t::Int
    d_old::Float64
    start_targ_dist::Float64
    rng::MersenneTwister

    function PointPushEnv(sim::MJSim; rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            agent_pos=VectorShape(Float64, 2),
            agent_vel=VectorShape(Float64, 2),
            t=ScalarShape(Float64)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, PUSH_TARGET, 0, 0, 0, rng)
        getsim(env).mn[:geom_pos][:, Val(:goal)] = [PUSH_TARGET..., 0]
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{PointPushEnv}, n::Integer; easy=false, seed=nothing)
    antmodelpath = joinpath(AssetManager.dir, "pointmass.xml")
    filename="pointpushtmp.xml"

    structure = easy ? WorldStructure.ez_push_maze : WorldStructure.push_maze
    WorldStructure.create_world(antmodelpath, structure=structure, wsize=8, filename=filename)
    modelpath = joinpath(AssetManager.dir, filename)

    Tuple(PointPushEnv(s, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

PointPushEnv(;easy=false, seed=nothing) = first(tconstruct(PointPushEnv, 1; easy=easy, seed=seed))

function LyceumMuJoCo.getobs!(obs, env::PointPushEnv)
    checkaxes(obsspace(env), obs)

    dn = env.sim.dn
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        targetvec = env.target - _torso_xy(env)
        # angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)
        # copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])
        copyto!(shaped.targetvec, normalize(targetvec))
        copyto!(shaped.d_old, [env.d_old / 1000])

        shaped.agent_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent]
        shaped.agent_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
        shaped.t = env.t * 0.001
    end

    obs
end

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::PointPushEnv) = false

@inline _torso_xy(env::PointPushEnv) = [env.sim.dn.xpos[:x, :agent], env.sim.dn.xpos[:y, :agent]]
@inline _torso_xy(shapedstate::ShapedView, env::PointPushEnv) = [env.sim.dn.xpos[:x, :agent], env.sim.dn.xpos[:y, :agent]]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::PointPushEnv) = 0.
@inline LyceumMuJoCo._torso_height(env::PointPushEnv) = 0.

@inline LyceumMuJoCo._torso_ang(env::PointPushEnv) = 0.
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::PointPushEnv) = 0.
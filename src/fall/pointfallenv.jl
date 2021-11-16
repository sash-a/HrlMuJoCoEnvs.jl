mutable struct PointFallEnv{SIM<:MJSim, S, O} <: AbstractFallEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    target::Vector{Number}
    t::Int
    d_old::Float64
    start_targ_dist::Float64
    rng::MersenneTwister


    function PointFallEnv(sim::MJSim; rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 3),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            agent_pos=VectorShape(Float64, 3),
            agent_vel=VectorShape(Float64, 2),
            t=ScalarShape(Float64)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, FALL_TARGET, 0, 0, 0, rng)
        getsim(env).mn[:geom_pos][ngeom=:target_geom] = FALL_TARGET
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{PointFallEnv}, n::Integer; seed=nothing)
    antmodelpath = joinpath(AssetManager.dir, "pointmass_xyz.xml")
    filename="pointfalltmp.xml"

    WorldStructure.create_world(antmodelpath, structure=WorldStructure.ez_fall_maze, wsize=8, filename=filename)
    modelpath = joinpath(AssetManager.dir, filename)

    Tuple(PointFallEnv(s, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

PointFallEnv(;seed=nothing) = first(tconstruct(PointFallEnv, 1; seed=seed))

function LyceumMuJoCo.getobs!(obs, env::PointFallEnv)
    checkaxes(obsspace(env), obs)

    dn = env.sim.dn
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        targetvec = env.target - _torso_xyz(env)
        # angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)
        # copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])
        copyto!(shaped.targetvec, normalize(targetvec[1:2]))
        copyto!(shaped.d_old, [env.d_old / 1000])

        shaped.agent_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent], dn.xpos[:z, :agent]
        shaped.agent_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
        shaped.t = env.t * 0.001
    end

    obs
end

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::PointFallEnv) = false

@inline _torso_xyz(env::PointFallEnv) = [env.sim.dn.xpos[:x, :agent], env.sim.dn.xpos[:y, :agent], env.sim.dn.xpos[:z, :agent]]
@inline _torso_xy(env::PointFallEnv) = [env.sim.dn.xpos[:x, :agent], env.sim.dn.xpos[:y, :agent]]
@inline _torso_xy(shapedstate::ShapedView, ::PointFallEnv) = shapedstate.agent_pos

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::PointFallEnv) = 0.
@inline LyceumMuJoCo._torso_height(env::PointFallEnv) = 0.

@inline LyceumMuJoCo._torso_ang(env::PointFallEnv) = 0.
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::PointFallEnv) = 0.
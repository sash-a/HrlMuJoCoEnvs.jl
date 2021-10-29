mutable struct AntFallEnv{SIM<:MJSim, S, O} <: AbstractFallEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    target::Vector{Number}
    t::Int
    d_old::Float64
    start_targ_dist::Float64
    rng::MersenneTwister


    function AntFallEnv(sim::MJSim; rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 3),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            qpos=VectorShape(Float64, sim.m.nq),
            qvel=VectorShape(Float64, sim.m.nv),
            t=ScalarShape(Float64)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, FALL_TARGET, 0, 0, 0, rng)
        getsim(env).mn[:geom_pos][ngeom=:target_geom] = FALL_TARGET
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntFallEnv}, n::Integer; seed=nothing)
    antmodelpath = joinpath(AssetManager.dir, "easier_ant.xml")
    filename="antfalltmp.xml"

    WorldStructure.create_world(antmodelpath, structure=WorldStructure.fall_maze, wsize=8, filename=filename)
    modelpath = joinpath(AssetManager.dir, filename)

    Tuple(AntFallEnv(s, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

AntFallEnv(;seed=nothing) = first(tconstruct(AntFallEnv, 1; seed=seed))

function LyceumMuJoCo.getobs!(obs, env::AntFallEnv)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        targetvec = env.target - _torso_xyz(env)[1:2]
        # angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)
        # copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])

        copyto!(shaped.targetvec, normalize(targetvec))
        copyto!(shaped.d_old, [env.d_old / 1000])
        copyto!(shaped.qpos, qpos)
        copyto!(shaped.qvel, env.sim.d.qvel)
        shaped.t = env.t * 0.001
        clamp!(shaped.qvel, -10, 10)
    end

    obs
end

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntFallEnv) = false

@inline _torso_xy(env::AntFallEnv) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntFallEnv) = shapedstate.simstate.qpos[1:2]
@inline _torso_xyz(env::AntFallEnv) = env.sim.d.qpos[1:3]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntFallEnv) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntFallEnv) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntFallEnv) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntFallEnv) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
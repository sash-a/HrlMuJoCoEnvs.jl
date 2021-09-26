mutable struct AntPushEnv{SIM<:MJSim, S, O} <: AbstractPushEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    target::Vector{Number}
    t::Int
    d_old::Float64
    start_targ_dist::Float64
    rng::MersenneTwister


    function AntPushEnv(sim::MJSim; rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
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
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, PUSH_TARGET, 0, 0, 0, rng)
        getsim(env).mn[:geom_pos][ngeom=:target_geom] = [PUSH_TARGET..., 0]
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntPushEnv}, n::Integer; seed=nothing)
    antmodelpath = joinpath(AssetManager.dir, "easier_ant.xml")
    filename="antpushtmp.xml"

    WorldStructure.create_world(antmodelpath, structure=WorldStructure.push_maze, wsize=8, filename=filename)
    modelpath = joinpath(AssetManager.dir, filename)

    Tuple(AntPushEnv(s, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

AntPushEnv(;seed=nothing) = first(tconstruct(AntPushEnv, 1; seed=seed))

function LyceumMuJoCo.getobs!(obs, env::AntPushEnv)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        targetvec = env.target - _torso_xy(env)
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

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntPushEnv)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        !(all(isfinite, state) && 0.2 <= height <= 1)
    end
end

@inline _torso_xy(env::AntPushEnv) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntPushEnv) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntPushEnv) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntPushEnv) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntPushEnv) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntPushEnv) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
mutable struct AntMazeEnv{SIM<:MJSim, S, O} <: AbstractMazeEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    structure::Matrix{AbstractBlock}
    target::Vector{Number}
    t::Int
    d_old::Float64
    start_targ_dist::Float64
    rng::MersenneTwister


    function AntMazeEnv(sim::MJSim; structure=WorldStructure.basic_maze_structure, rng=MersenneTwister())
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
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, structure, [0, 0], 0, 0, 0, rng)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntMazeEnv}, n::Integer; structure::Matrix{<:AbstractBlock}=WorldStructure.basic_maze_structure, seed=nothing, filename="tmp.xml")
    antmodelpath = joinpath(AssetManager.dir, "easier_ant.xml")
    WorldStructure.create_world(antmodelpath, structure=structure, filename=filename)
    modelpath = joinpath(AssetManager.dir, filename)

    Tuple(AntMazeEnv(s, structure=structure, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

AntMazeEnv(;structure::Matrix{<: AbstractBlock}=WorldStructure.basic_maze_structure, seed=nothing) = first(tconstruct(AntMazeEnv, 1; structure=structure, seed=seed))

function LyceumMuJoCo.getobs!(obs, env::AntMazeEnv)
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

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntMazeEnv)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.2 <= height <= 1) || env.d_old < FLAGRUN_DIST_THRESH
        done
    end
end

@inline _torso_xy(env::AntMazeEnv) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntMazeEnv) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntMazeEnv) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntMazeEnv) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntMazeEnv) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntMazeEnv) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
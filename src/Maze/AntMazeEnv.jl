mutable struct AntMazeEnv{SIM<:MJSim,S,O} <: AbstractMazeEnv
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


    function AntMazeEnv(sim::MJSim; structure = WorldStructure.basic_maze_structure, rng = MersenneTwister())
        sspace = MultiShape(
            target = VectorShape(Float64, 2),
            simstate = statespace(sim),
            last_torso_x = ScalarShape(Float64),
        )
        ospace = MultiShape(
            # targetvec=VectorShape(Float64, 2),
            # d_old=VectorShape(Float64, 1),
            qpos = VectorShape(Float64, sim.m.nq),
            qvel = VectorShape(Float64, sim.m.nv),
            t = ScalarShape(Float64),
            target = VectorShape(Float64, 2),
        )
        env = new{typeof(sim),typeof(sspace),typeof(ospace)}(sim, sspace, ospace, 0, structure, [0, 0], 0, 0, 0, rng)
        reset!(env)
    end
end

function LyceumBase.tconstruct(
    ::Type{AntMazeEnv},
    n::Integer;
    structure::Matrix{<:AbstractBlock} = WorldStructure.basic_maze_structure,
    antmodelpath = joinpath(AssetManager.dir, "easier_ant.xml"),
    seed = nothing,
)
    # just the file - rand numbers for mpi parallelism to not create the same file
    filename = "antmaze-tmp$(rand(1:1000000)).xml"  
    newmodelpath = joinpath(AssetManager.dir, filename)  # the whole path

    WorldStructure.create_world(antmodelpath, structure = structure, wsize = 8, filename = newmodelpath)

    envs = Tuple(
        AntMazeEnv(s, structure = structure, rng = MersenneTwister(seed)) for
        s in LyceumBase.tconstruct(MJSim, n, newmodelpath, skip = 5)
    )

    rm(newmodelpath)

    envs
end

AntMazeEnv(; structure::Matrix{<:AbstractBlock} = WorldStructure.basic_maze_structure, seed = nothing) =
    first(tconstruct(AntMazeEnv, 1; structure = structure, seed = seed))

function LyceumMuJoCo.getobs!(obs, env::AntMazeEnv)
    checkaxes(obsspace(env), obs)
    @views begin
        shaped = obsspace(env)(obs)
        # targetvec = env.target - _torso_xy(env)
        # angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)
        # copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])

        # copyto!(shaped.targetvec, normalize(targetvec))
        # copyto!(shaped.d_old, [env.d_old / 1000])

        copyto!(shaped.qpos, env.sim.d.qpos)
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
        shaped.t = env.t * 0.001
        copyto!(shaped.target, env.target * 0.01)
    end

    obs
end

@inline _torso_xy(env::AntMazeEnv) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntMazeEnv) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntMazeEnv) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntMazeEnv) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntMazeEnv) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind+3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntMazeEnv) =
    torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind+3])

MAZE_DIST_THRESH = 20

mutable struct AntMaze{SIM<:MJSim, S, O} <: WalkerBase.AbstractWalkerMJEnv
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


    function AntMaze(sim::MJSim; structure=WorldStructure.basic_maze_structure, rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            cropped_qpos=VectorShape(Float64, sim.m.nq - 2),
            qvel=VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, structure, [0, 0], 0, 0, 0, rng)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntMaze}, n::Integer; structure::Matrix{<:AbstractBlock}=WorldStructure.basic_maze_structure, seed=nothing, filename="tmp.xml")
    antmodelpath = joinpath(AssetManager.dir, "ant.xml")
    WorldStructure.create_world(antmodelpath, structure=structure, filename=filename)
    modelpath = joinpath(AssetManager.dir, filename)

    Tuple(AntMaze(s, structure=structure, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntMaze(;structure::Matrix{<: AbstractBlock}=WorldStructure.basic_maze_structure, seed=nothing) = first(tconstruct(AntMaze, 1; structure=structure, seed=seed))

function LyceumMuJoCo.step!(env::AntMaze)
    env.t += 1
    WalkerBase._step!(env)
end

function LyceumMuJoCo.getobs!(obs, env::AntMaze)
    # TODO sensor readings
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        targetvec = env.target - _torso_xy(env)
        angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)

        copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])
        copyto!(shaped.d_old, [env.d_old / 1000])
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end

    obs
end

function _movetarget!(env::AntMaze)
    zones = [(12, 20, -4, 12), (-4, 20, 12, 20), (-4, 12, -4, 4)]  # [(xmin, xmax, ymin, ymax)...]
    areas = map(((xmin, xmax, ymin, ymax),)->((xmax-xmin)*(ymax-ymin)), zones)
    weighting = map(a->a/sum(areas), areas)
    xmin, xmax, ymin, ymax = sample(zones, Weights(weighting))

    env.target = [(xmin + xmax) * rand(env.rng) - xmin, (ymin + ymax) * rand(env.rng) - ymin]

    env.start_targ_dist = env.d_old = sqeuclidean(_torso_xy(env), env.target)
    getsim(env).mn[:geom_pos][ngeom=:target_geom] = [env.target..., 0]

end

function LyceumMuJoCo.reset!(env::AntMaze)
    env.t = 0
    _movetarget!(env)
    WalkerBase._reset!(env)
end

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntMaze)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.2 <= height <= 1) || env.d_old < FLAGRUN_DIST_THRESH
        done
    end
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AntMaze)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        env.d_old = sqeuclidean(_torso_xy(shapedstate, env), env.target)
        1 - (env.d_old / env.start_targ_dist)
    end
end

LyceumMuJoCo.geteval(env::AntMaze) = sqeuclidean(_torso_xy(env), [0, 16]) < MAZE_DIST_THRESH ? 1 : 0


@inline _torso_xy(env::AntMaze) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntMaze) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntMaze) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntMaze) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
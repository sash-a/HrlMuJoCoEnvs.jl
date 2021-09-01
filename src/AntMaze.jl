mutable struct AntMaze{SIM<:MJSim, S, O} <: WalkerBase.AbstractWalkerMJEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    structure::Matrix{AbstractBlock}
    target::Vector{Number}
    t::Int

    function AntMaze(sim::MJSim; structure=MazeStructure.basic_maze_structure)
        sspace = MultiShape(
            targetvec = VectorShape(Float64, 2),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec = VectorShape(Float64, 2),
            cropped_qpos = VectorShape(Float64, sim.m.nq - 2),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, structure, [0, 0], 0)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntMaze}, n::Integer; structure::Matrix{<:AbstractBlock}=MazeStructure.basic_maze_structure, filename="tmp.xml")
    antmodelpath = joinpath(@__DIR__, "..", "assets", "ant.xml")
    MazeStructure.create_world(antmodelpath, structure=structure, filename=filename)
    modelpath = joinpath(@__DIR__, "..", "assets", filename)

    Tuple(AntMaze(s, structure=structure) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntMaze() = first(tconstruct(AntMaze, 1))
AntMaze(structure::Matrix{<: AbstractBlock}) = first(tconstruct(AntMaze, 1; structure=structure))

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
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end

    # vcat(sin(angle_to_target), cos(angle_to_target), obs)
    obs
end

function LyceumMuJoCo.reset!(env::AntMaze)
    env.t = 0
    env.target = [24 * rand() - 4, 24 * rand() - 4]
    @show env.target
    WalkerBase._reset!(env)
end

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntMaze)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.2 <= height <= 1)
        done
    end
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AntMaze)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        d = Euclidean()(_torso_xy(shapedstate, env), env.target)
        d < 5 ? 1 : 0
    end
end

LyceumMuJoCo.geteval(env::AntMaze) = Euclidean()(_torso_xy(env), [0, 16]) < 5 ? 1 : 0

@inline _torso_xy(env::AntMaze) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntMaze) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntMaze) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntMaze) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
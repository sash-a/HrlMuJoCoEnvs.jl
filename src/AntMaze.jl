using Distributions
using UnsafeArrays
import UnsafeArrays: @uviews
using LightXML

using LyceumBase, LyceumBase.Tools, LyceumMuJoCo, MuJoCo, Shapes

const maze_structure = [1 1 1 1 1 ; 
                        1 2 0 0 1 ; 
                        1 1 1 0 1 ; 
                        1 0 0 0 1 ; 
                        1 1 1 1 1]

abstract type AbstractBlock end
abstract type StaticBlock <: AbstractBlock end
struct Robot <: AbstractBlock end
struct EmptyBlock <: StaticBlock end
struct SingleBlock <: StaticBlock end
struct MoveableBlock <: AbstractBlock # TODO SpinnableBlock
    x::Bool
    y::Bool
    z::Bool
end

height(::AbstractBlock) = 0
height(::SingleBlock) = 1
height(::MoveableBlock) = 1

ismoveable(::Robot) = false
ismoveable(::StaticBlock) = false
ismoveable(::MoveableBlock) = true

ismoveable_x(::AbstractBlock) = false
ismoveable_x(b::MoveableBlock) = b.x
ismoveable_y(::AbstractBlock) = false
ismoveable_y(b::MoveableBlock) = b.y
ismoveable_z(::AbstractBlock) = false
ismoveable_z(b::MoveableBlock) = b.z

mutable struct AntMaze{SIM<:MJSim, S, O} <: AbstractMuJoCoEnvironment
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    structure::Matrix{Int}

    function AntMaze(sim::MJSim)
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 2),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, maze_structure)
        reset!(env)
    end
end

function create_world(modelpath::String, structure::Matrix{Int}=maze_structure, wsize=8)
    torso_x, torso_y = start_torso_xy(structure, wsize)
    height = 0.5

    xdoc = parse_file(modelpath)
    xroot = root(xdoc)
    worldbody = find_element(xroot, "worldbody")

    for i in 1:size(structure, 1)
        for j in 1:size(structure, 2)
            if structure[i, j] == 1
                geom = new_child(worldbody, "geom")
                set_attributes(geom; name        = "block_$(i - 1)_$(j - 1)", 
                                     pos         = "$((j - 1) * wsize - torso_x) $((i - 1) * wsize - torso_y) $(height / 2 * wsize)",
                                     size        = "$(wsize / 2) $(wsize / 2) $(height / 2 * wsize)",
                                     type        = "box",
                                     material    = "",
                                     contype     = "1",
                                     conaffinity = "1",
                                     rgba        = "0.4 0.4 0.4 1")
            end
        end
    end
    save_file(xdoc, "test.xml")
    free(xdoc)
end

function start_torso_xy(structure::Matrix{Int}, wsize)
    # robot position marked with 2
    for i in 1:size(structure, 1)
        for j in 1:size(structure, 2)
            if structure[i, j] == 2
                return (j - 1) * wsize, (i - 1) * wsize
            end
        end
    end

    @assert false "Could not find robot in structure, should be marked with the number 2"
end

function LyceumBase.tconstruct(::Type{AntMaze}, n::Integer)
    # modelpath = joinpath(@__DIR__, "..", "assets", "ant.xml")
    modelpath = joinpath(@__DIR__, "newxml.xml")
    Tuple(AntMaze(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntMaze() = first(tconstruct(AntMaze, 1))

@inline LyceumMuJoCo.getsim(env::AntMaze) = env.sim

@inline LyceumMuJoCo.statespace(env::AntMaze) = env.statespace
function LyceumMuJoCo.getstate!(state, env::AntMaze)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        getstate!(shaped.simstate, env.sim)
        shaped.last_torso_x = env.last_torso_x
    end
    state
end
function LyceumMuJoCo.setstate!(env::AntMaze, state)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        setstate!(env.sim, shaped.simstate)
        env.last_torso_x = LyceumMuJoCo._torso_x(env)
    end
    env
end

@inline LyceumMuJoCo.obsspace(env::AntMaze) = env.obsspace
function LyceumMuJoCo.getobs!(obs, env::AntMaze)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
        # missing contact forces
    end
    obs
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AntMaze)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        alive_bonus = 1.0
        reward = (LyceumMuJoCo._torso_x(env) - env.last_torso_x) / timestep(env)
        reward += alive_bonus
        reward -= 5e-4 * sum(x->x^2, action)
        reward
    end
end

function LyceumMuJoCo.geteval(state, ::Any, ::Any, env::AntMaze)
    checkaxes(statespace(env), state)
    @uviews state begin
        LyceumMuJoCo._torso_x(env)
    end
end

function LyceumMuJoCo.reset!(env::AntMaze)
    reset!(env.sim)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    env
end

function LyceumMuJoCo.step!(env::AntMaze)
    env.last_torso_x = LyceumMuJoCo._torso_x(env)
    step!(env.sim)
    env
end

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntMaze)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = _torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.2 >= height >= 1)
        done
    end
end

# @inline LyceumMuJoCo._torso_x(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[1]
@inline LyceumMuJoCo._torso_x(env::AntMaze) = env.sim.d.qpos[1]

@inline _torso_xy(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[1:2]
@inline _torso_xy(env::AntMaze) = env.sim.d.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[3]
# TODO get the quat and project on xy
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[4]
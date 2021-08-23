include("MazeStructure.jl")
using .MazeStructure

mutable struct AntMaze{SIM<:MJSim, S, O} <: WalkerBase
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    structure::Matrix{AbstractBlock}

    function AntMaze(sim::MJSim; structure=MazeStructure.basic_maze_structure)
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 2),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, structure)
        reset!(env)
    end
end

function create_world(modelpath::String; structure::Matrix{<: AbstractBlock}=MazeStructure.basic_maze_structure, wsize=8, filename="tmp.xml")
    torso_x, torso_y = start_torso_xy(structure, wsize)

    xdoc = parse_file(modelpath)
    xroot = root(xdoc)
    worldbody = find_element(xroot, "worldbody")

    for i in 1:size(structure, 1)
        for j in 1:size(structure, 2)
            block = structure[i, j]
            if height(block) > 0
                geom = new_child(worldbody, "geom")
                set_attributes(geom; name        = "block_$(i - 1)_$(j - 1)", 
                                     pos         = "$((j - 1) * wsize - torso_x) $((i - 1) * wsize - torso_y) $(height(block) / 2 * wsize)",
                                     size        = "$(wsize / 2) $(wsize / 2) $(height(block) / 2 * wsize)",
                                     type        = "box",
                                     material    = "",
                                     contype     = "1",
                                     conaffinity = "1",
                                     rgba        = "0.4 0.4 0.4 1")
            end
        end
    end

    outfile = joinpath(@__DIR__, "..", "assets", filename)
    save_file(xdoc, outfile)
    free(xdoc)
end

function start_torso_xy(structure::Matrix{<: AbstractBlock}, wsize)
    for i in 1:size(structure, 1)
        for j in 1:size(structure, 2)
            if isrobot(structure[i, j])
                return (j - 1) * wsize, (i - 1) * wsize
            end
        end
    end

    @assert false "Could not find robot in structure"
end

function LyceumBase.tconstruct(::Type{AntMaze}, n::Integer; structure::Matrix{<: AbstractBlock}=MazeStructure.basic_maze_structure, filename="tmp.xml")
    antmodelpath = joinpath(@__DIR__, "..", "assets", "ant.xml")
    create_world(antmodelpath, structure=structure, filename=filename)
    modelpath = joinpath(@__DIR__, "..", "assets", filename)
    Tuple(AntMaze(s, structure=structure) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntMaze() = first(tconstruct(AntMaze, 1))
AntMaze(structure::Matrix{<: AbstractBlock}) = first(tconstruct(AntMaze, 1; structure=structure))

function LyceumMuJoCo.getobs!(obs, env::T) where T <: WalkerBase
    # TODO sensor readings
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end
    obs
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

@inline q_inv(a::Vector) = [a[1], -a[2], -a[3], -a[4]]
@inline function q_mult(a, b) # multiply two quaternion
    w = a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4]
    i = a[1]*b[2] + a[2]*b[1] + a[3]*b[4] - a[4]*b[3]
    j = a[1]*b[3] - a[2]*b[4] + a[3]*b[1] + a[4]*b[2]
    k = a[1]*b[4] + a[2]*b[3] - a[3]*b[2] + a[4]*b[1]
    
    [w, i, j, k]
end

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntMaze) = shapedstate.simstate.qpos[3]
@inline function LyceumMuJoCo._torso_ang(env::AntMaze)
    ori_ind = 4
    env.sim.d.qpos[ori_ind:ori_ind + 3]
    ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
    ori = atan(ori[1], ori[0])
    return ori
end
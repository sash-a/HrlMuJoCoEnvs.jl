module MazeStructure

using LightXML

export Robot, AbstractBlock, StaticBlock, EmptyBlock, SingleBlock, MoveableBlock, height, ismoveable, basic_maze_structure, isrobot

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
height(::SingleBlock) = 0.5
height(::MoveableBlock) = 0.5

isrobot(::AbstractBlock) = false
isrobot(::Robot) = true

ismoveable(::AbstractBlock) = false
ismoveable(::MoveableBlock) = true

ismoveable_x(::AbstractBlock) = false
ismoveable_x(b::MoveableBlock) = b.x
ismoveable_y(::AbstractBlock) = false
ismoveable_y(b::MoveableBlock) = b.y
ismoveable_z(::AbstractBlock) = false
ismoveable_z(b::MoveableBlock) = b.z


const _X = SingleBlock()
const _O = EmptyBlock()
const _R = Robot()

const basic_maze_structure = [_X _X _X _X _X;
                              _X _R _O _O _X;
                              _X _X _X _O _X;
                              _X _O _O _O _X;
                              _X _X _X _X _X]

const wall_structure = [_X _X _X _X _X;
                        _X _O _O _O _X;
                        _X _O _R _O _X;
                        _X _O _O _O _X;
                        _X _X _X _X _X]

function create_world(modelpath::String; structure::Matrix{<:AbstractBlock}=MazeStructure.wall_structure, wsize=8, filename="tmp.xml")
    xdoc = parse_file(modelpath)
    xdoc = _create_maze(xdoc, structure, wsize)
    finish_xml(xdoc, filename)
end

function create_world(modelpath::String, napples::Int, nbombs::Int, nbins::Int; structure::Matrix{<:AbstractBlock}=MazeStructure.wall_structure, wsize=8, viz=false, filename="tmp.xml")
    xdoc = parse_file(modelpath)
    if viz
        xdoc = _create_collectibles(xdoc, napples, nbombs, nbins)
    end
    xdoc = _create_maze(xdoc, structure, wsize)
    finish_xml(xdoc, filename)
end

function _create_collectibles(xdoc, napples::Int, nbombs::Int, nbins::Int)
    xroot = root(xdoc)
    worldbody = find_element(xroot, "worldbody")

    for i in 1:napples
        apple = new_child(worldbody, "geom")
        set_attributes(apple; name="apple_$i",
                                pos="-1 -1 0",
                                size="0.25",
                                type="sphere",
                                material="",
                                contype="2",
                                conaffinity="2",
                                condim="6",
                                rgba="0.4 1 0.4 1")
    end
    for j in 1:nbombs
        apple = new_child(worldbody, "geom")
        set_attributes(apple; name="bomb_$j",
                                pos="-2 -2 0",
                                size="0.25",
                                type="sphere",
                                material="",
                                contype="2",
                                conaffinity="2",
                                condim="6",
                                rgba="1 0.4 0.4 1")
    end

    for i in 1:nbins
        apple_sensor = new_child(worldbody, "geom")
        set_attributes(apple_sensor; name="apple_sensor_$i",
                        pos="0 0 -2",
                        size="0.25",
                        type="sphere",
                        material="",
                        contype="2",
                        conaffinity="2",
                        condim="6",
                        rgba="0.2 1 0.8 1")

        bomb_sensor = new_child(worldbody, "geom")
        set_attributes(bomb_sensor; name="bomb_sensor_$i",
                        pos="0 0 -2",
                        size="0.25",
                        type="sphere",
                        material="",
                        contype="2",
                        conaffinity="2",
                        condim="6",
                        rgba="1 0.8 0.2 1")
    end

    xdoc
end

function _create_maze(xdoc, structure::Matrix{<:AbstractBlock}, wsize)
    torso_x, torso_y = start_torso_xy(structure, wsize)

    xroot = root(xdoc)
    worldbody = find_element(xroot, "worldbody")

    for i in 1:size(structure, 1)
        for j in 1:size(structure, 2)
            block = structure[i, j]
            if height(block) > 0
                geom = new_child(worldbody, "geom")
                set_attributes(geom; name="block_$(i - 1)_$(j - 1)", 
                                        pos="$((j - 1) * wsize - torso_x) $((i - 1) * wsize - torso_y) $(height(block) / 2 * wsize)",
                                        size="$(wsize / 2) $(wsize / 2) $(height(block) / 2 * wsize)",
                                        type="box",
                                        material="",
                                        contype="1",
                                        conaffinity="1",
                                        rgba="0.4 0.4 0.4 1")
            end
        end
    end

    xdoc
end

function finish_xml(xdoc, outfile::String)
    outfile = joinpath(@__DIR__, "..", "assets", outfile)
    save_file(xdoc, outfile)
    free(xdoc)
end

function start_torso_xy(structure::Matrix{<:AbstractBlock}, wsize)
    for i in 1:size(structure, 1)
        for j in 1:size(structure, 2)
            if isrobot(structure[i, j])
                return (j - 1) * wsize, (i - 1) * wsize
            end
        end
    end

    @assert false "Could not find robot in structure"
end
                        
end  # module
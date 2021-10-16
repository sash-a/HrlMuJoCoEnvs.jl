module WorldStructure

import ..LightXML
import ..AssetManager
export Robot, AbstractBlock, StaticBlock, EmptyBlock, SingleBlock, MoveableBlock, height, ismoveable, basic_maze_structure, isrobot

const MAZE_HEIGHT = 0.5

abstract type AbstractBlock end
abstract type StaticBlock <: AbstractBlock end

struct Robot <: AbstractBlock end
struct EmptyBlock <: StaticBlock end
struct SingleBlock <: StaticBlock end
struct ChasmBlock <: StaticBlock end
struct MoveableBlock <: AbstractBlock # TODO SpinnableBlock
    x::Bool
    y::Bool
    z::Bool
end

height(::AbstractBlock) = 0
height(::SingleBlock) = MAZE_HEIGHT
height(::MoveableBlock) = MAZE_HEIGHT
height(::ChasmBlock) = -MAZE_HEIGHT

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


const _B = SingleBlock()
const _M = MoveableBlock(true, true, false)
const _X = MoveableBlock(true, false, false)
const _F = MoveableBlock(false, true, true)
const _C = ChasmBlock()
const _E = EmptyBlock()
const _R = Robot()

const basic_maze_structure = [_B _B _B _B _B;
                              _B _R _E _E _B;
                              _B _B _B _E _B;
                              _B _E _E _E _B;
                              _B _B _B _B _B]

const wall_structure = [_B _B _B _B _B;
                        _B _E _E _E _B;
                        _B _E _R _E _B;
                        _B _E _E _E _B;
                        _B _B _B _B _B]

const push_maze =  [_B _B _B _B _B;
                    _B _E _R _B _B;
                    _B _E _M _E _B;
                    _B _B _E _B _B;
                    _B _B _B _B _B]

const ez_push_maze =  [_B _B _B _B _B;
                       _B _E _R _B _B;
                       _B _E _X _E _B;
                       _B _B _E _B _B;
                       _B _B _B _B _B]

const fall_maze =  [_B _B _B _B;
                    _B _R _E _B;
                    _B _E _F _B;
                    _B _C _C _B;
                    _B _E _E _B;
                    _B _B _B _B]

function create_world(modelpath::String; structure::Matrix{<:AbstractBlock}=WorldStructure.wall_structure, wsize=8, filename="tmp.xml")
    xdoc = LightXML.parse_file(modelpath)
    xdoc = _create_maze(xdoc, structure, wsize)
    finish_xml(xdoc, filename)
end

function create_world(modelpath::String, napples::Int, nbombs::Int, nbins::Int; structure::Matrix{<:AbstractBlock}=WorldStructure.wall_structure, wsize=8, viz=false, filename="tmp.xml")
    xdoc = LightXML.parse_file(modelpath)
    if viz
        xdoc = _create_collectibles(xdoc, napples, nbombs, nbins)
    end
    xdoc = _create_maze(xdoc, structure, wsize)
    finish_xml(xdoc, filename)
end

function _create_collectibles(xdoc, napples::Int, nbombs::Int, nbins::Int)
    xroot = LightXML.root(xdoc)
    worldbody = LightXML.find_element(xroot, "worldbody")

    for i in 1:napples
        apple = LightXML.new_child(worldbody, "geom")
        LightXML.set_attributes(apple; name="apple_$i",
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
        apple = LightXML.new_child(worldbody, "geom")
        LightXML.set_attributes(apple; name="bomb_$j",
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
        apple_sensor = LightXML.new_child(worldbody, "geom")
        LightXML.set_attributes(apple_sensor; name="apple_sensor_$i",
                        pos="0 0 -2",
                        size="0.25",
                        type="sphere",
                        material="",
                        contype="2",
                        conaffinity="2",
                        condim="6",
                        rgba="0.2 1 0.8 1")

        bomb_sensor = LightXML.new_child(worldbody, "geom")
        LightXML.set_attributes(bomb_sensor; name="bomb_sensor_$i",
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

    xroot = LightXML.root(xdoc)
    worldbody = LightXML.find_element(xroot, "worldbody")

    heightoffset = 0.  # used for ant fall
    elevated = any(i -> height(i) == -MAZE_HEIGHT, structure)
    if elevated  # Increase initial z-pos of ant.
        heightoffset = MAZE_HEIGHT * wsize
        for n in LightXML.child_elements(worldbody)
            if LightXML.attribute(n, "name") == "torso"
                LightXML.set_attributes(n; pos="0 0 $(0.75 + heightoffset)")
                break
            end
        end
    end
    # If there are movable blocks, change simulation settings to perform better contact detection.
    if any(i -> i isa MoveableBlock,  structure)
        default = LightXML.find_element(xroot, "default")
        defaultgeom = LightXML.find_element(default, "geom")
        LightXML.set_attributes(defaultgeom; solimp=".995 .995 .01")
    end

    for i in 1:size(structure, 1)
        for j in 1:size(structure, 2)
            block = structure[i, j]
            if elevated && height(block) != -MAZE_HEIGHT
                # Create elevated platform.
                geom = LightXML.new_child(worldbody, "geom")
                LightXML.set_attributes(geom; name="elevated_$(i - 1)_$(j - 1)", 
                        pos="$((j - 1) * wsize - torso_x) $((i - 1) * wsize - torso_y) $(MAZE_HEIGHT / 2 * wsize)",
                        size="$(wsize / 2) $(wsize / 2) $(MAZE_HEIGHT / 2 * wsize)",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.9 0.9 1")
            end
            if height(block) > 0
                if !ismoveable(block)
                    geom = LightXML.new_child(worldbody, "geom")
                    LightXML.set_attributes(geom; name="block_$(i - 1)_$(j - 1)", 
                            pos="$((j - 1) * wsize - torso_x) $((i - 1) * wsize - torso_y) $(heightoffset + height(block) / 2 * wsize)",
                            size="$(wsize / 2) $(wsize / 2) $(height(block) / 2 * wsize)",
                            type="box",
                            material="",
                            contype="1",
                            conaffinity="1",
                            rgba="0.4 0.4 0.4 1")
                else
                    shrink = ismoveable_z(block) ? 0.99 : 1.

                    body = LightXML.new_child(worldbody, "body")
                    LightXML.set_attributes(body; 
                        name="moveable_$(i - 1)_$(j - 1)",
                        pos="$((j - 1) * wsize - torso_x) $((i - 1) * wsize - torso_y) $(heightoffset + height(block) / 2 * wsize)")

                    geom = LightXML.new_child(body, "geom")
                    LightXML.set_attributes(geom; 
                        name="block_$(i - 1)_$(j - 1)",
                        pos="0 0 0",
                        size="$(wsize / 2 * shrink) $(wsize / 2 * shrink) $(height(block) / 2 * wsize)",
                        type="box",
                        material="",
                        mass=ismoveable_z(block) ? "0.001" : "0.0002",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.1 0.1 1"
                    )

                    limited = ismoveable_z(block)
                    if ismoveable_x(block)
                        _addjoint(body, "movable_x_$(i - 1)_$(j - 1)", wsize, limited, 1, 0, 0)
                    end
                    if ismoveable_y(block)
                        _addjoint(body, "movable_y_$(i - 1)_$(j - 1)", wsize, limited, 0, 1, 0)
                    end
                    if ismoveable_z(block)
                        _addjoint(body, "movable_z_$(i - 1)_$(j - 1)", wsize, limited, 0, 0, 1; heightoffset=heightoffset)
                    end
                end
            end
        end
    end

    xdoc
end

function _addjoint(body, name, wsize, limited, axis_x, axis_y, axis_z; heightoffset=0)
    range = "$(-wsize) $wsize"
    if axis_z != 0 && heightoffset != 0
        range = "$(-heightoffset) 0"
    end

    joint = LightXML.new_child(body, "joint")
    LightXML.set_attributes(joint; 
                            name=name,
                            armature="0",
                            axis="$axis_x $axis_y $axis_z",
                            damping="0.0",
                            limited=string(limited),
                            range=range,
                            margin="0.01",
                            pos="0 0 0",
    type="slide")
end

function finish_xml(xdoc, outfile::String)
    outfile = joinpath(AssetManager.dir, outfile)
    LightXML.save_file(xdoc, outfile)
    LightXML.free(xdoc)
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
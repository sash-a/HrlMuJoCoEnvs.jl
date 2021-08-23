module MazeStructure

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

ismoveable(::Robot) = false
ismoveable(::StaticBlock) = false
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

end
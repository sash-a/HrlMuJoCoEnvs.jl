abstract type AbstractRobot end

getfile(::AbstractRobot) = nothing
getfile(::Type{AbstractRobot}) = nothing
# obsspace(::AbstractRobot) = nothing
# getobs!(::Any, ::AbstractRobot) = []
controlcost(::AbstractRobot) = 0
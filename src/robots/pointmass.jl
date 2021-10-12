mutable struct PointMass{O} <: AbstractRobot
    obsspace::O

    function PointMass(::MJSim=nothing)  # doesn't need sim but needs to keep consistent with other envs
        ospace = MultiShape(
            agent_pos=VectorShape(Float64, 2),
            agent_vel=VectorShape(Float64, 2),
        )
        new{typeof(ospace)}(ospace)
    end
end

getfile(::PointMass) = joinpath(AssetManager.dir, "pointmass.xml")
getfile(::Type{PointMass}) = joinpath(AssetManager.dir, "pointmass.xml")

@inline LyceumMuJoCo.obsspace(rob::PointMass) = rob.obsspace
function LyceumMuJoCo.getobs!(sim::MJSim, rob::PointMass, obs)
    dn = sim.dn
    shaped = obsspace(rob)(obs)

    @uviews shaped @inbounds begin
        shaped.agent_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent]
        shaped.agent_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
    end

    obs
end

# not sure if I should leave in state, obs and action args here
LyceumMuJoCo.isdone(sim::MJSim, ::PointMass) = false

controlcost(::PointMass) = 0

@inline _torso_xy(sim::MJSim, ::PointMass) = [sim.dn.xpos[:x, :agent], sim.dn.xpos[:y, :agent]]
@inline _torso_xy(shapedstate::ShapedView, ::PointMass) = @assert false

@inline LyceumMuJoCo._torso_height(::ShapedView, ::PointMass) = 0.
@inline LyceumMuJoCo._torso_height(::MJSim, ::PointMass) = 0.

@inline LyceumMuJoCo._torso_ang(::MJSim, ::PointMass) = 0.
@inline LyceumMuJoCo._torso_ang(::ShapedView, ::PointMass) = 0.
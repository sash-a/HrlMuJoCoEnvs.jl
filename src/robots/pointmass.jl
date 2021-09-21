mutable struct PointMass{SIM<:MJSim, O} <: AbstractRobot
    sim::SIM
    obsspace::O

    function PointMass(sim::MJSim)
        ospace = MultiShape(
            agent_pos=VectorShape(Float64, 2),
            agent_vel=VectorShape(Float64, 2),
        )
        new{typeof(sim), typeof(ospace)}(sim, ospace)
    end
end

getfile(::PointMass) = joinpath(AssetManager.dir, "pointmass.xml")
getfile(::Type{PointMass}) = joinpath(AssetManager.dir, "pointmass.xml")

@inline LyceumMuJoCo.obsspace(rob::PointMass) = rob.obsspace
function LyceumMuJoCo.getobs(rob::PointMass)
    # checkaxes(obsspace(env), obs)
    dn = rob.sim.dn
    obs = allocate(obsspace(rob))
    shaped = obsspace(rob)(obs)

    @uviews shaped @inbounds begin
        shaped.agent_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent]
        shaped.agent_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
    end

    obs
end

# not sure if I should leave in state, obs and action args here
LyceumMuJoCo.isdone(::PointMass) = false

controlcost(::PointMass) = 0

@inline _torso_xy(rob::PointMass) = [rob.sim.dn.xpos[:x, :agent], rob.sim.dn.xpos[:y, :agent]]
@inline _torso_xy(shapedstate::ShapedView, rob::PointMass) = _torso_xy(rob)

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::PointMass) = 0.
@inline LyceumMuJoCo._torso_height(::PointMass) = 0.

@inline LyceumMuJoCo._torso_ang(::PointMass) = 0.
@inline LyceumMuJoCo._torso_ang(::ShapedView, ::PointMass) = 0.
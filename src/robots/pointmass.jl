mutable struct PointMass{O} <: AbstractRobot
    obsspace::O

    function PointMass()
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
function LyceumMuJoCo.getobs!(sim::MJSim, rob::PointMass)
    # checkaxes(obsspace(env), obs)
    qpos = sim.d.qpos
    dn = sim.dn

    obs = allocate(obsspace(rob))
    
    @views @uviews qpos obs begin
        shaped = obsspace(rob)(obs)
        copyto!(shaped.agent_pos, [dn.xpos[:x, :agent], dn.xpos[:y, :agent]])
        copyto!(shaped.agent_vel, [dn.qvel[:agent_x], dn.qvel[:agent_y]])
    end

    obs
end

LyceumMuJoCo.isdone(::Any, ::Any, ::Any, ::PointMass) = false

controlcost(::PointMass) = 0

@inline _torso_xy(env::PointMass) = [env.sim.dn.xpos[:x, :agent], env.sim.dn.xpos[:y, :agent]]
@inline _torso_xy(shapedstate::ShapedView, ::PointMass) = shapedstate.agent_pos

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::PointMass) = 0.
@inline LyceumMuJoCo._torso_height(env::PointMass) = 0.

@inline LyceumMuJoCo._torso_ang(env::PointMass) = 0.
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::PointMass) = 0.
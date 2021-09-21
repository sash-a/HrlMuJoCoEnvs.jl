mutable struct Ant{SIM <: MJSim,O} <: AbstractRobot
    sim::SIM
    obsspace::O

    cropqpos::Bool

    function Ant(sim::MJSim; cropqpos=false)
        ospace = MultiShape(
            qpos=VectorShape(Float64, sim.m.nq - (cropqpos ? 2 : 0)),
            qvel=VectorShape(Float64, sim.m.nv),
        )
        new{typeof(sim),typeof(ospace)}(sim, ospace, cropqpos)
    end
end

getfile(::Type{Ant}) = joinpath(AssetManager.dir, "ant.xml")
getfile(a::Ant) = getfile(typeof(a))

@inline LyceumMuJoCo.obsspace(rob::Ant) = rob.obsspace
function LyceumMuJoCo.getobs(rob::Ant)
    qpos = rob.sim.d.qpos
    obs = allocate(obsspace(rob))
    
    @views @uviews qpos obs begin
        shaped = obsspace(rob)(obs)
        copyto!(shaped.qpos, rob.cropqpos ? qpos[3:end] : qpos)
        copyto!(shaped.qvel, rob.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end
    obs
end

LyceumMuJoCo.isdone(rob::Ant) = !(0.2 <= LyceumMuJoCo._torso_height(rob) <= 1)
controlcost(::Ant) = 5e-4

@inline _torso_xy(rob::Ant) = rob.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::Ant) = shapedstate.simstate.qpos[1:2]
@inline LyceumMuJoCo._torso_height(rob::Ant) = rob.sim.d.qpos[3]
@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::Ant) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_ang(env::Ant) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::Ant) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])

const ori_ind = 4
@inline function torso_ori(rot::Vector{Float64})
    ori = [0, 1, 0, 0]
    ori = q_mult(q_mult(rot, ori), q_inv(rot))[2:3]  # project onto x-y plane
    ori = atan(ori[2], ori[1])
    return ori
end

@inline q_inv(a::Vector) = [a[1], -a[2], -a[3], -a[4]]
@inline function q_mult(a, b) # multiply two quaternions
    w = a[1] * b[1] - a[2] * b[2] - a[3] * b[3] - a[4] * b[4]
    i = a[1] * b[2] + a[2] * b[1] + a[3] * b[4] - a[4] * b[3]
    j = a[1] * b[3] - a[2] * b[4] + a[3] * b[1] + a[4] * b[2]
    k = a[1] * b[4] + a[2] * b[3] - a[3] * b[2] + a[4] * b[1]
    
    [w, i, j, k]
end

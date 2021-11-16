mutable struct Ant{SIM<:MJSim, S, O} <: WalkerBase.AbstractWalkerMJEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    randreset_distribution::Uniform{Float64}

    function Ant(sim::MJSim)
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 2),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, Uniform(-0.1, 0.1))
        reset!(env)
    end
end

AntV3() = first(tconstruct(Ant, 1, joinpath(AssetManager.dir, "ant.xml")))
AntV1() = first(tconstruct(Ant, 1, joinpath(AssetManager.dir, "easier_ant_geared.xml"), skip=5))

function LyceumMuJoCo.getobs!(obs, env::Ant)
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

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::Ant)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.2 <= height <= 1)
        done
    end
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::Ant) 
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        alive_bonus = 1.0
        reward = (LyceumMuJoCo._torso_x(shapedstate, env) - shapedstate.last_torso_x) / timestep(env)
        reward += alive_bonus
        reward -= 5e-4 * sum(x->x^2, action)
        reward
    end
end

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::Ant) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::Ant) = env.sim.d.qpos[3]
@inline _torso_xy(env::Ant) = env.sim.d.qpos[1:2]


const ori_ind = 4
@inline function torso_ori(rot::Vector{Float64})
    ori = [0, 1, 0, 0]
    ori = q_mult(q_mult(rot, ori), q_inv(rot))[2:3]  # project onto x-y plane
    ori = atan(ori[2], ori[1])
    return ori
end

@inline q_inv(a::Vector) = [a[1], -a[2], -a[3], -a[4]]
@inline function q_mult(a, b) # multiply two quaternions
    w = a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4]
    i = a[1]*b[2] + a[2]*b[1] + a[3]*b[4] - a[4]*b[3]
    j = a[1]*b[3] - a[2]*b[4] + a[3]*b[1] + a[4]*b[2]
    k = a[1]*b[4] + a[2]*b[3] - a[3]*b[2] + a[4]*b[1]
    
    [w, i, j, k]
end
@inline LyceumMuJoCo._torso_ang(env::Ant) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::Ant) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
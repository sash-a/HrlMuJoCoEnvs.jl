mutable struct Flagrun{SIM <: MJSim,S,O} <: WalkerBase.AbstractWalkerMJEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    structure::Matrix{AbstractBlock}
    target::Vector{Number}
    evalrew::Float64
    t::Int
    interval::Int
    d_old::Float64
    rng::MersenneTwister

    function Flagrun(sim::MJSim; structure=MazeStructure.wall_structure, interval=100, rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            cropped_qpos=VectorShape(Float64, sim.m.nq - 2),
            qvel=VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim),typeof(sspace),typeof(ospace)}(sim, sspace, ospace, 0, structure, [0, 0], 0, 0, interval, 0, rng)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{Flagrun}, assetfile::String, n::Integer; interval=100, seed=123, outfile="tmp.xml")
    antmodelpath = joinpath(@__DIR__, "..", "assets", assetfile)
    MazeStructure.create_world(antmodelpath, structure=MazeStructure.wall_structure, filename=outfile)
    modelpath = joinpath(@__DIR__, "..", "assets", outfile)

    Tuple(Flagrun(s, structure=MazeStructure.wall_structure, interval=interval, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntFlagrun(;interval=100, seed=123) = first(tconstruct(Flagrun, "ant.xml", 1; interval=interval, seed=seed))

function LyceumMuJoCo.step!(env::Flagrun)
    env.evalrew -= env.d_old
    
    env.t += 1
    if env.t % env.interval == 0
        env.target = _createtarget(env)
        env.d_old = Euclidean()(_torso_xy(env), env.target)
    end

    WalkerBase._step!(env)
end

function LyceumMuJoCo.getobs!(obs, env::Flagrun)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        targetvec = env.target - _torso_xy(env)
        angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)

        copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end

    obs
end

function _createtarget(env::Flagrun)
    offset = rand(env.rng, Uniform(-1, 1), 2)
    return offset * 5 + _torso_xy(env)
    # dist = Euclidean()
    # targ = [20 * rand(env.rng) - 10, 20 * rand(env.rng) - 10]
    # while dist(_torso_xy(env), targ) < 1
    #     targ = [20 * rand(env.rng) - 10, 20 * rand(env.rng) - 10]
    # end
    # targ
end

function LyceumMuJoCo.reset!(env::Flagrun)
    env.t = 0
    env.target = _createtarget(env)
    env.d_old = Euclidean()(_torso_xy(env), env.target)
    env.evalrew = 0
    WalkerBase._reset!(env)
end

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::Flagrun)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.38 <= height <= 1)
        done
    end
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::Flagrun)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        d_new = Euclidean()(_torso_xy(shapedstate, env), env.target)
        r = (d_new - env.d_old) / timestep(env)
        if d_new < 1
            r += 5000
            env.target = _createtarget(env)  # create new target if close to this one
            env.d_old = Euclidean()(_torso_xy(shapedstate, env), env.target)  # recalc target dist
        end
        env.d_old = d_new
        r
    end
end

LyceumMuJoCo.geteval(env::Flagrun) = env.evalrew

@inline _torso_xy(env::Flagrun) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::Flagrun) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::Flagrun) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::Flagrun) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::Flagrun) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::Flagrun) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
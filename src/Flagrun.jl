const FLAGRUN_DIST_THRESH = 1

mutable struct Flagrun{SIM <: MJSim,S,O} <: WalkerBase.AbstractWalkerMJEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    randreset_distribution::Uniform{Float64}
    structure::Matrix{AbstractBlock}
    target::Vector{Number}
    evalrew::Float64
    t::Int
    interval::Int
    d_old::Float64
    rew_once::Bool
    rng::MersenneTwister

    function Flagrun(sim::MJSim; structure=MazeStructure.wall_structure, interval=100, rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            cropped_qpos=VectorShape(Float64, sim.m.nq - 2),
            qvel=VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim),typeof(sspace),typeof(ospace)}(
            sim, sspace, ospace, 0, Uniform(-0.1, 0.1), structure, [0, 0], 0f0, 0, interval, 0, true, rng)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{Flagrun}, assetfile::String, n::Integer; interval=100, seed=nothing, outfile="tmp.xml")
    antmodelpath = joinpath(@__DIR__, "..", "assets", assetfile)
    MazeStructure.create_world(antmodelpath, structure=MazeStructure.wall_structure, filename=outfile)
    modelpath = joinpath(@__DIR__, "..", "assets", outfile)

    Tuple(Flagrun(s, structure=MazeStructure.wall_structure, interval=interval, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntFlagrun(;interval=100, seed=nothing) = first(tconstruct(Flagrun, "ant.xml", 1; interval=interval, seed=seed))

function LyceumMuJoCo.step!(env::Flagrun)
    env.evalrew -= env.d_old
    
    env.t += 1
    if env.t % env.interval == 0
        env.rew_once = true
        _movetarget!(env)
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
        # copyto!(shaped.targetvec, normalize(targetvec))
        copyto!(shaped.d_old, [env.d_old / 1000])
        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end

    obs
end

function _movetarget!(env::Flagrun, pos::Vector{T}) where T
    offset = rand(env.rng, Uniform(-1, 1), 2)
    targ = offset * 5 + pos
    
    # targ = [20 * rand(env.rng) - 10, 20 * rand(env.rng) - 10]
    while sqeuclidean(pos, targ) < FLAGRUN_DIST_THRESH
        # targ = [20 * rand(env.rng) - 10, 20 * rand(env.rng) - 10]

        offset = rand(env.rng, Uniform(-1, 1), 2)
        targ = offset * 5 + pos
    end

    # bit of a hack: moving the extra geom in the xml to indicate target position
    getsim(env).mn[:geom_pos][ngeom=:target_geom] = [targ..., 0]
    env.target = targ
    env.d_old = sqeuclidean(pos, env.target)
    # println("moved target: $(_torso_xy(env)) - $(targ) - $(sqeuclidean(pt, targ))")
    targ
end

_movetarget!(env::Flagrun) = _movetarget!(env, _torso_xy(env))

function LyceumMuJoCo.reset!(env::Flagrun)
    env.t = 0
    env.evalrew = 0
    env.rew_once = true
    r = WalkerBase._reset!(env)
    _movetarget!(env, [0,0])
    r
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
    rew = @uviews state begin
        shapedstate = statespace(env)(state)
        d_new = sqeuclidean(_torso_xy(shapedstate, env), env.target)  # this should possibly go in step!
        r = (d_new - env.d_old) / timestep(env)
        if d_new < FLAGRUN_DIST_THRESH && env.rew_once
            r += 5000
            env.rew_once = false
            # _movetarget!(env)  # create new target if close to this one
        end
        env.d_old = d_new
        r
    end
    rew
    # LyceumMuJoCo.isdone(env) ? -100000 : rew  # penalizing for dying
end

LyceumMuJoCo.geteval(env::Flagrun) = env.evalrew

@inline _torso_xy(env::Flagrun) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::Flagrun) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::Flagrun) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::Flagrun) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::Flagrun) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::Flagrun) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
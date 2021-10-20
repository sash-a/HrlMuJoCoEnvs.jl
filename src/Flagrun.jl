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
    cropqpos::Bool
    d_old::Float64
    targ_start_dist::Float64
    rew_once::Bool
    rng::MersenneTwister

    function Flagrun(sim::MJSim; structure=WorldStructure.wall_structure, interval=25, cropqpos=true, rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            cropped_qpos=VectorShape(Float64, sim.m.nq - (cropqpos ? 2 : 0)),
            qvel=VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim),typeof(sspace),typeof(ospace)}(
            sim, sspace, ospace, 0, Uniform(-0.1, 0.1), structure, 
            [0, 0], 0f0, 0, interval, cropqpos, 0, 0, true, rng
        )
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{Flagrun}, assetfile::String, n::Integer; interval=25, cropqpos=true, seed=nothing, outfile="antflagruntmp.xml")
    antmodelpath = joinpath(AssetManager.dir, assetfile)
    WorldStructure.create_world(antmodelpath, structure=WorldStructure.wall_structure, filename=outfile)
    modelpath = joinpath(AssetManager.dir, outfile)

    Tuple(Flagrun(s; structure=WorldStructure.wall_structure, interval=interval, cropqpos=cropqpos, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

AntFlagrun(;interval=25, cropqpos=true, seed=nothing) = first(tconstruct(Flagrun, "easier_ant.xml", 1; interval=interval, cropqpos=cropqpos, seed=seed))

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
        copyto!(shaped.cropped_qpos, qpos[(env.cropqpos ? 3 : 1):end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end

    obs
end

function _movetarget!(env::Flagrun, pos::Vector{T}) where T
    offset = rand(env.rng, Uniform(-1, 1), 2)
    targ = offset * 4 + pos
    
    while euclidean(pos, targ) < FLAGRUN_DIST_THRESH
        offset = rand(env.rng, Uniform(-1, 1), 2)
        targ = offset * 4 + pos
    end

    # bit of a hack: moving the extra geom in the xml to indicate target position
    getsim(env).mn[:geom_pos][ngeom=:target_geom] = [targ..., 0]
    env.target = targ
    env.d_old = euclidean(pos, env.target)
    env.targ_start_dist = env.d_old
    # println("moved target: $(_torso_xy(env)) - $(targ) - $(euclidean(pt, targ))")
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
        done = !(all(isfinite, state) && 0.2 <= height <= 1)
        done
    end
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::Flagrun)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    rew = @uviews state begin
        shapedstate = statespace(env)(state)
        d_new = euclidean(_torso_xy(shapedstate, env), env.target)  # this should possibly go in step!
        r = 1 - d_new / env.targ_start_dist + (d_new < FLAGRUN_DIST_THRESH ? 1 : 0)
        env.d_old = d_new
        r
    end
    rew
end

LyceumMuJoCo.geteval(env::Flagrun) = env.evalrew

@inline _torso_xy(env::Flagrun) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::Flagrun) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::Flagrun) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::Flagrun) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::Flagrun) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::Flagrun) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
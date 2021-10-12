FLAGRUN_DIST_THRESH = 1

mutable struct FlagrunEnv{SIM<:MJSim, S, O} <: AbstractWalker3DMJEnv
    sim::SIM
    robot::AbstractRobot
    statespace::S
    obsspace::O
    randreset_distribution::Uniform{Float64}
    target::Vector{Number}
    evalrew::Float64
    t::Int
    interval::Int
    d_old::Float64
    targ_start_dist::Float64
    rew_once::Bool
    rng::MersenneTwister

    function FlagrunEnv(sim::MJSim, robot; interval=50, rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            simstate=statespace(sim),
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            robobs=obsspace(robot), 
        )
        env = new{typeof(sim),typeof(sspace),typeof(ospace)}(
            sim, robot, sspace, ospace, Uniform(-0.1, 0.1), [0, 0], 0f0, 0, interval, 0, 0, true, rng)        
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{FlagrunEnv}, Rob::Type{<:AbstractRobot}, n::Integer; interval=50,
                                structure::Matrix{<:AbstractBlock}=WorldStructure.basic_maze_structure, seed=nothing)
    robmodelpath = getfile(Rob)
    outfile = "flagruntmp.xml"
    modelpath = joinpath(AssetManager.dir, outfile)

    WorldStructure.create_world(robmodelpath, structure=structure, filename=outfile)
    Tuple(FlagrunEnv(s, Rob(s), interval=interval, rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

AntFlagrunEnv(;interval=100, seed=nothing) = first(tconstruct(FlagrunEnv, Ant, 1; interval=interval, seed=seed))

robot(env::FlagrunEnv) = env.robot
LyceumMuJoCo.obsspace(env::FlagrunEnv) = env.obsspace

function _movetarget!(env::FlagrunEnv, pos::Vector{T}) where T
    offset = rand(env.rng, Uniform(-1, 1), 2)
    targ = offset * 5 + pos
    
    # targ = [20 * rand(env.rng) - 10, 20 * rand(env.rng) - 10]
    while euclidean(pos, targ) < FLAGRUN_DIST_THRESH
        # targ = [20 * rand(env.rng) - 10, 20 * rand(env.rng) - 10]
        offset = rand(env.rng, Uniform(-1, 1), 2)
        targ = offset * 5 + pos
    end

    # bit of a hack: moving the extra geom in the xml to indicate target position
    getsim(env).mn[:geom_pos][ngeom=:target_geom] = [targ..., 0]
    env.target = targ
    env.d_old = euclidean(pos, env.target)
    env.targ_start_dist = env.d_old
    # println("moved target: $(_torso_xy(env)) - $(targ) - $(euclidean(pt, targ))")
    targ
end

_movetarget!(env::FlagrunEnv) = _movetarget!(env, _torso_xy(env))

function LyceumMuJoCo.step!(env::FlagrunEnv)
    env.evalrew -= env.d_old
    
    env.t += 1
    if env.t % env.interval == 0
        env.rew_once = true
        _movetarget!(env)
    end

    _step!(env)
end

function LyceumMuJoCo.getobs!(obs, env::FlagrunEnv)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        targetvec = env.target - _torso_xy(env)

        angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)
        copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])
        # copyto!(shaped.targetvec, normalize(targetvec))
        copyto!(shaped.d_old, [env.d_old / env.targ_start_dist])
        copyto!(shaped.robobs, getobs!(getsim(env), robot(env), shaped.robobs))
    end

    obs
end

function LyceumMuJoCo.reset!(env::FlagrunEnv)
    env.t = 0
    env.evalrew = 0
    env.rew_once = true
    r = _reset!(env)
    _movetarget!(env, [0,0])
    r
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::FlagrunEnv)
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
    # LyceumMuJoCo.isdone(env) ? -100000 : rew  # penalizing for dying
end

LyceumMuJoCo.geteval(env::FlagrunEnv) = env.evalrew

# TODO maybe torso_x should be specifically included for plain walker envs
_set_prev_pos!(env::FlagrunEnv, pos) = nothing
_set_prev_pos!(shapedstate, ::FlagrunEnv, pos) = nothing
get_prev_pos(env::FlagrunEnv) = nothing
get_prev_pos(shapedstate, ::FlagrunEnv) = nothing

getpos(env::FlagrunEnv) = _torso_xy(env)
getpos(shapedstate, env::FlagrunEnv) = _torso_xy(shapedstate, env)

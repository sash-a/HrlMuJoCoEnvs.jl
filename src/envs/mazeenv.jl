MAZE_DIST_THRESH = 5

mutable struct MazeEnv{SIM<:MJSim, S, O} <: AbstractWalker3DMJEnv
    sim::SIM
    robot::AbstractRobot
    statespace::S
    obsspace::O
    last_torso_x::Float64
    target::Vector{Number}
    t::Int
    d_old::Float64
    start_targ_dist::Float64
    rng::MersenneTwister

    function MazeEnv(sim::MJSim, robot; rng=MersenneTwister())
        sspace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            targetvec=VectorShape(Float64, 2),
            d_old=VectorShape(Float64, 1),
            robobs=obsspace(robot), 
            t=ScalarShape(Float64)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, robot, sspace, ospace, 0, [0, 0], 0, 0, 0, rng)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{MazeEnv}, rob::Type{<:AbstractRobot},n::Integer; structure::Matrix{<:AbstractBlock}=WorldStructure.basic_maze_structure, seed=nothing)
    filename = "mazetmp.xml"
    robmodelpath = getfile(rob)

    WorldStructure.create_world(robmodelpath, structure=structure, wsize=8, filename=filename)
    modelpath = joinpath(AssetManager.dir, filename)

    Tuple(MazeEnv(s, rob(s), rng=MersenneTwister(seed)) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

PointMazeEnv1(;structure=WorldStructure.basic_maze_structure, seed=nothing) = first(tconstruct(MazeEnv, PointMass, 1; structure=structure, seed=seed))
AntMazeEnv1(;structure=WorldStructure.basic_maze_structure, seed=nothing) = first(tconstruct(MazeEnv, Ant, 1; structure=structure, seed=seed))

robot(env::MazeEnv) = env.robot
LyceumMuJoCo.obsspace(env::MazeEnv) = env.obsspace

function _movetarget!(env::MazeEnv)
    zones = [(12, 20, -4, 12), (-4, 20, 12, 20), (-4, 12, -4, 4)]  # [(xmin, xmax, ymin, ymax)...]
    areas = map(((xmin, xmax, ymin, ymax),)->((xmax-xmin)*(ymax-ymin)), zones)
    weighting = map(a->a/sum(areas), areas)
    xmin, xmax, ymin, ymax = sample(zones, Weights(weighting))

    env.target = [rand(env.rng, Uniform(xmin, xmax)), rand(env.rng, Uniform(ymin, ymax))]

    env.start_targ_dist = env.d_old = euclidean(_torso_xy(env), env.target)
    getsim(env).mn[:geom_pos][ngeom=:target_geom] = [env.target..., 0]

end

function LyceumMuJoCo.step!(env::MazeEnv)
    env.t += 1
    _step!(env)
end

function LyceumMuJoCo.reset!(env::MazeEnv)
    env.t = 0
    _movetarget!(env)
    _reset!(env)
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::MazeEnv)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        env.d_old = euclidean(_torso_xy(env), env.target)
        1 - (env.d_old / env.start_targ_dist) + (env.d_old < MAZE_DIST_THRESH ? 2 : 0)
    end
end

LyceumMuJoCo.geteval(env::MazeEnv) = euclidean(_torso_xy(env), [0, 16]) < MAZE_DIST_THRESH ? 1 : 0

function LyceumMuJoCo.getobs!(obs, env::MazeEnv)
    checkaxes(obsspace(env), obs)

    dn = env.sim.dn
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        targetvec = env.target - _torso_xy(env)
        # angle_to_target = atan(targetvec[2], targetvec[1]) - LyceumMuJoCo._torso_ang(env)
        # copyto!(shaped.targetvec, [sin(angle_to_target), cos(angle_to_target)])
        
        copyto!(shaped.targetvec, normalize(targetvec))
        copyto!(shaped.d_old, [env.d_old / 1000])
        copyto!(shaped.robobs, getobs(getsim(env), robot(env)))
        shaped.t = env.t * 0.001
    end

    obs
end
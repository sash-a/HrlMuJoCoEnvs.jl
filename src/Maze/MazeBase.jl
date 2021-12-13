abstract type AbstractMazeEnv <: WalkerBase.AbstractWalkerMJEnv end

const MAZE_DIST_THRESH = 5
const MAZE_TARGET = [0, 16]


function LyceumMuJoCo.step!(env::AbstractMazeEnv)
    env.t += 1
    WalkerBase._step!(env)
end

function _movetarget!(env::AbstractMazeEnv)
    zones = [(12, 20, -4, 12), (-4, 20, 12, 20), (-4, 12, -4, 4)]  # [(xmin, xmax, ymin, ymax)...]
    areas = map(((xmin, xmax, ymin, ymax),)->((xmax-xmin)*(ymax-ymin)), zones)
    weighting = map(a->a/sum(areas), areas)
    xmin, xmax, ymin, ymax = sample(zones, Weights(weighting))

    env.target = [rand(env.rng, Uniform(xmin, xmax)), rand(env.rng, Uniform(ymin, ymax))]

    env.start_targ_dist = env.d_old = euclidean(_torso_xy(env), env.target)
    getsim(env).mn[:geom_pos][:, Val(:target_geom)] = [env.target..., 0]
end

function LyceumMuJoCo.reset!(env::AbstractMazeEnv)
    env.t = 0
    r = WalkerBase._reset!(env)
    _movetarget!(env)
    r
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AbstractMazeEnv)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        env.d_old = euclidean(_torso_xy(shapedstate, env), env.target)
        # 1 - (env.d_old / env.start_targ_dist) + (env.d_old < MAZE_DIST_THRESH ? 2 : 0)
        -env.d_old
    end
end

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AbstractMazeEnv) = false

LyceumMuJoCo.geteval(env::AbstractMazeEnv) = euclidean(_torso_xy(env), MAZE_TARGET) < MAZE_DIST_THRESH ? 1 : 0
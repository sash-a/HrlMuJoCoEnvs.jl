# Cheating a bit by assuming Gather envs will have certain attributes 
# TODO make accesor methods for these attributes

abstract type Collectible end
mutable struct Apple <: Collectible
    pos::Tuple{Float64, Float64}
end
mutable struct Bomb <: Collectible 
    pos::Tuple{Float64, Float64}
end
Apple() = Apple((0.,0.))
Bomb() = Bomb((0.,0.))
getpos(::Collectible) = (0, 0)
getpos(a::Apple) = a.pos
getpos(a::Bomb) = a.pos

abstract type AbstractGatherEnv <: WalkerBase.AbstractWalkerMJEnv end

collectibles(env::AbstractGatherEnv) = merge(env.apples, env.bombs)

function _sensor_readings(env::AbstractGatherEnv)
    apple_readings = zeros(env.nbins)
    bomb_readings = zeros(env.nbins)
    robot_x, robot_y = _torso_xy(env)

    collectibles_list = collect(keys(collectibles(env)))
    collectible_dists = map(c-> (c, sqeuclidean(collect(getpos(c)), [robot_x, robot_y])), collectibles_list)
    sorted_collectibles = sort(collectible_dists; by=c->last(c), rev=true)
    bin_res = env.sensor_span / env.nbins
    ori = LyceumMuJoCo._torso_ang(env)

    if env.viz
        for i in 1:env.nbins
            getsim(env).mn[:geom_pos][:, Val(Symbol("apple_sensor_$i"))] = [0, 0, -2]
            getsim(env).mn[:geom_pos][:, Val(Symbol("bomb_sensor_$i"))] = [0, 0, -2]
        end
    end

    for (c, dist) in sorted_collectibles
        if dist > env.sensor_range^2
            continue
        end

        cx, cy = getpos(c)
        angle = atan(cy - robot_y, cx - robot_x) - ori
        
        angle = angle % (2 * π)
        if angle > π
            angle = angle - 2 * π
        end
        if angle < -π
            angle = angle + 2 * π
        end

        half_span = env.sensor_span * 0.5
        if abs(angle) >= half_span
            continue
        end
        
        bin_number = Int(floor((angle + half_span) / bin_res)) + 1
        intensity = 1.0 - dist / env.sensor_range^2
        if c isa Apple
            apple_readings[bin_number] = intensity
            if env.viz
                getsim(env).mn[:geom_pos][:, Val(Symbol("apple_sensor_$bin_number"))] = [cx, cy, 2]
            end
        else
            bomb_readings[bin_number] = intensity
            if env.viz
                getsim(env).mn[:geom_pos][:, Val(Symbol("bomb_sensor_$bin_number"))] = [cx, cy, 2]
            end
        end
    end
    
    apple_readings, bomb_readings
end

function _collect_collectibles!(env::AbstractGatherEnv)
    pos = _torso_xy(env)
    r = 0
    collected = []
    for (c, i) in collectibles(env)
        if sqeuclidean(collect(getpos(c)), pos) < env.catch_range
            r += c isa Apple ? 1 : -1
            push!(collected, c)

            if env.viz
                id = c isa Apple ? "apple_$i" : "bomb_$i"
                getsim(env).mn[:geom_pos][:, Val(Symbol(id))] = [0, 0, -10]
            end
        end
    end

    filter!(((a, _),) -> !(a in collected), env.apples)
    filter!(((b, _),) -> !(b in collected), env.bombs)

    r
end

function _move_collectibles!(env::AbstractGatherEnv)
    env.apples = Dict{Apple, Int}()
    env.bombs = Dict{Bomb, Int}()

    for i in 1:env.napples
        x, y = 0, 0
        while x^2 + y^2 < env.robot_object_spacing^2
            x = rand(env.rng, Uniform(-env.activity_range / 2, env.activity_range / 2)) * 2
            y = rand(env.rng, Uniform(-env.activity_range / 2, env.activity_range / 2)) * 2    
        end

        env.apples[Apple((x, y))] = i
        if env.viz
            getsim(env).mn[:geom_pos][:, Val(Symbol("apple_$i"))] = [x, y, 0]
        end
    end

    for i in 1:env.nbombs
        x, y = 0, 0
        while x^2 + y^2 < env.robot_object_spacing^2
            x = rand(Uniform(-env.activity_range / 2, env.activity_range / 2)) * 2
            y = rand(Uniform(-env.activity_range / 2, env.activity_range / 2)) * 2    
        end

        env.bombs[Bomb((x, y))] = i
        if env.viz
            getsim(env).mn[:geom_pos][:, Val(Symbol("bomb_$i"))] = [x, y, 0]
        end
    end
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AbstractGatherEnv)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)

    _collect_collectibles!(env)
end
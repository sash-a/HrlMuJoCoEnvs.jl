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

mutable struct GatherEnv{SIM<:MJSim, S, O} <: AbstractWalker3DMJEnv
    sim::SIM
    robot::AbstractRobot
    statespace::S
    obsspace::O

    target::Vector{Number}
    t::Int

    napples::Int
    nbombs::Int
    activity_range::Float64
    robot_object_spacing::Float64
    catch_range::Float64
    nbins::Int
    sensor_range::Float64
    sensor_span::Float64

    apples::Dict{Apple, Int}  # (Apple => it's ID/name in the .xml file)
    bombs::Dict{Bomb, Int}

    rng::MersenneTwister
    viz::Bool

    function GatherEnv(sim::MJSim, robot;                                                          
                        napples=8,
                        nbombs=8,
                        activity_range=10.,
                        robot_object_spacing=2.,
                        catch_range=1,
                        nbins=8,
                        sensor_range=8.,
                        sensor_span=2*π,
                        rng=MersenneTwister(),
                        viz=false)
                    
        sspace = MultiShape(
            simstate=statespace(sim),
            sensor_readings=VectorShape(Float64, nbins * 2)
        )
        ospace = MultiShape(
            robobs=obsspace(robot), 
            sensor_readings=VectorShape(Float64, nbins * 2),
            t=ScalarShape(Int)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, robot, sspace, ospace, [0, 0], 0,            
                                                                napples,
                                                                nbombs,
                                                                activity_range,
                                                                robot_object_spacing,
                                                                catch_range,
                                                                nbins,
                                                                sensor_range,
                                                                sensor_span,
                                                                Dict([(Apple(), i) for i in 1:napples]),
                                                                Dict([(Bomb(), i) for i in 1:nbombs]),
                                                                rng,
                                                                viz)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{GatherEnv}, Robot::Type{<:AbstractRobot}, n::Integer; 
                                structure::Matrix{<:AbstractBlock}=WorldStructure.wall_structure, 
                                napples::Int=8, nbombs::Int=8, nbins::Int=8, seed=nothing, viz=false)
    modelpath = getfile(Robot)
    outfile = "gathertmp.xml"
    WorldStructure.create_world(modelpath, napples, nbombs, nbins; structure=structure, wsize=8, viz=viz, filename=outfile)
    modelpath = joinpath(AssetManager.dir, outfile)

    Tuple(GatherEnv(s, Robot(s); napples=napples, nbombs=nbombs, rng=MersenneTwister(seed), viz=viz) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

PointGatherEnv1(;viz=false) = first(LyceumBase.tconstruct(GatherEnv, PointMass, 1; viz=viz))
AntGatherEnv1(;viz=false) = first(LyceumBase.tconstruct(GatherEnv, Ant, 1; viz=viz))

collectibles(env::GatherEnv) = merge(env.apples, env.bombs)
robot(env::GatherEnv) = env.robot
LyceumMuJoCo.obsspace(env::GatherEnv) = env.obsspace

function _sensor_readings(env::GatherEnv)
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
            getsim(env).mn[:geom_pos][ngeom=Symbol("apple_sensor_$i")] = [0, 0, -2]
            getsim(env).mn[:geom_pos][ngeom=Symbol("bomb_sensor_$i")] = [0, 0, -2]
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
                getsim(env).mn[:geom_pos][ngeom=Symbol("apple_sensor_$bin_number")] = [cx, cy, 2]
            end
        else
            bomb_readings[bin_number] = intensity
            if env.viz
                getsim(env).mn[:geom_pos][ngeom=Symbol("bomb_sensor_$bin_number")] = [cx, cy, 2]
            end
        end
    end
    
    apple_readings, bomb_readings
end

function _collect_collectibles!(env::GatherEnv)
    pos = _torso_xy(env)
    r = 0
    collected = []
    for (c, i) in collectibles(env)
        if sqeuclidean(collect(getpos(c)), pos) < env.catch_range
            r += c isa Apple ? 1 : -1
            push!(collected, c)

            if env.viz
                id = c isa Apple ? "apple_$i" : "bomb_$i"
                getsim(env).mn[:geom_pos][ngeom=Symbol(id)] = [0, 0, -10]
            end
        end
    end

    filter!(((a, _),) -> !(a in collected), env.apples)
    filter!(((b, _),) -> !(b in collected), env.bombs)

    r
end

function _move_collectibles!(env::GatherEnv)
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
            getsim(env).mn[:geom_pos][ngeom=Symbol("apple_$i")] = [x, y, 0]
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
            getsim(env).mn[:geom_pos][ngeom=Symbol("bomb_$i")] = [x, y, 0]
        end
    end
end

function LyceumMuJoCo.step!(env::GatherEnv)
    env.t += 1
    _step!(env)
end

function LyceumMuJoCo.getobs!(obs, env::GatherEnv)
    checkaxes(obsspace(env), obs)

    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        copyto!(shaped.robobs, getobs!(getsim(env), robot(env), shaped.robobs))
        shaped.sensor_readings .= vcat(_sensor_readings(env)...)
        shaped.t = env.t * 0.001
    end
    obs
end

function LyceumMuJoCo.reset!(env::GatherEnv)
    env.t = 0
    _move_collectibles!(env)
    _reset!(env)
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::GatherEnv)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)

    _collect_collectibles!(env)
end

_set_prev_pos!(env::GatherEnv, pos) = nothing
_set_prev_pos!(shapedstate, ::GatherEnv, pos) = nothing
get_prev_pos(env::GatherEnv) = nothing
get_prev_pos(shapedstate, ::GatherEnv) = nothing

getpos(env::GatherEnv) = _torso_xy(env)
getpos(shapedstate, env::GatherEnv) = _torso_xy(shapedstate, env)
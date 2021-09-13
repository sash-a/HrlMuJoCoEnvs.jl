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

mutable struct AntGather{SIM<:MJSim, S, O} <: WalkerBase.AbstractWalkerMJEnv
    sim::SIM
    statespace::S
    obsspace::O
    last_torso_x::Float64
    structure::Matrix{AbstractBlock}
    target::Vector{Number}
    t::Int

    napples::Int
    nbombs::Int
    activity_range::Float64
    robot_object_spacing::Float64
    catch_range::Float64
    n_bins::Int
    sensor_range::Float64
    sensor_span::Float64
    dying_cost::Float64

    apples::Dict{Apple, Int}  # (Apple => it's ID/name in the .xml file)
    bombs::Dict{Bomb, Int}

    rng::MersenneTwister
    viz::Bool

    function AntGather(sim::MJSim; structure=MazeStructure.wall_structure,                                                                 
                        napples=8,
                        nbombs=8,
                        activity_range=6.,
                        robot_object_spacing=2.,
                        catch_range=1.,
                        n_bins=10,
                        sensor_range=6.,
                        sensor_span=π,
                        dying_cost=-10,
                        rng=MersenneTwister(),
                        viz=false)
                    
        sspace = MultiShape(
            simstate=statespace(sim),
            sensor_readings = VectorShape(Float64, n_bins * 2),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 2),
            qvel = VectorShape(Float64, sim.m.nv),
            sensor_readings = VectorShape(Float64, n_bins * 2)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, structure, [0, 0], 0,            
                                                                napples,
                                                                nbombs,
                                                                activity_range,
                                                                robot_object_spacing,
                                                                catch_range,
                                                                n_bins,
                                                                sensor_range,
                                                                sensor_span,
                                                                dying_cost,
                                                                Dict([(Apple(), i) for i in 1:napples]),
                                                                Dict([(Bomb(), i) for i in 1:nbombs]),
                                                                rng,
                                                                viz)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{AntGather}, n::Integer; 
                                structure::Matrix{<:AbstractBlock}=MazeStructure.wall_structure, 
                                napples::Int=8, nbombs::Int=8, nbins::Int=10, seed=nothing, viz=false)
    antmodelpath = joinpath(@__DIR__, "..", "assets", "ant.xml")
    outfile = "gathertmp.xml"
    MazeStructure.create_world(antmodelpath, napples, nbombs, nbins; structure=structure, wsize=6, viz=viz, filename=outfile)
    modelpath = joinpath(@__DIR__, "..", "assets", outfile)
    
    Tuple(AntGather(s; structure=structure, napples=napples, nbombs=nbombs, rng=MersenneTwister(seed), viz=viz) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntGather(;viz=false) = first(LyceumBase.tconstruct(AntGather, 1; viz=viz))
collectibles(env::AntGather) = merge(env.apples, env.bombs)

function _sensor_readings(env::AntGather)
    apple_readings = zeros(env.n_bins)
    bomb_readings = zeros(env.n_bins)
    robot_x, robot_y = _torso_xy(env)

    collectibles_list = collect(keys(collectibles(env)))
    collectible_dists = map(c-> (c, sqeuclidean(collect(getpos(c)), [robot_x, robot_y])), collectibles_list)
    sorted_collectibles = sort(collectible_dists; by=c->last(c), rev=true)
    bin_res = env.sensor_span / env.n_bins
    ori = LyceumMuJoCo._torso_ang(env)

    if env.viz
        for i in 1:env.n_bins
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

function _collect_collectibles!(env::AntGather)
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

function _move_collectibles!(env::AntGather)
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

function LyceumMuJoCo.getobs!(obs, env::AntGather)
    # TODO sensor readings
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)

        copyto!(shaped.cropped_qpos, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        copyto!(shaped.sensor_readings, vcat(_sensor_readings(env)...))
        clamp!(shaped.qvel, -10, 10)
    end

    obs
end

function LyceumMuJoCo.reset!(env::AntGather)
    _move_collectibles!(env)
    WalkerBase._reset!(env)
end

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntGather)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.38 <= height <= 1)
        done
    end
end

function LyceumMuJoCo.getreward(state, action, ::Any, env::AntGather)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)

    _collect_collectibles!(env)
end

@inline _torso_xy(env::AntGather) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntGather) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntGather) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntGather) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntGather) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntGather) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
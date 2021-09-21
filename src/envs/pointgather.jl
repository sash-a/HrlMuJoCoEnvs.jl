mutable struct GatherEnv{SIM<:MJSim, S, O} <: AbstractGatherEnv
    sim::SIM
    robot::AbstractRobot
    statespace::S
    obsspace::O
    last_torso_x::Float64


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
    dying_cost::Float64

    apples::Dict{Apple, Int}  # (Apple => it's ID/name in the .xml file)
    bombs::Dict{Bomb, Int}

    rng::MersenneTwister
    viz::Bool

    function GatherEnv(sim::MJSim, robot;                                                          
                        napples=8,
                        nbombs=8,
                        activity_range=6.,
                        robot_object_spacing=2.,
                        catch_range=1.5,
                        nbins=10,
                        sensor_range=6.,
                        sensor_span=2*π,
                        dying_cost=-10,
                        rng=MersenneTwister(),
                        viz=false)
                    
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64),
            sensor_readings = VectorShape(Float64, nbins * 2)
        )
        @show typeof(robot) obsspace(robot)
        ospace = MultiShape(
            robobs = obsspace(robot), 
            sensor_readings = VectorShape(Float64, nbins * 2)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, robot, sspace, ospace, 0, [0, 0], 0,            
                                                                napples,
                                                                nbombs,
                                                                activity_range,
                                                                robot_object_spacing,
                                                                catch_range,
                                                                nbins,
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

function LyceumBase.tconstruct(Env::Type{GatherEnv}, Robot::Type{<:AbstractRobot}, n::Integer; 
                                structure::Matrix{<:AbstractBlock}=WorldStructure.wall_structure, 
                                napples::Int=8, nbombs::Int=8, nbins::Int=8, seed=nothing, viz=false)
    antmodelpath = getfile(Robot)
    outfile = "pointgathertmp.xml"
    WorldStructure.create_world(antmodelpath, napples, nbombs, nbins; structure=structure, wsize=6, viz=viz, filename=outfile)
    modelpath = joinpath(AssetManager.dir, outfile)

    Tuple(GatherEnv(s, Robot(); napples=napples, nbombs=nbombs, rng=MersenneTwister(seed), viz=viz) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=5))
end

PointGatherEnv1(;viz=false) = first(LyceumBase.tconstruct(GatherEnv, PointMass, 1; viz=viz))

robot(env::GatherEnv) = env.robot

function LyceumMuJoCo.getobs!(obs, env::GatherEnv)
    checkaxes(obsspace(env), obs)

    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        copyto!(shaped.robobs, getobs!(getsim(env), robot(env)))
        shaped.sensor_readings .= vcat(_sensor_readings(env)...)
    end
    obs
end

function LyceumMuJoCo.reset!(env::GatherEnv)
    _move_collectibles!(env)
    WalkerBase._reset!(env)
end
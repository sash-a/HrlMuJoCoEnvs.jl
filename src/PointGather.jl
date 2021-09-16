mutable struct PointGatherEnv{SIM<:MJSim, S, O} <: AbstractGatherEnv
    sim::SIM
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

    function PointGatherEnv(sim::MJSim;                                                          
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
        ospace = MultiShape(
            agent_pos = VectorShape(Float64, 2),
            agent_vel = VectorShape(Float64, 2),
            sensor_readings = VectorShape(Float64, nbins * 2)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, [0, 0], 0,            
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

function LyceumBase.tconstruct(::Type{PointGatherEnv}, n::Integer; 
                                structure::Matrix{<:AbstractBlock}=MazeStructure.wall_structure, 
                                napples::Int=8, nbombs::Int=8, nbins::Int=10, seed=nothing, viz=false)
    antmodelpath = joinpath(@__DIR__, "..", "assets", "pointmass.xml")
    outfile = "pointgathertmp.xml"
    MazeStructure.create_world(antmodelpath, napples, nbombs, nbins; structure=structure, wsize=6, viz=viz, filename=outfile)
    modelpath = joinpath(@__DIR__, "..", "assets", outfile)

    Tuple(PointGatherEnv(s; napples=napples, nbombs=nbombs, rng=MersenneTwister(seed), viz=viz) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

PointGatherEnv(;viz=false) = first(LyceumBase.tconstruct(PointGatherEnv, 1; viz=viz))

@inline LyceumBase.obsspace(env::PointGatherEnv) = env.obsspace
function LyceumMuJoCo.getobs!(obs, env::PointGatherEnv)
    checkaxes(obsspace(env), obs)

    dn = env.sim.dn
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        shaped.agent_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent]
        shaped.agent_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
        shaped.sensor_readings .= vcat(_sensor_readings(env)...)
    end
    obs
    # @views @uviews qpos obs begin
    #     shaped = obsspace(env)(obs)

    #     copyto!(shaped.cropped_qpos, qpos[3:end])
    #     copyto!(shaped.qvel, env.sim.d.qvel)
    #     copyto!(shaped.sensor_readings, vcat(_sensor_readings(env)...))
    #     clamp!(shaped.qvel, -10, 10)
    # end

    # obs
end

function LyceumMuJoCo.reset!(env::PointGatherEnv)
    _move_collectibles!(env)
    WalkerBase._reset!(env)
end

LyceumMuJoCo.isdone(state, ::Any, ::Any, env::PointGatherEnv) = false


@inline _torso_xy(env::PointGatherEnv) = [env.sim.dn.xpos[:x, :agent], env.sim.dn.xpos[:y, :agent]]
@inline _torso_xy(shapedstate::ShapedView, ::PointGatherEnv) = shapedstate.agent_pos

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::PointGatherEnv) = 0.
@inline LyceumMuJoCo._torso_height(env::PointGatherEnv) = 0.

@inline LyceumMuJoCo._torso_ang(env::PointGatherEnv) = 0.
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::PointGatherEnv) = 0.
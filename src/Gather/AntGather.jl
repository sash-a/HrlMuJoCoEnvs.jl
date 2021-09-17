mutable struct AntGatherEnv{SIM<:MJSim, S, O} <: AbstractGatherEnv
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
    nbins::Int
    sensor_range::Float64
    sensor_span::Float64

    apples::Dict{Apple, Int}  # (Apple => it's ID/name in the .xml file)
    bombs::Dict{Bomb, Int}

    rng::MersenneTwister
    viz::Bool

    function AntGatherEnv(sim::MJSim; structure=WorldStructure.wall_structure,                                                                 
                        napples=8,
                        nbombs=8,
                        activity_range=10.,
                        robot_object_spacing=2.,
                        catch_range=1,
                        nbins=10,
                        sensor_range=8.,
                        sensor_span=2*π,
                        rng=MersenneTwister(),
                        viz=false)
                    
        sspace = MultiShape(
            simstate=statespace(sim),
            sensor_readings = VectorShape(Float64, nbins * 2),
            last_torso_x=ScalarShape(Float64),
            t=ScalarShape(Int)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq),
            qvel = VectorShape(Float64, sim.m.nv),
            sensor_readings = VectorShape(Float64, nbins * 2),
            t=ScalarShape(Int)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(sim, sspace, ospace, 0, structure, [0, 0], 0,            
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

function LyceumBase.tconstruct(::Type{AntGatherEnv}, n::Integer; 
                                structure::Matrix{<:AbstractBlock}=WorldStructure.wall_structure, 
                                napples::Int=8, nbombs::Int=8, nbins::Int=10, seed=nothing, viz=false)
    antmodelpath = joinpath(AssetManager.dir, "easier_ant.xml")
    outfile = "gathertmp.xml"
    WorldStructure.create_world(antmodelpath, napples, nbombs, nbins; structure=structure, wsize=8, viz=viz, filename=outfile)
    modelpath = joinpath(AssetManager.dir, outfile)
    
    Tuple(AntGatherEnv(s; structure=structure, napples=napples, nbombs=nbombs, rng=MersenneTwister(seed), viz=viz) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip=4))
end

AntGatherEnv(;viz=false) = first(LyceumBase.tconstruct(AntGatherEnv, 1; viz=viz))

function LyceumMuJoCo.step!(env::AntGatherEnv)
    env.t += 1
    WalkerBase._step!(env)
end

function LyceumMuJoCo.getobs!(obs, env::AntGatherEnv)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)

        copyto!(shaped.cropped_qpos, qpos)
        copyto!(shaped.qvel, env.sim.d.qvel)
        copyto!(shaped.sensor_readings, vcat(_sensor_readings(env)...))
        shaped.t = env.t
        clamp!(shaped.qvel, -10, 10)
    end

    obs
end

function LyceumMuJoCo.reset!(env::AntGatherEnv)
    env.t = 0
    _move_collectibles!(env)
    WalkerBase._reset!(env)
end

function LyceumMuJoCo.isdone(state, ::Any, ::Any, env::AntGatherEnv)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        height = LyceumMuJoCo._torso_height(shapedstate, env)
        done = !(all(isfinite, state) && 0.2 <= height <= 1)
        done
    end
end

@inline _torso_xy(env::AntGatherEnv) = env.sim.d.qpos[1:2]
@inline _torso_xy(shapedstate::ShapedView, ::AntGatherEnv) = shapedstate.simstate.qpos[1:2]

@inline LyceumMuJoCo._torso_height(shapedstate::ShapedView, ::AntGatherEnv) = shapedstate.simstate.qpos[3]
@inline LyceumMuJoCo._torso_height(env::AntGatherEnv) = env.sim.d.qpos[3]

@inline LyceumMuJoCo._torso_ang(env::AntGatherEnv) = torso_ori(env.sim.d.qpos[ori_ind:ori_ind + 3])
@inline LyceumMuJoCo._torso_ang(shapedstate::ShapedView, ::AntGatherEnv) = torso_ori(shapedstate.simstate.qpos[ori_ind:ori_ind + 3])
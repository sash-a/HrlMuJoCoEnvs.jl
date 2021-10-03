# HrlMuJoCoEnvs.jl

HRL envs from [Data efficient hierarchical reinforcement learning](https://arxiv.org/abs/1805.08296) in Julia using [Lyceum MuJoCo](https://github.com/Lyceum/LyceumMuJoCo.jl)

## Available environments
* Ant gather
* Point gather
* Ant maze
* Point maze
* Ant push
* Ant fall

## Installation

Make sure you check the install instructions for [Lyceum MuJoCo](https://github.com/Lyceum/LyceumMuJoCo.jl)
```
pkg> add https://github.com/sash-a/HrlMuJoCoEnvs.jl/
```

## Example usage
```
using HrlMuJoCoEnvs
using LyceumMuJoCo

env = HrlMuJoCoEnvs.AntGatherEnv()

for i in 1:100
    step!(env)
    o = getobs(env)
    setaction!(env, rand(8))
    @show getreward(env)
    if isdone(env)
        println("done")
    end
end

```

ps. appologies for some poor coding practise, this is research code.

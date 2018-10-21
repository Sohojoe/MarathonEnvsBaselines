# MarathonEnvs + OpenAi.Baselines

Explority implementation of 
* MarathonEnvs
* ml-agents
* openai.baselines
* stable.baselines


### source versions
* MarathonEnvs
* ml-agents = 0.5.1
* openai.baselines = 7bfbcf1
* stable.baselines = [v2.1.0](https://github.com/hill-a/stable-baselines/tree/v2.1.0)

-----
# Stable.Baselines
Note: Stable Baselines is a fork of OpenAI.Baselines which addresses some issues with OpenAI.Baselines (main one for me is that OpenAI.Baselines cannot save enviroments with normalized observations)

python train_simple.py - example of training a single agent
python train.py - trains across 4 cpus
python run.py - example that loads and runs a trained model

-----
# OpenAI.Baselines
### Example command lines

#### To enable Tensorboard
```
export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' 
export OPENAI_LOGDIR=summaries
```

#### ppo2 for 1m steps

```
mpiexec -n 4 python -m baselines.run_unity --alg=ppo2 --env=./envs/hopper --num_timesteps=1e6 --save_path=./models/hopper_1m_ppo2
```
Note: play is currently broken in openai.baselines for normalized enviorments
```
python -m baselines.run_unity --alg=ppo2 --env=./envs/hopper-run —num_timesteps=0 --load_path=./models/hopper_1m_ppo2 --play
```

#### acktr
```
mpiexec -n 4 python -m baselines.run_unity --alg=acktr --env=./envs/walker --num_timesteps=1e6  --save_path=./models/walker_1m_acktr

python -m baselines.run_unity --alg=acktr --env=./envs/walker-run --num_timesteps=0 --load_path=./models/walker_1m_acktr --play
```

#### acer
```
mpiexec -n 4 python -m baselines.run_unity --alg=acer --env=./envs/walker --num_timesteps=1e6  --save_path=./models/walker_1m_acer
```

#### a2c
```
mpiexec -n 4 python -m baselines.run_unity --alg=a2c --env=./envs/walker --num_timesteps=1e6  --save_path=./models/walker_1m_a2c
```


#### gail
```
mpiexec -n 4 python -m baselines.run_unity --alg=gail --env=./envs/walker --num_timesteps=1e6  --save_path=./models/walker_1m_gail
```



## example command lines - not working yet
#### her
```
mpiexec -n 4 python -m baselines.run_unity --alg=her --env=./envs/walker --num_timesteps=1e6  --save_path=./models/walker_1m_her

python -m baselines.run_unity --alg=her --env=./envs/walker-run --num_timesteps=0 --load_path=./models/walker_1m_her —-play
```


-----
# ml-agents

#### train using marathon_envs_config.yaml
```
mlagents-learn config/marathon_envs_config.yaml --train --worker-id=10 --env=./envs/hopper-x16 --run-id=hopper.001

set CUDA_VISIBLE_DEVICES=-1 & mlagents-learn config/joints_config.yaml --train --worker-id=2 --env="\b\TestJoint001b\Unity Environment.exe" --run-id=TestJoint001.057
```

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
Note: Stable Baselines is a fork of OpenAI.Baselines which addresses some issue (main one for me is that OpenAI.Baselines cannot save enviroments with normalized observations)

python train_simple.py - example of training a single agent
python train.py - trains across 4 cpus
python run.py - example that loads and runs a trained model



-----
# OpenAI.Baselines
### Example command lines

###### ppo2
mpiexec -n 4 python -m baselines.run_unity --alg=ppo2 --env=./envs/Hopper --num_timesteps=2e5 --save_path=./models/hopper_200k_ppo2

python -m baselines.run_unity --alg=ppo2 --env=./envs/Hopper —num_timesteps=0 --load_path=./models/hopper_200k_ppo2 --play

###### acktr
mpiexec -n 4 python -m baselines.run_unity --alg=acktr --env=./envs/Walker --num_timesteps=2e5 --save_path=./models/walker_200k_acktr

python -m baselines.run_unity --alg=acktr --env=./envs/Walker --num_timesteps=0 --load_path=./models/walker_200k_acktr --play

###### acer
mpiexec -n 4 python -m baselines.run_unity --alg=acer --env=./envs/Walker --num_timesteps=2e5 --save_path=./models/walker_200k_acer

###### a2c
mpiexec -n 4 python -m baselines.run_unity --alg=a2c --env=./envs/Walker --num_timesteps=2e5 --save_path=./models/walker_200k_a2c


###### gail
mpiexec -n 4 python -m baselines.run_unity --alg=gail --env=./envs/Walker --num_timesteps=2e5 --save_path=./models/walker_200k_gail



## example command lines - not working yet
###### her
mpiexec -n 4 python -m baselines.run_unity --alg=her --env=./envs/Walker --num_timesteps=2e5 --save_path=./models/walker_200k_her

python -m baselines.run_unity --alg=her --env=./envs/Walker --num_timesteps=0 --load_path=./models/walker_200k_her —-play
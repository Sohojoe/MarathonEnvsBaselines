# MarathonEnvs + OpenAi.Baselines

Explority implementation of 
* MarathonEnvs
* ml-agents
* openai.baselines


### source versions
* MarathonEnvs
* ml-agents = 0.5.1
* openai.baselines = 7bfbcf1



### example command lines

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
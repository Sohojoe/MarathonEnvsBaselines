import gym

from baselines import deepq
from gym_unity.envs import UnityEnv

def main():
    env = UnityEnv("./envs/GridWorld", 0, use_visual=True)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
    )
    print("Saving model to unity_model.pkl")
    act.save("unity_model.pkl")


if __name__ == '__main__':
    main()

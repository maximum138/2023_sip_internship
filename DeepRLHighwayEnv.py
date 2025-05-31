import gym
from highway_env.envs.highway_env import HighwayEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def agent(observation):
    # Create a gym environment
    env = gym.make("highway-v0")

    # Wrap the environment in a vectorized environment
    env = DummyVecEnv([lambda: env])

    # Create the PPO agent
    agent = PPO2('MlpPolicy', env)

    # Use the agent to select an action
    action, _ = agent.predict(observation)
    return action

if __name__ == "__main__":
    config = {
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 100,
        "initial_spacing": 2,
        "simulation_frequency": 15,
        "policy_frequency": 0.25,
        "render_agent": True,
    }

    env = HighwayEnv(render_mode="rgb_array")
    env.configure(config)

    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = agent(obs)  # Pass the observation to the agent
            obs, reward, done, truncated, info = env.step(action)
            env.render()
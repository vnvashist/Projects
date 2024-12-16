import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
import matplotlib.pyplot as plt


class ResourceManagementEnv(gym.Env):
    def __init__(self, render_mode = None):
        super(ResourceManagementEnv, self).__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(8,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.food = 100
        self.wood = 100
        self.gold = 50
        self.stone = 50
        self.houses = 1
        self.soldiers = 0
        self.population = 5
        self.turn = 0
        self.max_turns = 1000 #APM (around 50) * (avg game time (20 mins))

        self.state = np.array(
            [self.food, self.wood, self.gold, self.stone,
             self.houses, self.soldiers, self.population, self.turn], dtype=np.float32
        )
        return self.state, {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.food > 50:
            reward += 2
        if self.wood > 50:
            reward += 2
        if self.gold > 30:
            reward += 1
        if self.stone > 30:
            reward += 1

        if action == 0: # Gather wood
            self.wood += self.population * 2
            reward +=1
        elif action == 1: # Gather food
            self.food += self.population * 2
            reward +=1
        elif action == 2: # Gather gold
            self.gold += self.population * 2
            reward +=1
        elif action == 3: # Gather stone
            self.stone += self.population * 2
            reward +=1
        elif action == 4: # Build house
            if self.wood >=50: # Need 50 wood to build house
                self.wood -= 50
                self.houses += 1
                reward += 10
            else:
                reward -= 5
        elif action == 5:
            if self.food >=50 and self.gold >=20:
                self.food -= 50
                self.gold -= 20
                self.soldiers += 1
                reward += 5
            else:
                reward -= 5

        # Simulates an attack every 10 actions
        if self.turn %10 == 0 and self.turn != 0:
            if self.soldiers >= 2:
                reward += 20
            else:
                self.population -=1
                reward -=20

        # Simulates demand for villager costs and also general importance of having food
        food_consumed = self.population * 1
        self.food -= food_consumed
        if self.food < 0:
            self.food = 0
            reward -= 10

        self.turn += 1

        self.state = np.array([self.wood, self.food, self.gold, self.stone,
                               self.houses, self.soldiers, self.population, self.turn], dtype=np.float32)
        if self.population <= 0:
            terminated = True
        if self.turn >= self.max_turns:
            truncated = True

        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Turn: {self.turn}")
            print(f"Food: {self.food}, Wood: {self.wood}, Gold: {self.gold}, Stone: {self.stone}")
            print(f"Houses: {self.houses}, Soldiers: {self.soldiers}, Population: {self.population}\n")

    def close(self):
        pass


env = ResourceManagementEnv(render_mode='human')
check_env(env)

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

env = DummyVecEnv([lambda: env])

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=100000)

model.save("ppo_resouce_management")

obs = env.reset()
print("reset output:" +  str(obs))
for i in range(50):
    action, _states = model.predict(obs)  # Predict an action
    obs, reward, terminated, truncated = env.step(action)  # Step through the environment

    terminated = terminated[0] if isinstance(terminated, list) or isinstance(terminated, np.ndarray) else terminated
    truncated = truncated[0] if isinstance(truncated, list) or isinstance(truncated, np.ndarray) else truncated
    if isinstance(truncated, dict):
        truncated = truncated.get("TimeLimit.truncated", False)

    env.render()
    print(f"Turn: {i+1}, Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        print("Game Over.")
        break


results = pd.read_csv(os.path.join(log_dir, 'monitor.csv'), skiprows=1)
plt.figure(figsize=(12, 6))
plt.plot(results['r'])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.show()

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
        self.action_space = spaces.Discrete(7) # for the number of actions (0-6)
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
        self.max_population = self.houses * 5 # Each house allows you to have 5 more population
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
            self.wood += (self.population // (10/3)) * 10 # Each villager can gather 10, 30% of pop to wood
            if self.population >= self.max_population -1: # Incentivize gathering wood when houses are needed
                reward += 3
            else:
                reward +=1
        elif action == 1: # Gather food
            self.food += (self.population // (10/3)) * 10 # 30% of villagers to food
            reward +=1
        elif action == 2: # Gather gold
            self.gold += (self.population // (10/2)) * 10 # 20% of villagers to gold
            reward +=1
        elif action == 3: # Gather stone
            self.stone += (self.population // (10/2)) * 10 # 20% of villagers to stone
            reward +=1
        elif action == 4: # Build house
            if self.wood >=50: # Need 50 wood to build house
                self.wood -= 50
                self.houses += 1
                old_max_population = self.max_population
                self.max_population = self.houses * 5

                if self.population >= old_max_population - 2:
                    reward += 10
                else:
                    reward += 5
            else:
                print(f"Failed to build house: Wood: {self.wood}")
                reward -= 5
        elif action == 5: # Train Soldiers
            if self.food >=50 and self.gold >=20:
                self.food -= 50
                self.gold -= 20
                self.soldiers += 1
                reward += 5
            else:
                print(f"Failed to train soldier: Food: {self.food}, Gold: {self.gold}")
                reward -= 5
        elif action == 6: # Create Villagers
            if self.population < self.max_population:
                if self.food >=20:
                    self.food -= 20
                    self.population += 1

                    if self.population >= self.max_population - 1:
                        reward += 20
                    else:
                        reward += 10
                else:
                    print(f"Failed to create villager: Food: {self.food}")
                    reward -= 5
            else:
                print(f"Failed to create villager: Not enough Max Pop Size")
                reward -= 5
        if self.population == self.max_population:
            reward -= 5

        if self.population < self.max_population and self.food >=20:
            reward -= 2


        # Simulates an attack every 10 actions
        if self.turn %10 == 0 and self.turn >= 500:
            if self.soldiers >= self.population // 4:
                reward += 20
            else:
                self.population -=10
                reward -= 40
        elif self.turn %10 == 0 and self.turn != 0:
            if self.soldiers >= self.population // 4:
                reward += 20
            else:
                self.population -=2
                reward -=20

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
            print(f"Houses: {self.houses}, Soldiers: {self.soldiers}, Population: {self.population}")

    def close(self):
        pass


env = ResourceManagementEnv(render_mode='human')
check_env(env)

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

env = DummyVecEnv([lambda: env])

model = PPO('MlpPolicy', env, verbose=1,
            learning_rate = 0.0003,
            gamma = 0.99,
            n_steps = 2048,
            batch_size = 64,
            ent_coef = 0.01)

model.learn(total_timesteps=100000)

model.save("ppo_resouce_management")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)  # Predict an action
    obs, reward, terminated, truncated = env.step(action)  # Step through the environment

    terminated = terminated[0] if isinstance(terminated, list) or isinstance(terminated, np.ndarray) else terminated
    truncated = truncated[0] if isinstance(truncated, list) or isinstance(truncated, np.ndarray) else truncated
    if isinstance(truncated, dict):
        truncated = truncated.get("TimeLimit.truncated", False)

    env.render()
    print(f"Turn: {i+1}, Action: {action}, Terminated: {terminated}, Truncated: {truncated}")

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

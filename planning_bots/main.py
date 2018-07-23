import numpy as np
from copy import deepcopy
import gym
import gym_minigrid
import planner
import minigrid_tasks
import time

grid_size = 16
environment_name ='MiniGrid-DoorKey-{0}x{0}-v0'.format(grid_size)
key = gym_minigrid.minigrid.Key(color='yellow')
locked_door = gym_minigrid.minigrid.LockedDoor(color='yellow')
goal = gym_minigrid.minigrid.Goal()
bot = planner.PlanningAgent(grid_size^2,grid_size)

##env = gym.make(environment_name)
env = gym_minigrid.envs.DoorKeyEnv(grid_size)
first_obs = env.reset()
if hasattr(env, 'mission'):
    print('Mission: %s' % env.mission)
# Create a window to render into
#renderer = env.render('human')
time.sleep(0.5)
num_experiments = 100
rez = np.zeros(num_experiments)
for i in range(num_experiments):
    tasks = [
    minigrid_tasks.Locate(key),
    minigrid_tasks.MoveTo(key),
    minigrid_tasks.Interact(key,gym_minigrid.minigrid.MiniGridEnv.Actions.pickup),
    minigrid_tasks.Locate(locked_door),
    minigrid_tasks.MoveTo(locked_door),
    minigrid_tasks.Interact(locked_door,gym_minigrid.minigrid.MiniGridEnv.Actions.toggle),
    minigrid_tasks.Locate(goal),
    minigrid_tasks.MoveTo(goal,vicinity=False)
    ]
    for obs, reward, done, info in bot.play_episode(env,first_obs,tasks):
        nothing = 0
    #     print('step=%s, reward=%.2f' % (env.step_count, reward))
    #     time.sleep(0.5)
    #     env.render('human')
    #     if renderer.window == None:
    #         break
    # if renderer.window == None:
    #         break
    rez[i] = reward
    bot.reset()
    first_obs = env.reset()
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)

import numpy as np
import gym
import gym_minigrid
import planner
import minigrid_tasks
import time
from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option('--grid-size',
                      dest="grid_size",
                      help='minigrid grid size',
                      default = 16)
    options,args = parser.parse_args()

    environment_name ='MiniGrid-DoorKey-{0}x{0}-v0'.format(options.grid_size)
    env = gym_minigrid.envs.DoorKeyEnv(options.grid_size)
    bot = planner.PlanningAgent(options.grid_size^2,options.grid_size)
    first_obs = env.reset()
    renderer=env.render('human')
    time.sleep(0.5)
    
    key = gym_minigrid.minigrid.Key(color='yellow')
    locked_door = gym_minigrid.minigrid.LockedDoor(color='yellow')
    goal = gym_minigrid.minigrid.Goal()
    
    while True:
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
        if renderer.window == None:
                break
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
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            time.sleep(0.5)
            env.render('human')
            if renderer.window == None:
                break
        bot.reset()
        first_obs = env.reset()

if __name__=='__main__':
    main()

import gym
from gym_minigrid.minigrid import *
from copy import deepcopy
from planner_core import *

class PlanningAgent:
    def __init__(self, max_steps, max_grid,obs_grid_size=AGENT_VIEW_SIZE,obs_coordinates=AGENT_OBS_COORDINATES):
        self._max_steps = max_steps
        self._max_grid = max_grid
        self._obs_grid_size = obs_grid_size
        self._obs_coordinates = obs_coordinates
        self.reset()
        
    def reset(self):
        self.grid = Grid(2*self._max_grid,2*self._max_grid)
        self.current_pos = (self._max_grid,self._max_grid)
        self.direction = None
        self.direction_vec = None
        self.current_steps = 0
        self.carrying = None
        self.memory = PlannerMemory()
        self.spatial_reasoning = PlannerSpatialReasoning(2*self._max_grid)
        self.actions_taken = []
        
    @property
    def vicinity_coords(self):
        return self.spatial_reasoning.vicinity(self.current_pos,self.direction)
    
    def update_state_from_obs(self,obs):
        self.direction = obs['direction']
        self.direction_vec = DIR_TO_VEC[self.direction]
        grid_encoded = obs['image']
        self.spatial_reasoning.add_grid_point(self.current_pos[0],self.current_pos[1],visitable=True,visited=True)
        def rotate_right(array):
            ##only in square case
            new_array = deepcopy(array)
            for i in range(len(array)):
                for j in range(len(array[0])):
                    new_array[len(array[0])-1-j][i] = array[i][j]
            return new_array

        for i in range(self.direction+1):
            grid_encoded = rotate_right(grid_encoded)
        grid = Grid.decode(grid_encoded)

        ##update spatial connections
        self.spatial_reasoning.update_rep_from_view(grid,self.current_pos,self.direction,self._obs_coordinates[self.direction])

        ##update memory
        for j in range(len(grid_encoded[0])):
            for i in range(len(grid_encoded)):
                if (i,j)==tuple(self._obs_coordinates[self.direction]):
                    if self.carrying and grid.get(i,j) is None:
                        raise ValueError('bot thinks he is carrying an object while not')
                view_x=self.current_pos[0]-self._obs_coordinates[self.direction][0]+i
                view_y=self.current_pos[1]-self._obs_coordinates[self.direction][1]+j
                if view_x>=0 and view_x<self.grid.width and view_y>=0 and view_y<self.grid.height:
                    obj = grid.get(i,j)
                    self.grid.set(view_x,view_y,obj)
                    if obj:
                        self.memory.update(obj,*[hasattr(obj,'is_open') and obj.is_open,view_x,view_y])
    
    ##TODO
    ##Interaction with environment
    ##some duplicate logic exists here with respect to environment,
    ##should refactor outside
    def update_state_with_action(self,action,fwd_cell):
        self.actions_taken.append(action)
        self.current_steps = self.current_steps + 1 
        fwd_pos = np.array(self.current_pos) + self.direction_vec
        ##assuming action is valid!
        if action == MiniGridEnv.Actions.left:
            self.direction -= 1
            if self.direction < 0:
                self.direction += 4
        elif action == MiniGridEnv.Actions.right:
            self.direction = (self.direction + 1) % 4
        elif action == MiniGridEnv.Actions.forward:
            self.current_pos = tuple(fwd_pos)
        elif action == MiniGridEnv.Actions.pickup:
            self.carrying = fwd_cell
        elif action == MiniGridEnv.Actions.drop:
            self.carrying = None
        elif action == MiniGridEnv.Actions.toggle:
            if isinstance(fwd_cell,Door):
                self.memory.update(fwd_cell,not fwd_cell.is_open,fwd_pos[0],fwd_pos[1])
            elif isinstance(fwd_cell,LockedDoor):
                #must have known to hold the right key 
                self.memory.update(fwd_cell,True,fwd_pos[0],fwd_pos[1])
                self.memory.remove(self.carrying)
                self.carrying = None
            elif isinstance(fwd_cell,Box):
                self.memory.update(fwd_cell.contains,False,fwd_pos[0],fwd_pos[1])
                self.memory.remove(fwd_cell)

    ##TODO
    ##This should be a static function within environment interaction 
    def action_from_next_point(self,next_x,next_y):
        next_direction = np.array([next_x-self.current_pos[0],next_y-self.current_pos[1]])
        left_rotation = np.array([[0,-1],[1,0]])
        if(all(next_direction==self.direction_vec)):
            return MiniGridEnv.Actions.forward
        elif next_direction.dot(self.direction_vec)==-1:
            return MiniGridEnv.Actions.right
        elif all(left_rotation.dot(next_direction)==self.direction_vec):
            return MiniGridEnv.Actions.left
        elif all(left_rotation.dot(self.direction_vec)==next_direction):
            return MiniGridEnv.Actions.right
        else:
            raise ValueError("next point not close enough to current position")

    def play_episode(self,env,first_obs,tasks):
        ##ideally tasks should be an ordered list of unordered tasks for optimality
        ##assuming an ordered list of tasks for now 
        self.reset()
        self.update_state_from_obs(first_obs)
        for task in tasks:
            while not task.is_completed(self):
                action = task.next_action(self)
                fwd_cell = self.grid.get(*(np.array(self.current_pos)+self.direction_vec))
                self.update_state_with_action(action,fwd_cell)
                obs, reward, done, info = env.step(action)
                self.update_state_from_obs(obs)
                yield obs,reward,done,info

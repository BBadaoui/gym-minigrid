import numpy as np

class SimpleTask:
    def __init__(self,obj):
        self.target_obj = obj
        self.completed = False

    def is_completed(self,planner):
        raise NotImplementedError
    
    def next_action(self,planner):
        raise NotImplementedError

class Locate(SimpleTask):
    def __init__(self,obj):
        super(Locate,self).__init__(obj)

    def is_completed(self,planner):
        return planner.memory.exists(self.target_obj)
    
    def next_action(self,planner):
        plan = planner.spatial_reasoning.explore_next(planner.current_pos,planner.direction_vec)        
        return planner.action_from_next_point(*plan[1])

class MoveTo(SimpleTask):
    def __init__(self,obj,offset=[np.zeros(2)],vicinity=True):
        ##either move to or to the vicinity of an object or any point offset from obj
        super(MoveTo,self).__init__(obj)
        self.offset_set = offset
        self.to_vicinity = vicinity
        self.action_stack = []

    def _acceptable_targets(self,planner):
        if(not planner.memory.exists(self.target_obj)):
            raise ValueError("cannot move to object if it hasn't been located")
        _,obj_x,obj_y = planner.memory.retrieve(self.target_obj)
        acceptable_targets = [tuple(np.array((obj_x,obj_y))+ offset_vec) for offset_vec in self.offset_set]
        if(self.to_vicinity):
            current_cell_xy = np.array(planner.current_pos)+planner.direction_vec
        else:
            current_cell_xy = np.array(planner.current_pos)
        return tuple(current_cell_xy),acceptable_targets
        
    def is_completed(self,planner):
        current_cell_xy,acceptable_targets = self._acceptable_targets(planner)
        return any([current_cell_xy==target for target in acceptable_targets])

    def next_action(self,planner):        
        if len(self.action_stack)>0:
            curr_index = self.action_stack.index(planner.current_pos)
            self.action_stack = self.action_stack[curr_index:]
        if len(self.action_stack)<=1:
            _,acceptable_targets = self._acceptable_targets(planner)
            plans = map(lambda target: planner.spatial_reasoning.shortest_path(*(planner.current_pos+tuple(target))),acceptable_targets)
            plans = list(filter(lambda p: p!=None, plans))
            plans_len = np.array(list(map(len, plans)))
            if(len(plans)):
                self.action_stack = plans[np.argmin(plans_len)]
            else:##should never be here 
                self.action_stack = planner.spatial_reasoning.explore_next(planner.current_pos,planner.direction_vec)
        return planner.action_from_next_point(*self.action_stack[1])

class Interact(SimpleTask):
    def __init__(self,obj,action):
        super(Interact,self).__init__(obj)
        self.interaction = action
        self.done = False

    def is_completed(self,planner):
        return self.done
    
    def next_action(self,planner):
        self.done = True
        return self.interaction


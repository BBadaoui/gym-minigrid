import gym
from gym_minigrid.minigrid import *
import pandas as pd
from  collections import namedtuple
from copy import deepcopy
import networkx as nx

ObjMemRep = namedtuple('ObjMemRep', 'ObjType ObjColor')
ValMemRep = namedtuple('ValMemRep', 'ObjState X Y')
AGENT_OBS_COORDINATES = [(0, AGENT_VIEW_SIZE//2),(AGENT_VIEW_SIZE//2, 0), (AGENT_VIEW_SIZE-1, AGENT_VIEW_SIZE//2), (AGENT_VIEW_SIZE//2, AGENT_VIEW_SIZE-1)]

class PlannerMemory:
    ##plain memory storing object locations
    ##only acts as a data store
    def __init__(self):
        self.memory = pd.DataFrame(columns=list(ObjMemRep._fields+ValMemRep._fields))
        self.memory.set_index(list(ObjMemRep._fields),inplace=True)

    def exists(self,obj):
        return ObjMemRep(obj.type,obj.color) in self.memory.index

    def update(self,obj,*args):
        self.memory.loc[ObjMemRep(obj.type,obj.color),] = ValMemRep(*args)

    def retrieve(self,obj):
        return ValMemRep(*self.memory.loc[ObjMemRep(obj.type,obj.color),].values)

    def remove(self,obj):
        self.memory.drop(ObjMemRep(obj.type,obj.color))

##TODO
##Add cues in exploration if prior info is known
##exploration is currently stateless but should not be in optimality
class PlannerSpatialReasoning:
    def __init__(self,grid_side):
        self.graph_rep = nx.Graph()
        self.grid_side = grid_side
        self.current_max_x = grid_side-1
        self.current_max_y = grid_side-1
        self.current_min_x = 1
        self.current_min_y  = 1
        self._initialize_graph_rep()

    def _coord_to_index(self,x,y):
        return x*self.grid_side+y

    def _index_to_coord(self,ind):
        x_coord = ind//self.grid_side
        y_coord = ind - x_coord*self.grid_side
        return (x_coord,y_coord)

    def _initialize_graph_rep(self):
        for i in range(self.grid_side):
            for j in range(self.grid_side):
                self.graph_rep.add_node(self._coord_to_index(i,j),visitable=True,visited=False,seen=False)
                neighbors = [np.array((i,j))+v for v in DIR_TO_VEC]
                valid_neighbors = [ n for n in neighbors if self._coord_to_index(*n)>=0 and self._coord_to_index(*n)<self.grid_side*self.grid_side]
                for n in valid_neighbors:
                    self.graph_rep.add_edge(self._coord_to_index(i,j),self._coord_to_index(*n))
        
    def _update_boundaries(self,x,y):
        self.current_max_x = min(self.current_max_x,x-1+self.grid_side//2)
        self.current_max_y = min(self.current_max_y,y-1+self.grid_side//2)
        self.current_min_x = max(self.current_min_x,x+1-self.grid_side//2)
        self.current_min_y = max(self.current_min_y,y+1-self.grid_side//2)
        ##could be more efficient but ok for now
        for i in range(self.grid_side):
            for j in range(self.grid_side):
                if i<self.current_min_x or i > self.current_max_x or j<self.current_min_y or j>self.current_max_y:
                    self.graph_rep.remove_node(self._coord_to_index(i,j))
                    self.graph_rep.add_node(self._coord_to_index(i,j),visitable=False,visited=False,seen=False)
        
    def _navigation_graph(self,end_index=None):
        diddle_graph = self.graph_rep.copy()
        if end_index:
            diddle_graph.node[end_index]['visitable'] = True
        non_visitable_points = filter(lambda pair: not diddle_graph.node[pair[0]]['visitable'],diddle_graph.nodes(data=True))
        for n in non_visitable_points:
            diddle_graph.remove_node(n[0])
        return diddle_graph

    def add_grid_point(self,x,y,**attr):
        self.graph_rep.add_node(self._coord_to_index(x,y),**attr)
        
    def add_grid_connection(self,x1,y1,x2,y2):
        self.graph_rep.add_edge(self._coord_to_index(x1,y1),self._coord_to_index(x2,y2))
        
    @staticmethod
    def vicinity(position,direction):
        coords = [DIR_TO_VEC[direction]]
        if(coords[0][0] ==0):
            ##right or left looking
            for i in [-1,1]:
                coords.append(np.array((i,0)))
        else:
            for i in [-1,1]:
                coords.append(np.array((0,i)))
        coords_shifted = [ np.array(position )+ v for v in coords]
        return [tuple(cs) for cs in coords_shifted]

    @staticmethod
    def absolute_coordinates(abs_pos,obs_coord,offset):
        return abs_pos[0]-obs_coord[0]+offset[0],abs_pos[1]-obs_coord[1]+offset[1]

    def path_action_length(self,direction_vec,path):
        length = 0
        last_direction = direction_vec
        for i in range(len(path)-1):
            new_direction = np.array(self._index_to_coord(path[i+1]))-np.array(self._index_to_coord(path[i]))
            length = length+1+np.sum(np.abs(new_direction-last_direction))
            last_direction = new_direction
        return length
                           
    def update_rep_from_view(self,obs_grid,current_pos,direction,obs_coordinates):
        height = obs_grid.height
        width = obs_grid.width
        to_visit = [tuple(obs_coordinates)]
        visited = set()
        self._update_boundaries(*current_pos)
        self.add_grid_point(current_pos[0],current_pos[1],seen=True,visited=True)
        while len(to_visit):
            next_to_visit = to_visit.pop(0)
            ntv_abs_coord = self.absolute_coordinates(current_pos,obs_coordinates,next_to_visit)
            self.add_grid_point(ntv_abs_coord[0],ntv_abs_coord[1],seen=True)
            visited.add(next_to_visit)
            neighbors = self.vicinity(next_to_visit,direction)
            for neighbor in neighbors:
                if neighbor[0]>=0 and neighbor[0]<width and neighbor[1]>=0 and neighbor[1]<height:
                    obj = obs_grid.get(neighbor[0],neighbor[1])
                    neighbor_abs_coord = self.absolute_coordinates(current_pos,obs_coordinates,neighbor)
                    if (obj and (((isinstance(obj,Door) or isinstance(obj,LockedDoor)) and obj.is_open) or isinstance(obj,Goal))) or (obj is None):
                        self.add_grid_point(neighbor_abs_coord[0],neighbor_abs_coord[1],visitable=True)##to flip back to visitable just in case
                        if neighbor not in visited and neighbor not in to_visit:
                            to_visit.append(tuple(neighbor))
                    elif isinstance(obj,Wall):
                        self.graph_rep.remove_node(self._coord_to_index(*neighbor_abs_coord))
                        self.add_grid_point(neighbor_abs_coord[0],neighbor_abs_coord[1],visitable=False,visited=False,seen=True)
                    else:
                        if neighbor_abs_coord!=current_pos:##crucial or visiting status of current pos will flip when carrying object  
                            self.add_grid_point(neighbor_abs_coord[0],neighbor_abs_coord[1],visitable=False,visited=False,seen=True)

    def shortest_path(self, x_start,y_start,x_end,y_end):
        ##starting from start and ending in the vicinity of end, intermediate points
        ##must be visitable. return None if no path is available
        end_index = self._coord_to_index(x_end,y_end)
        start_index = self._coord_to_index(x_start,y_start)
        if start_index not in self.graph_rep or end_index not in self.graph_rep:
            return None
        diddle_graph = self._navigation_graph(end_index)
        try:
            path = nx.shortest_path(diddle_graph,source=start_index,target=end_index)
            path = list(map(self._index_to_coord,path))
        except:
            path = None
        return path
    
    def explore_next(self, current_pos,direction_vec):
        x_start,y_start = current_pos
        start_index = self._coord_to_index(x_start,y_start)
        diddle_graph = self._navigation_graph()
        paths = nx.shortest_path(diddle_graph,source=start_index)
        paths_new =[(k,v) for (k,v) in paths.items() if not self.graph_rep.node[k]['seen']]
        paths_len = list(map(lambda ele: self.path_action_length(direction_vec,ele[1]),paths_new))
        if len(paths_new):
            shortest_len = np.min(paths_len)
            shortest_paths = [ p for (p,l) in zip(paths_new,paths_len) if l==shortest_len]
            shortest_paths.sort(key=lambda ele:ele[0],reverse=True)
            path_translated = list(map(self._index_to_coord,shortest_paths[0][1]))
            return path_translated

import numpy as np
import gym
from gym import spaces
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io
import itertools

'''
This is the environment currently used for continuous MWM task

'''

MAX_MARCH = 20
EPSILON = 0.1
DEG_TO_RAD = 0.0174533
WINDOW_SIZE = (300, 300) # Width x Height in pixels
MAX_LEN = np.linalg.norm(WINDOW_SIZE)

object_to_idx = {
    'wall': 1,
    'goal': 2
}
color_to_idx = {
    'invisible': 0,
    'red': 1,
    'green': 2,
    'blue': 3,
    'yellow': 4,
    'purple': 5,
    'white': 6
}

idx_to_rgb = {
    0: np.array([0.6, 0.6, 0.6]),
    1: np.array([0.9, 0, 0]),
    2: np.array([0, 0.9, 0]),
    3: np.array([0, 0, 0.9]),
    4: np.array([0.9, 0.9, 0]),
    5: np.array([0.9, 0, 0.9]),
    6: np.array([0.9, 0.9, 0.9])
}



def dist(v):
    '''calculate length of vector'''
    return np.linalg.norm(v)


class Character:
    def __init__(self, pos=[WINDOW_SIZE[0]/2, WINDOW_SIZE[1]/2], angle=0, color=4, size=10,
                fov=120*DEG_TO_RAD, num_rays=30, render_rays=True, one_hot_obs=False):
        '''
        Generate a character that can move through the window
        pos: starting position
        angle: starting angle (radians) angle always takes on values from -pi to pi
        color: color
        size: size
        fov: range of angles character can see using rays
        num_rays: fidelity of depth perception
        draw_rays: whether or not to draw the characters rays
        '''
        self.pos = pos
        self.angle = (angle + np.pi) % (2*np.pi) - np.pi
        self.color = color
        self.size = size
        self.fov = fov
        self.render_rays = render_rays
        self.num_rays = num_rays
        self.one_hot_obs = one_hot_obs        
        self.ray_max_len = np.linalg.norm(WINDOW_SIZE)+100

        
    
    def update_walls(self, vis_walls, vis_wall_refs, col_walls, col_wall_refs):
        '''
        Update references to walls. This should be called when the environment changes
        or upon initialization
        vis_walls: walls that can be seen by vision rays
        col_walls: walls that can be collided with
        '''
        self.vis_walls = list(vis_walls.values())
        self.vis_wall_refs = np.array(list(vis_wall_refs.values()))
        
        self.col_walls = list(col_walls.values())
        self.col_wall_refs = np.array(list(col_wall_refs.values()))
        self.col_wall_keys = list(col_walls.keys())
        #Walls organized into [x3, y3, x4, y4] which tell us where walls start and end
        self.wall_p = np.array(self.vis_walls).reshape(-1, 4)
        self.x3 = self.wall_p[:, 0].reshape(1, -1)
        self.y3 = self.wall_p[:, 1].reshape(1, -1)
        self.x4 = self.wall_p[:, 2].reshape(1, -1)
        self.y4 = self.wall_p[:, 3].reshape(1, -1)
        
        #Same for collidable walls for collision detection
        self.col_wall_p = np.array(self.col_walls).reshape(-1, 4)
        self.cx3 = self.col_wall_p[:, 0].reshape(1, -1)
        self.cy3 = self.col_wall_p[:, 1].reshape(1, -1)
        self.cx4 = self.col_wall_p[:, 2].reshape(1, -1)
        self.cy4 = self.col_wall_p[:, 3].reshape(1, -1)

        
    
    def update_rays(self):
        '''
        Update ray intersection computations
        This should be called every time the agent moves or turns
        https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
        '''
        fov_start = self.angle - self.fov/2
        fov_end = fov_start + self.fov
        
        ray_angles = np.linspace(fov_start, fov_end, self.num_rays, endpoint=False)
        ray_mults = np.array([np.cos(ray_angles), np.sin(ray_angles)]).T
        ray_starts = np.full((self.num_rays, 2), self.pos)
        
        x1 = ray_starts[:, 0].reshape(-1, 1)
        y1 = ray_starts[:, 1].reshape(-1, 1)
        ray_ends = ray_mults * self.ray_max_len + self.pos
        x2 = ray_ends[:, 0].reshape(-1, 1)
        y2 = ray_ends[:, 1].reshape(-1, 1)
        
        #Compute intersect metrics
        epsilon = 1e-8
        denom = (self.y4-self.y3)*(x2-x1) - (self.x4-self.x3)*(y2-y1) + 1e-8
        ua = ((self.x4-self.x3)*(y1-self.y3) - (self.y4-self.y3)*(x1-self.x3)) / denom
        ub = ((x2-x1)*(y1-self.y3) - (y2-y1)*(x1-self.x3)) / denom

        #Compute x y intersects (they ARE both supposed to be ua)
        x = x1 + ua*(x2-x1)
        y = y1 + ua*(y2-y1)

        #Compute distances to intersects
        dists = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)

        #Only keep distances with valid intersects
        mults = np.full(x.shape, 1.)
        mults[((ua < 0) | (ua > 1) | (ub < 0) | (ub > 1))] = np.inf

        #We get np.nan where lines are parallel which throws off the argmin
        # Setting parallel to inf should fix the issue
        dists[np.isnan(dists)] = np.inf

        wall_idxs = np.argmin(mults*dists, axis=1)
        wall_idxs2 = np.stack([np.arange(self.num_rays), wall_idxs])

        # inter_x = x[wall_idxs2[0], wall_idxs2[1]]
        # inter_y = y[wall_idxs2[0], wall_idxs2[1]]

        # Get distances, colors and a plottable        
        self.ray_dists = dists[wall_idxs2[0], wall_idxs2[1]]
        
        if self.one_hot_obs:
            self.ray_colors = list(map(lambda x: x.color-1, self.vis_wall_refs[wall_idxs]))
        else:
            self.ray_colors = list(map(lambda x: x.color-1, self.vis_wall_refs[wall_idxs]))
        
        ray_ends = ray_mults * self.ray_dists.reshape(-1, 1) + self.pos
        
        self.ray_plot = np.stack([ray_ends, 
                              np.array(list(self.pos)*len(ray_ends)).reshape(-1, 2)], 
                              axis=1).reshape(-1, 2)
            
    
    def draw(self, ax=None):
        '''
        draw the character
        '''
        angle1 = self.angle - 0.3
        angle2 = self.angle + 0.3
        point1 = [self.pos[0], self.pos[1]]
        point2 = [self.pos[0] - math.cos(angle1)*self.size, self.pos[1] - math.sin(angle1)*self.size]
        point3 = [self.pos[0] - math.cos(angle2)*self.size, self.pos[1] - math.sin(angle2)*self.size]

        draw_color = idx_to_rgb[self.color]
        
        poly = plt.Polygon([point1, point2, point3], fc=draw_color)
        if ax == None:
            plt.gca().add_patch(poly)
            
            if self.render_rays:
                plt.gca().plot(self.ray_plot.T[0], self.ray_plot.T[1], c='w', linewidth=1)
        else:
            ax.add_patch(poly)
            if self.render_rays:            
                ax.plot(self.ray_plot.T[0], self.ray_plot.T[1], c='w', linewidth=1)

       
        
    def move(self, speed):
        '''
        move in the faced direction with number of pixels of speed
        collision detection uses the same ray marching algorithm
        after moving, update the rays
        
        Note we have to pass the walls that can be collided with for movement
        '''
        start = self.pos
        end = [self.pos[0] + math.cos(self.angle) * speed, self.pos[1] + math.sin(self.angle) * speed]
        
        min_dist, collision_obj, collision_key = self.march(start, end)

        if collision_obj == None:
            self.pos[0] += math.cos(self.angle) * speed
            self.pos[1] += math.sin(self.angle) * speed
            
        else:
            min_dist = max(min_dist, speed * 0.1) # Make sure agent can't move backwards
            
            self.pos[0] += math.cos(self.angle) * (min_dist - speed * 0.1)
            self.pos[1] += math.sin(self.angle) * (min_dist - speed * 0.1)
        self.update_rays()

        return collision_obj, collision_key
            
            
    def march(self, start, end):
        '''
        March forward to check if there is collision with any walls
        start: starting position
        end: desired end position (may be stopped earlier if collision occurs)
        col_walls: wall objects that can detect collision
        col_wall_refs: corresponding wall object references
        '''                
        x1 = start[0]
        y1 = start[1]
        x2 = end[0]
        y2 = end[1]
        
        epsilon = 1e-8
        
        #Compute intersect metrics
        denom = (self.cy4-self.cy3)*(x2-x1) - (self.cx4-self.cx3)*(y2-y1) + epsilon
        ua = ((self.cx4-self.cx3)*(y1-self.cy3) - (self.cy4-self.cy3)*(x1-self.cx3)) / denom
        ub = ((x2-x1)*(y1-self.cy3) - (y2-y1)*(x1-self.cx3)) / denom

        #Compute x y intersects
        x = x1 + ua*(x2-x1)
        y = y1 + ua*(y2-y1)

        #Compute dists
        dists = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2).squeeze()
        
        #Only keep distances with valid intersects
        mults = np.full(x.shape, 1.)
        mults[((ua < 0) | (ua > 1) | (ub < 0) | (ub > 1))] = np.inf

        #We get np.nan where lines are parallel which throws off the argmin
        # Setting parallel to inf should fix the issue
        dists[np.isnan(dists)] = np.inf
        
        #Some intersection occurs
        if (mults != np.inf).any():
            min_idx = np.argmin(mults*dists)
            return dists[min_idx], self.col_wall_refs[min_idx], self.col_wall_keys[min_idx]
        else:
            return np.min(dists), None, None
                    
    
        
    def rotate(self, angle):
        self.angle += angle
        self.angle = (self.angle + np.pi) % (2*np.pi) - np.pi
        self.update_rays()
        
    
    def ray_obs(self, max_depth=MAX_LEN):
        '''
        Get all rays and their distances to objects
        normalize_depth: divide depth readings by value 
        '''
        
        if self.one_hot_obs:
            ray_colors = np.zeros((self.num_rays, 6))
            ray_colors[np.arange(self.num_rays), self.ray_colors] = 1
            ray_colors = ray_colors.reshape(-1)
        else:
            ray_colors = np.array(self.ray_colors) / 6
        ray_dists = np.array(self.ray_dists) / max_depth
        
        visual = np.append(ray_colors, ray_dists)
        return visual






class Box():
    def __init__(self, corner, size, color=1, is_goal=False, angle=0):
        self.size = size #this is a size 2 array for length and height
        self.color = color
        self.is_goal = is_goal
        self.corner = corner
        self.center = [self.corner[0] + self.size[0]/2, self.corner[1] + self.size[1]/2]
        self.angle = angle
            
    def draw(self, ax=None):
        rect = plt.Rectangle(self.corner, self.size[0], self.size[1], 
                             fc=idx_to_rgb[self.color], angle=self.angle)

        draw_color = idx_to_rgb[self.color]
        
        if ax == None:
            plt.gca().add_patch(rect)
        else:
            ax.add_patch(rect)
        
    def get_walls(self):
        width = self.size[0]
        height = self.size[1]
        
        a = self.angle * np.pi/180 #angle in radians
        dx1 = np.cos(a+np.pi/2)*height #x shift from UL point of corner
        dy1 = np.sin(a+np.pi/2)*height #y shift from UL point of corner
        dx2 = np.cos(a)*width #x shift from LR point of corner
        dy2 = np.sin(a)*width #y shift from LR point of corner
        
        # Corners of rectangle
        ll = (self.corner[0], self.corner[1])
        ul = (self.corner[0]+dx1, self.corner[1]+dy1)
        lr = (self.corner[0]+dx2, self.corner[1]+dy2)
        ur = (self.corner[0]+dx1+dx2, self.corner[1]+dy1+dy2)
        
        walls = [
                 [ll, ul], 
                 [ll, lr], 
                 [ul, ur], 
                 [lr, ur],
                ]
        return walls






class Ray():
    def __init__(self, start, angle, color=6):
        '''
        Ray for ray marching
        if render_march is True, then we render the sdf circles used to calculate march 
        '''
        self.start = start
        self.angle = angle
        self.color = color
        self.touched_obj = None
        self.obj_dist = MAX_LEN
        
        
    def update(self, start=None, angle=None, vis_walls=[], vis_wall_refs=[]):
        '''
        update position and angle, perform march, determine object and distance
        '''
        if start is not None:
            self.start = start
        if angle is not None:
            self.angle = angle
        self.obj_dist, self.touched_obj = self.march(vis_walls, vis_wall_refs)
        
                
    def march(self, vis_walls, vis_wall_refs):
        '''
        perform ray march, find collision with object
        '''
        end = self.start + np.array([np.cos(self.angle), np.sin(self.angle)]) * MAX_LEN
        # print(end)
        intersects = []
        for vis_wall in vis_walls:
            intersects.append(intersect(self.start, end, vis_wall[0], vis_wall[1]))
        
        min_dist = np.inf
        min_idx = 0
        for idx, inter in enumerate(intersects):
            if inter != None:
                d = dist((inter[0]-self.start[0], inter[1]-self.start[1]))
                if d < min_dist:
                    min_dist = d
                    min_idx = idx
        # print(min_dist)
        if min_idx == None:
            return min_dist, min_idx
        else:
            return min_dist, vis_wall_refs[min_idx]
    
    def draw(self, ax=None):
        rect = plt.Rectangle(self.start, self.obj_dist, 1, self.angle * 180 / np.pi, fc=idx_to_rgb[self.color])

        draw_color = idx_to_rgb[self.color]
        if ax == None:
            plt.gca().add_patch(rect)
        else:
            ax.add_patch(rect)
        # plt.scatter([self.start[0]+self.obj_dist*math.cos(self.angle)], [self.start[1]+self.obj_dist*math.sin(self.angle)])
        
            
            
            

def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)






class PlumNavEnv(gym.Env):
    metadata = {"render.modes": ['rgb_array', 'human'], 'video.frames_per_second': 24}
    def __init__(self, num_rays=12, fov=1, max_steps=500,
                rew_structure='goal', give_heading=0, verbose=0,
                world_gen_func=None, world_gen_params={}, give_dist=True,
                give_time=False, collission_penalty=0, default_reward=0,
                sub_goal_reward=0.01, goal_reward=0.1, goal_visible=True,
                task_structure=1, poster=False, auxiliary_tasks=[],
                auxiliary_task_args=[], fixed_reset=[None, None],
                character_reset_pos=0, turn_speed=0.3, move_speed=10,
                num_actions=4, num_grid_slices=5, goal_size=25, goal_corner=None,
                separate_aux_tasks=False, poster_thickness=None,
                render_character=True, wall_thickness=None,
                one_hot_obs=True, num_plums=2,
                wall_colors=1, shortcut_probability=0.):
        '''
        rew_structure: 'dist' - reward given based on distance to goal
                        'goal' - reward only given when goal reached
                        'explore' - additional reward given for each section explored
                        'explorepunish' - negative reward given when spending time near previous spots
        give_heading: whether to additionally give a distance and direction to goal
        flat: whether to give observations in a flattened state
        world_gen_func: a function can be passed to manually create a world
            using some other rules. Note that it needs to generate objects, a goal, and
            set the agent position and heading
            The character will be passed as the argument
        wall_colors: 
            1: red, red, red, red
            1.5: white, white, white, white (matching shortcut wall)
            4: red, green, blue, purple
            
        shortcut_config:
            1: 1 standard shortcut
            2: 2 shortcuts where shortcuts are farther from respective entrances
            2.5: 2 shortcuts where shortcuts are closer to respective entrances
        task_structure:
            1: visible goal, randomize position
            2: invisible goal, fixed position
            3: invisible goal, randomize position of char and goal at start
                try to reach the goal as many times as possible over fixed length episode
            4: invisible target. Agent is told which grid the target is assigned to through a one-hot
                encoding (pool divided into 5x5 grid of possible target cells). Every 40 timesteps the
                location changes
            5: same as 2 (invisible goal, fixed pos), but an additional action can be pressed to
                end the episode. If the agent is on top of the platform, they get a reward on reset
        poster:
            whether there should be a poster and on which wall [0-3]
        auxiliary_tasks: (pass as list of tasks desired)
            'null': task to output constant 0
            'euclidean_start': task to output euclidean distance travelled from start of episode
            'wall_direction': task to output relative degrees required to face a certain wall
            'terminal': task to output expected number of steps to episode end
                weighted by average episode length
        auxiliary_task_args: (pass as a list of arguments)
            'null': None (might later implement task to output n values)
            'euclidean_start': None
            'wall_direction': 1 (default), 2, 3, or 4 - pass wall that should be faced (east is 1, ccw)
        fixed_reset: (list of [np.ndarray, float]):
            First entry - fixed point to reset to
            Second entry - fixed angle to reset to
        character_reset_pos: 
            0: Default reset position - starting at the bottom and facing up
            1: Starting from anywhere in bottom right quadrant facing top half
            2: Starting from anywhere in bottom right quadrant facing any direction
            3: Starting from anywhere below shortcut wall
        goal_size: size of goal as int or array
        goal_corner: array giving x and y position of bottom left corner. If None, use default world
            generator function values
        separate_aux_tasks: !!must set to True to use new aux task learners that expect
            a list of auxiliary outputs (e.g. PPOAux, RolloutStorageAux etc.)
        poster_thickness: if given a value, set poster thickness to that value. Implemented
            to draw plots where the poster is clearer
        render_character: whether to render the character when rending the environment
        wall_thickness: if given a value, set wall thickness to that value
        one_hot_obs: wheter to set colors to be a one-hot encoded vector
            If False, colors are evenly spaced float values between 0 and 1, spaced by 7 possible values
        '''
        super(PlumNavEnv, self).__init__()

        #this gives the auxiliary tasks available and dimension of output
        available_auxiliary_tasks = {
            'null': 1,
            'euclidean_start': 1,
            'wall_direction': 1,
            'goal_dist': 1
        }
        auxiliary_task_to_idx = {
            'null': 0,
            'euclidean_start': 1,
            'wall_direction': 2,
            'goal_dist': 3
        }
        
        self.total_rewards = 0
        self.give_dist = give_dist
        self.give_heading = give_heading
        self.give_time = give_time
        self.collission_penalty = collission_penalty
        self.default_reward = default_reward
        self.sub_goal_reward = sub_goal_reward
        self.goal_reward = goal_reward
        self.rew_structure = rew_structure
        self.task_structure = task_structure
        self.verbose = verbose
        self.world_gen_func = world_gen_func
        self.world_gen_params = world_gen_params
        self.goal_visible = goal_visible # Note: not used, visibility defined by
                                         # task structure at the moment
        if type(goal_size) == list:
            self.goal_size = np.array(goal_size)
        self.goal_size = goal_size
        self.goal_corner = goal_corner
        self.poster = poster
        self.num_rays = num_rays
        self.fov = fov
        if type(fixed_reset[0]) == list:
            fixed_reset[0] = np.array(fixed_reset[0])
        self.fixed_reset = fixed_reset
        self.character_reset_pos = character_reset_pos
        self.turn_speed = turn_speed
        self.move_speed = move_speed
        self.num_actions = num_actions
        self.num_grid_slices = num_grid_slices
        self.target_grid = 0
        self.last_reward = 0
        self.poster_thickness = poster_thickness
        self.wall_thickness = wall_thickness
        self.wall_colors = wall_colors
        self.num_plums = num_plums
        self.plum_grid_generated = False
        self.shortcut_probability = shortcut_probability
        
        observation_width = num_rays
        self.ray_obs_width = num_rays
        self.one_hot_obs = one_hot_obs
        
        if self.one_hot_obs:
            observation_width = num_rays * 6
        
        if give_dist:
            observation_width += num_rays
            self.ray_obs_width = observation_width
        if give_heading:
            observation_width += 1
        if give_time:
            observation_width += 1

        self.observation_space = spaces.Box(low=0, high=1, shape=(observation_width,))
        self.action_space = spaces.Discrete(self.num_actions) #turn left, forward, right as actions
        
        self.max_steps = max_steps
        self.current_steps = 0
        
        self.character = Character(num_rays=self.num_rays, fov=self.fov, one_hot_obs=one_hot_obs)
        self.initial_character_position = self.character.pos.copy()
        self.render_character = render_character
        
        self.fig = None
        
        self.boxes = {}
        self.vis_walls = {}
        self.vis_wall_refs = {}
        self.col_walls = {}
        self.col_wall_refs = {}

        self.visited_sections = np.zeros((num_grid_slices, num_grid_slices,))


        if self.world_gen_func is None:
            self.generate_world()
        else:
            self.world_gen_func(self.character, **self.world_gen_params)
        
        self.auxiliary_tasks = []
        self.auxiliary_task_args = auxiliary_task_args
        for task in auxiliary_tasks:
            if type(task) == int:
                self.auxiliary_tasks.append(task)
            elif task not in available_auxiliary_tasks.keys():
                raise NotImplementedError('Auxiliary task {} not found. Available options are '.format(
                    task, ', '.join(available_auxiliary_tasks.keys())))
            else:
                self.auxiliary_tasks.append(auxiliary_task_to_idx[task])
        self.separate_aux_tasks = separate_aux_tasks
        
        # self.auxiliary_tasks = [auxiliary_task_to_idx[task] for task in auxiliary_tasks]
        # print('auxiliary tasks', self.auxiliary_tasks)
        
        # New Character class: need to pass walls
        self.character.update_walls(self.vis_walls, self.vis_wall_refs,
                            self.col_walls, self.col_wall_refs)
        
    def step(self, action):
        reward = self.default_reward
        collision_obj, collision_key = None, None
        done = False
        info = {}
        
        if action == 0:
            self.character.rotate(-self.turn_speed)
        if action == 1:
            collision_obj, collision_key = self.character.move(self.move_speed)
        if action == 2:
            self.character.rotate(self.turn_speed)
        if action == 3:
            pass


        elif self.rew_structure == 'explore': 
            x_grids = np.linspace(0, WINDOW_SIZE[0], self.num_grid_slices+1)
            y_grids = np.linspace(0, WINDOW_SIZE[1], self.num_grid_slices+1)
            x = self.character.pos[0]
            y = self.character.pos[1]
            x_where = np.argwhere(x < x_grids)
            y_where = np.argwhere(y < y_grids)
            if len(x_where) > 0:
                x_grid = x_where[0] - 1
            else:
                x_grid = self.num_grid_slices - 1
            if len(y_where) > 0:
                y_grid = y_where[0] - 1
            else:
                y_grid = self.num_grid_slices - 1
            
            if self.visited_sections[x_grid, y_grid] == 0:
                self.visited_sections[x_grid, y_grid] = 1
                reward += self.sub_goal_reward
                info['bonus_reward'] = self.sub_goal_reward
            else:
                info['bonus_reward'] = 0
        elif self.rew_structure == 'explorepunish':
            pos = self.character.pos
            self.visited_positions[self.visited_idx] = pos.copy()
            self.visited_idx += 1

            lim_idx = int(np.clip(self.visited_idx-10, 0, np.inf))
            
            if lim_idx > 0:
                dists = self.visited_positions[:lim_idx] - pos
                dist = np.min(np.sum(dists**2, axis=1))
                dist = np.sqrt(dist)
                
                # This pretty much goes to 0 around dist of 50, which is tuned
                #  to a move speed of 10
                punish = -np.exp(-dist/10) * self.sub_goal_reward
                reward += punish
                info['bonus_reward'] = punish
            
            
        if collision_obj != None:
            if '_' in collision_key:
                base_key = collision_key.split('_')[0]
                if 'plum' in base_key:
                    if self.verbose:
                        print('plum obtained!')
                        
                    plum_id = base_key[4]                    
                    reward = float(self.goal_reward)
                    self.generate_plum(plum_id)
                    
                    self.character.update_walls(self.vis_walls, self.vis_wall_refs,
                            self.col_walls, self.col_wall_refs)
                    self.character.update_rays()
            
            else:
                reward = float(self.collission_penalty)
                

        self.last_reward = reward
        observation = self.get_observation()
        
        auxiliary_output = self.get_auxiliary_output()
        info['auxiliary'] = auxiliary_output
        
        if self.current_steps > self.max_steps:
            done = True
        
        self.current_steps += 1
        self.total_rewards += reward
        if done and self.verbose:
            print('done, total_reward:{}'.format(self.total_rewards))
        return observation, reward, done, info
    

    def reset(self):
        """
        start_point (numpy array, optional): Set the starting point
            of the agent
        start_angle (float, optional): Set the starting angle of the 
            agent
        """
        if self.world_gen_func is None:
            self.generate_world()
        else:
            self.world_gen_func(self.character, **self.world_gen_params)
        
        
        start_point = self.fixed_reset[0]
        start_angle = self.fixed_reset[1]
        if start_point is not None:
            if type(start_point) is not np.ndarray:
                raise ValueError("start_point must be np.ndarray")
            self.character.pos = start_point.copy()
        if start_angle is not None:
            self.character.angle = start_angle
        
        self.character.update_rays()
        observation = self.get_observation()
        self.initial_character_position = self.character.pos.copy()
        self.current_steps = 0
        self.total_rewards = 0
        
        self.visited_sections = np.zeros((self.num_grid_slices, self.num_grid_slices,))
        self.visited_positions = np.full((self.max_steps + 50, 2), np.inf)
        self.visited_idx = 0

        return observation

    
    def render(self, mode='rgb_array', ax=None):
        if ax == None:
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(6,6))
            plt.xlim([0, WINDOW_SIZE[0]])
            plt.ylim([0, WINDOW_SIZE[1]])

        if self.render_character:
            self.character.draw(ax=ax)
        
        for box in self.boxes.values():
            box.draw(ax=ax)

        # image_from_plot = image_from_plot[52:380,52:390,:]
        
        # with io.BytesIO() as buff:
        #     fig.savefig(buff, format='raw')
        #     buff.seek(0)
        #     data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        # w, h = fig.canvas.get_width_height()
        # im = data.reshape((int(h), int(w), -1))
        
        if mode == 'human' and ax == None:
            plt.show()
            
        if mode == 'rgb_array':
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image_from_plot
            
            # return im
            
        
    def get_observation(self):
#         ray_depths, ray_colors = self.character.ray_obs()
#         return np.append(ray_depths, ray_colors)

        if self.give_heading > 0:
            raise NotImplementedError('Have not adjusted give_heading code')
            #tell where the goal is distance and heading
            ray_obs = self.character.ray_obs()
            goal = objects[-1]
            dist_to_goal = np.clip(dist(goal.center - self.character.pos), 0, 1000) / 1000
            heading = goal.center - self.character.pos
            heading = np.arctan2(heading[1], heading[0])

            if self.give_heading == 1:
                #only give distance to goal
                obs = np.vstack([ray_obs, [dist_to_goal, 0, 0]])
            elif self.give_heading == 2:
                #give distance and angle to goal
                obs = np.vstack([ray_obs, [dist_to_goal, heading/3.14, 0]])
            elif self.give_heading == 3:
                #give distance and angle to goal and current agent angle
                obs = np.vstack([ray_obs, [dist_to_goal, heading/3.14, self.character.angle]])
            
                        
            return np.array(obs.reshape(-1), dtype='float')
            
        else:
            obs = self.character.ray_obs()
            if not self.give_dist:
                obs = obs[:self.num_rays]

            return obs
        
        
    def get_auxiliary_output(self):
        '''Return the auxiliary output dependent on what tasks are being measured'''
                
        auxiliary_output = []
        
        for i in range(len(self.auxiliary_tasks)):
        # for i, task in self.auxiliary_tasks:
            task = self.auxiliary_tasks[i]
            if i >= len(self.auxiliary_task_args):
                aux_arg = None
            else:
                aux_arg = self.auxiliary_task_args[i]
            #0: null task - predict 0
            if task == 0:
                output = [0]
            #1: euclidean distance from start task (normalized)
            if task == 1:
                euclid_dist_start = dist(self.character.pos - self.initial_character_position)
                euclid_dist_start = euclid_dist_start / MAX_LEN
                output = [euclid_dist_start]
            #2: relative angle from wall, depending on arg passed
                # give int from 0 to 3, 0 is east and 3 is south
            if task == 2:
                if aux_arg == None:
                    wall = 0
                else:
                    wall = aux_arg
                if wall < 0 or wall > 3:
                    raise Exception('Invalid wall number passed for relative wall direction auxiliary' + \
                        'task. Must use an integer between 0 and 3, or None (defaults to 0)')
                
                if wall > 2:
                    wall = wall - 4
                wall_angle = (wall * 0.5) * np.pi
                char_2pi_angle = (self.character.angle + 2 * np.pi) % (2 * np.pi)
                
                min_rel_angle = np.min([abs(wall_angle - char_2pi_angle), 
                                        abs(wall_angle + 2*np.pi - char_2pi_angle)])
                min_rel_angle = min_rel_angle / np.pi
                output = [min_rel_angle]
            #3: distance between agent and goal
            if task == 3:
                goal_pos = self.boxes[-1].center
                char_pos = self.character.pos
                goal_dist = dist(char_pos - goal_pos)
                #normalize distance to [0, 1]
                goal_dist = goal_dist / MAX_LEN
                output = [goal_dist]
            #4-6 categorical auxiliary tasks
            #4: currently faced wall - output label 0-3 for currently faced wall
            if task == 4:
                #conversion of angle to be between 0 and 2pi, then check quadrant of angle faced
                char_2pi_angle = (self.character.angle + 2 * np.pi) % (2 * np.pi)
                if char_2pi_angle < (np.pi/2):
                    quadrant = 0
                elif char_2pi_angle < (np.pi):
                    quadrant = 1
                elif char_2pi_angle < (np.pi*3/2):
                    quadrant = 2
                else:
                    quadrant = 3
                output = [quadrant]
            #5: left/right turn to face wall
            #  give int from 0 to 3 where 0 is east and 3 is south, agent
            #  must output 0 if a left turn or 1 if a right turn is faster to face that wall
            #  categorical analog to task 2
            if task == 5:
                if aux_arg == None:
                    wall = 0
                else:
                    wall = aux_arg
                if wall < 0 or wall > 3:
                    raise Exception('Invalid wall number passed for relative wall direction auxiliary' + \
                        'task. Must use an integer between 0 and 3, or None (defaults to 0)')
                
                if wall > 2:
                    wall = wall - 4
                wall_angle = (wall * 0.5) * np.pi
                char_2pi_angle = (self.character.angle + 2 * np.pi) % (2 * np.pi)
                
                left_angle = wall_angle - char_2pi_angle
                if left_angle < 0:
                    left_angle = 2*np.pi + left_angle
                right_angle = 2*np.pi - left_angle
                if left_angle > right_angle:
                    direction = 0
                else:
                    direction = 1
                output = [direction]
            #6: current quadrant location - output label 0-3 for current quadrant
            if task == 6:
                quadrant = 0
                char_pos = self.character.pos
                if char_pos[0] > WINDOW_SIZE[0]/2:
                    quadrant += 1
                if char_pos[1] > WINDOW_SIZE[1]/2:
                    quadrant += 2
                output = [quadrant]                
            if self.separate_aux_tasks:
                auxiliary_output.append(output)
            else:
                auxiliary_output += output
        if self.separate_aux_tasks:
            return auxiliary_output
        else:
            return np.array(auxiliary_output)
    

    def generate_world(self):
        if self.wall_thickness == None:
            wall_thickness = 1
        else:
            wall_thickness = self.wall_thickness
        
        # Same white outer walls used in all tasks
        boxes, walls, wall_refs = self.make_outer_walls(thickness=wall_thickness)
        for i in range(4):
            key = f'outer{i}'
            self.boxes[key] = boxes[i]
            self.vis_walls[key] = walls[i]
            self.col_walls[key] = walls[i]
            self.vis_wall_refs[key] = wall_refs[i]
            self.col_wall_refs[key] = wall_refs[i]
            
            
        
        # Generate 4 boxescd 
        if self.task_structure in [1, 1.5, 1.7, 1.8]:
            corners = [np.array([50., 175.]),
                    np.array([175., 175.]),
                    np.array([50., 50.]),
                    np.array([175., 50.])]
            if self.task_structure in [1, 1.5]:
                colors = ['red', 'green', 'yellow', 'purple']
            elif self.task_structure in [1.7, 1.8]:
                colors = ['white', 'white', 'white', 'purple']
            
            if self.task_structure in [1.5, 1.8]:
                colors = np.random.permutation(colors)
            
            for i, corner in enumerate(corners):
                key = f'square{i}'
                box = Box(corner, np.array([75., 75.]), color=color_to_idx[colors[i]])
                self.add_box_to_walls(box, key)
                            
        # Generate 'random' slanted walls
        if self.task_structure in [2, 2.5, 2.7, 2.8]:
            corners = [
                np.array([40., 80.]), #\
                np.array([175., 120.]), #^
                np.array([175., 120.]), 
                np.array([235., 125.]), #^
                np.array([235., 125.]),
                np.array([130., 180.]), #-
                np.array([110., 265.]), #^
                np.array([110., 265.]),
                np.array([210., 265.]), #\
            ]
            angles = [ #add some decimals to make weird parallels unlikely
                -30.11,
                -80.11,
                200.11,
                -45.11,
                -89.11,
                -5.11,
                -20.11,
                -120.11,
                -40.11,
            ]
            widths = [
                80.,
                70.,
                80.,
                40.,
                100.,
                100.,
                60.,
                160., 
                75.,
            ]
            color_nums = np.cumsum([1, 2, 2, 1, 2, 1])
            
            if self.task_structure in [2, 2.5]:
                colors = ['purple', 'red', 'green', 'yellow', 'white', 'purple']
            elif self.task_structure in [2.7, 2.8]:
                colors = ['purple', 'white', 'white', 'purple', 'white', 'white']
            if self.task_structure in [2.5, 2.8]:
                colors = np.random.permutation(colors)
            for i in range(len(corners)):
                corner = corners[i]
                angle = angles[i]
                width = widths[i]
                color = colors[np.argmax(i < color_nums)]
                color = color_to_idx[color]
                height = 5
                key = f'box{i}'
                box = Box(corner, np.array([width, height]), angle=angle, color=color)
                self.add_box_to_walls(box, key)
                
        # generate standard shortcut nav environment with 1.5 wall color setting
        #  i.e., white outer and corridor walls, purple shortcut wall
        if self.task_structure == 3:
            thickness = 1
            self.boxes['corridor1'] = Box(np.array([50, 250]), np.array([75, thickness]), color=color_to_idx['white'])
            self.boxes['corridor2'] = Box(np.array([175, 250]), np.array([125, thickness]), color=color_to_idx['white'])

            self.vis_walls['corridor1'] = [[175, 250], [300, 250]]
            self.vis_walls['corridor2'] = [[50, 250], [125, 250]]
            self.vis_wall_refs['corridor1'] = self.boxes['corridor1']
            self.vis_wall_refs['corridor2'] = self.boxes['corridor2']

            self.col_walls['corridor1'] = [[175, 250], [300, 250]]
            self.col_walls['corridor2'] = [[50, 250], [125, 250]]
            self.col_wall_refs['corridor1'] = self.boxes['corridor1']
            self.col_wall_refs['corridor2'] = self.boxes['corridor2']
            
            if np.random.random() < self.shortcut_probability:
                del self.boxes['shortwall']
                del self.vis_walls['shortwall']
                del self.vis_wall_refs['shortwall']
                del self.col_walls['shortwall']
                del self.col_wall_refs['shortwall']
            else:
                self.boxes['shortwall'] = Box(np.array([125, 250]), np.array([50, thickness]), color=color_to_idx['purple'])
                self.vis_walls['shortwall'] = [[125, 250], [175, 250]]
                self.vis_wall_refs['shortwall'] = self.boxes['shortwall']
                self.col_walls['shortwall'] = [[125, 250], [175, 250]]
                self.col_wall_refs['shortwall'] = self.boxes['shortwall']
                
        # generate the plum grid after walls are made
        if not self.plum_grid_generated:
            self.generate_valid_plum_grid()
            self.plum_grid_generated = True
                

        for i in range(self.num_plums):
            self.generate_plum(i)

            
        pos_x = np.random.uniform(125, 175)
        pos_y = np.random.uniform(10, 60)
        pos = np.array([pos_x, pos_y])
        angle = np.random.uniform(np.pi/2-0.2, np.pi/2+0.2)
            
        self.character = Character(pos, angle, num_rays=self.num_rays, fov=self.fov, one_hot_obs=self.one_hot_obs)
        self.character.update_walls(self.vis_walls, self.vis_wall_refs,
                                    self.col_walls, self.col_wall_refs)
        
        
        
    def generate_valid_plum_grid(self):
        # Based on task structure, find where apples can be generated from
        
        if self.task_structure in [1, 1.5, 1.7, 1.8]:
            xgrid = np.linspace(5, 285, 140)
            grid = np.vstack(list(itertools.product(xgrid, xgrid)))
            corners = [np.array([50., 175.]),
                    np.array([175., 175.]),
                    np.array([50., 50.]),
                    np.array([175., 50.])]
            for corner in corners:
                grid = grid[~( (grid[:, 0] > corner[0]-5) & (grid[:, 0] < corner[0]+80) & \
                            (grid[:, 1] > corner[1]-5) & (grid[:, 1] < corner[1]+80) )]
            self.plum_grid = grid
            
        elif self.task_structure in [2, 2.5, 2.7, 2.8]:
            self.plum_grid = self.find_valid_plum_grid()
        
        elif self.task_structure == 3:
            xgrid = np.linspace(5, 285, 140)
            grid = np.vstack(list(itertools.product(xgrid, xgrid)))
            grid = grid[~(( grid[:, 1] > 235 ) & (grid[:, 1] < 255))]
            self.plum_grid = grid

        
    def find_valid_plum_grid(self):
        # Find valid plum points by generating a grid and looking for intersections with walls
        wall_xy = get_wall_xy(self)

        xgrid = np.linspace(5, 295, 150)
        grid = list(itertools.product(xgrid, xgrid))
        valid_grids = []

        box_size = 10
        for corner in grid:
            box = Box(np.array(corner), np.array([box_size, box_size]))
            test_walls = box.get_walls()
            add_wall = True
            for wall in test_walls:
                if check_col(wall[0], wall[1], wall_xy):
                    add_wall = False
                    break
            if add_wall:
                valid_grids.append(corner)
        return np.array(valid_grids)
    
            
    def add_box_to_walls(self, box, base_key, vis=True, col=True):
        # Add the walls of a box to dictionary of walls
        self.boxes[base_key] = box
        walls = box.get_walls()
        for j, wall in enumerate(walls):
            key = base_key + f'_{j}'
            self.vis_walls[key] = wall
            self.vis_wall_refs[key] = box
            self.col_walls[key] = wall
            self.col_wall_refs[key] = box

        
    def generate_plum(self, id=1):
        # Pick a spot from the plum grid to generate a plum on
        idx = np.random.choice(range(len(self.plum_grid)))
        corner = self.plum_grid[idx]
        plum = Box(corner, np.array([10., 10.]), color=color_to_idx['blue'])
        key = f'plum{id}'
        self.add_box_to_walls(plum, key)
        


    def reset_character_position(self):
        #Reset character position alone, used for task_structure 3
        # to reset character but not end episode after goal found
        searching = True
        goal_center = np.array(self.boxes[-1].center)
        if type(self.goal_size) == int:
            goal_size = [self.goal_size, self.goal_size]
        elif type(self.goal_size) == list:
            goal_size = self.goal_size
        else:
            goal_size = [20, 20]
        wall_thickness = 1
        
        pos = self.fixed_reset[0]
        angle = self.fixed_reset[1]
                    
        if angle == None:
            angle = np.random.uniform(0, 2*np.pi)

        self.character = Character(pos, angle, num_rays=self.num_rays, fov=self.fov,
                                   one_hot_obs=self.one_hot_obs)
        self.character.update_walls(self.vis_walls, self.vis_wall_refs,
                                    self.col_walls, self.col_wall_refs)
        self.character.update_rays()


    def make_outer_walls(self, thickness=1, with_shortwall=[True]):
        boxes = []
        y = WINDOW_SIZE[1]
        x = WINDOW_SIZE[0]

        if self.wall_colors == 4:
            walls = ['red', 'green', 'blue', 'yellow']
            wall_colors = [color_to_idx[color] for color in walls]        
        elif self.wall_colors == 1:
            walls = ['white', 'white', 'white', 'white']
            wall_colors = [color_to_idx[color] for color in walls]

        #outer walls
        boxes.append(Box(np.array([0, 0]), np.array([thickness, y]), color=wall_colors[2]))
        boxes.append(Box(np.array([0, 0]), np.array([x, thickness]), color=wall_colors[3]))
        boxes.append(Box(np.array([0, y-thickness]), np.array([x, thickness]), color=wall_colors[1]))
        boxes.append(Box(np.array([x-thickness, 0]), np.array([thickness, y]), color=wall_colors[0]))
                    
        # manually create walls here so that we don't need to check more walls than necessary
        # on intersections
        walls = [
            [[thickness, 0], [thickness, y]],
            [[0, thickness], [x, thickness]],
            [[0, y-thickness], [x, y-thickness]],
            [[x-thickness, 0], [x-thickness, y]]
        ]
        wall_refs = [
            boxes[0],
            boxes[1],
            boxes[2],
            boxes[3]
        ]
        
        return boxes, walls, wall_refs
        
    def get_walls(self, boxes):
        '''
        Get tuples of points to intersect with for rays from a list of boxes
        '''
        walls = []
        wall_refs = []
        for box in boxes:
            walls = walls + box.get_walls()
            wall_refs = wall_refs + [box] * 4
        return walls, wall_refs
    
    def change_target_grid(self):
        num_grid = self.num_grid_slices ** 2
        self.target_grid = np.random.choice(np.arange(num_grid))
        
    def test_platform(self):
        '''
        Used for task_structure 5 where agent must perform action 4 when they think
        they are on the platform. This function will test whether the agent is indeed on
        the platform
        '''
        char_pos = self.character.pos
        goal_pos = self.boxes[-1].center
        goal_size = self.boxes[-1].size
        
        pos_diff = np.abs(char_pos - goal_pos)
        return np.all(pos_diff <= goal_size)
        
        
        

    def seed(self, seed=0):
        np.random.seed(seed)





'''
Helper functions for generating plum grid for arbitrary environment
'''
def get_wall_xy(env):
    walls = list(env.col_walls.values())
    wall_p = np.array(walls).reshape(-1, 4)
    x3 = wall_p[:, 0].reshape(1, -1)
    y3 = wall_p[:, 1].reshape(1, -1)
    x4 = wall_p[:, 2].reshape(1, -1)
    y4 = wall_p[:, 3].reshape(1, -1)
    return [x3, y3, x4, y4]

def check_col(start, end, wall_xy):
    '''
    March forward to check if there is collision with any walls
    start: starting position
    end: desired end position (may be stopped earlier if collision occurs)
    col_walls: wall objects that can detect collision
    col_wall_refs: corresponding wall object references
    '''               
    x3, y3, x4, y4 = wall_xy 
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]
    
    epsilon = 1e-8
    
    #Compute intersect metrics
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1) + epsilon
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom

    #Compute x y intersects
    x = x1 + ua*(x2-x1)
    y = y1 + ua*(y2-y1)

    #Compute dists
    dists = np.sqrt((x - start[0])**2 + (y - start[1])**2).squeeze()
    
    #Only keep distances with valid intersects
    mults = np.full(x.shape, 1.)
    mults[((ua < 0) | (ua > 1) | (ub < 0) | (ub > 1))] = np.inf

    #We get np.nan where lines are parallel which throws off the argmin
    # Setting parallel to inf should fix the issue
    dists[np.isnan(dists)] = np.inf
    
    #Some intersection occurs
    if (mults != np.inf).any():
        min_idx = np.argmin(mults*dists)
        return True
    else:
        return False
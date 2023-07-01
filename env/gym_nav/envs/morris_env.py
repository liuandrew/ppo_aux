import numpy as np
import gym
from gym import spaces
import math

MAX_MARCH = 20
EPSILON = 0.1
DEG_TO_RAD = 0.0174533
WINDOW_SIZE = [300, 300]

#
# Objects
#

def generate_box(pos=None, size=[10, 25], inside_window=True, color=(255, 255, 255), is_goal=False,
                is_visible=True, is_obstacle=True):
    '''
    Generate a box with width and height drawn randomly uniformly from size[0] to size[1]
    if inside_window is True, we force the box to stay inside the window
    '''
    box_size = np.random.uniform([size[0], size[0]], [size[1], size[1]])
    if pos is None:
        if inside_window:
            pos = np.random.uniform([box_size[0], box_size[1]], 
                                     [WINDOW_SIZE[0] - box_size[0], WINDOW_SIZE[1] - box_size[1]])
        else:
            pos = np.random.uniform(WINDOW_SIZE)
            
    if inside_window:
        return Box(pos, box_size, color=color, is_goal=is_goal)
    else:
        return Box(pos, box_size, color=color, is_goal=is_goal)

def generate_circle(pos=None, radius=[10, 25], inside_window=True, color=(255, 255, 255), is_goal=False,
                   is_visible=True, is_obstacle=True):
    circ_rad = np.random.uniform(radius[0], radius[1])
    if pos is None:
        if inside_window:
            pos = np.random.uniform([circ_rad, circ_rad], [WINDOW_SIZE[0]-circ_rad, WINDOW_SIZE[1]-circ_rad])
        else:
            pos = np.random.uniform(WINDOW_SIZE)
    
    if inside_window:
        return Circle(pos, circ_rad, color=color, is_goal=is_goal)
    else:
        return Circle(pos, circ_rad, color=color, is_goal=is_goal)

def dist(v):
    '''calculate length of vector'''
    return np.linalg.norm(v)



class Circle():
    def __init__(self, center, radius, color=(255, 255, 255), is_goal=False, is_visible=True,
                is_obstacle=True):
        self.center = center
        self.radius = radius
        self.color = color
        self.is_goal = is_goal
        self.is_visible = is_visible
        self.is_obstacle = is_obstacle
        self.objects_type = 'circle'
    
    def sdf(self, p):
        return dist(self.center - p) - self.radius
    
    def draw(self):
        pygame.draw.circle(display, self.color, self.center, self.radius)
        

class Box():
    def __init__(self, center, size, color=(255, 255, 255), is_goal=False, is_visible=True,
                is_obstacle=True):
        self.center = center
        self.size = size #this is a size 2 array for length and height
        self.color = color
        self.rect = pygame.Rect(center-size, size*2)
        self.is_goal = is_goal
        self.is_visible = is_visible
        self.is_obstacle = is_obstacle
        self.objects_type = 'box'
        
    def sdf(self, p):
        offset = np.abs(p-self.center) - self.size
        unsigned_dist = dist(np.clip(offset, 0, np.inf))
        dist_inside_box = np.max(np.clip(offset, -np.inf, 0))
        return unsigned_dist + dist_inside_box
    
    def draw(self):
        pygame.draw.rect(display, self.color, self.rect)
        

#
# Character Class
#

class Ray():
    def __init__(self, start, angle, color='white', render_march=False):
        '''
        Ray for ray marching
        if render_march is True, then we render the sdf circles used to calculate march 
        '''
        self.start = start
        self.angle = angle
        self.color = color
        self.render_march = render_march
        self.touched_obj = None
        self.obj_dist = np.inf
        
        self.sdf = None
        
    def update(self, start=None, angle=None):
        '''
        update position and angle, perform march, determine object and distance
        '''
        if start is not None:
            self.start = start
        if angle is not None:
            self.angle = angle
        self.march()
        
    def march(self):
        '''
        perform ray march, find collision with object
        '''
        depth = 0
        p = self.start
        for i in range(MAX_MARCH):
            dist, obj = self.sdf(p)
            depth += dist
            
            if self.render_march:
                pygame.draw.circle(display, (255, 255, 255, 0.3), p, dist, width=1)

            if dist < EPSILON:
                self.touched_obj = obj
                self.obj_dist = depth
                return depth, obj
            else:
                p = p + np.array([np.cos(self.angle), np.sin(self.angle)]) * dist
                
        self.touched_obj = obj
        self.obj_dist = depth
        return depth, obj
    
    def draw(self):
        end = self.start + np.array([np.cos(self.angle), np.sin(self.angle)]) * self.obj_dist
        pygame.draw.line(display, self.color, self.start, end)
    
            
            
            
class Character:
    def __init__(self, pos=[WINDOW_SIZE[0]/2, WINDOW_SIZE[1]/2], angle=0, color='yellow', size=5,
                fov=120*DEG_TO_RAD, num_rays=30, render_rays=True, max_depth=424):
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
        self.ray_splits = fov / num_rays
        self.render_rays = render_rays
        self.num_rays = num_rays
        self.max_depth = max_depth
        
        self.obstacle_sdf = None
        self.visible_sdf = None
        
        self.rays = []
        
        fov_start = self.angle - self.fov/2
        for i in range(num_rays):
            self.rays.append(Ray(self.pos, fov_start + i*self.ray_splits))
            
#         print(len(self.rays))
#         print(self.num_rays)
    
    def update_sdf_funcs(self, obstacle_sdf, visible_sdf):
        '''
        Update the current held sdf functions which allow the character
        to calculate distance to objects and for rays
        '''
        self.obstacle_sdf = obstacle_sdf
        self.visible_sdf = visible_sdf
        fov_start = self.angle - self.fov/2

        for i in range(self.num_rays):
            self.rays[i].sdf = visible_sdf
            self.rays[i].update(start=self.pos, angle=fov_start + i*self.ray_splits)

    
    
    def update_rays(self):
        '''
        update the angle of the rays using own position and angle
        '''
        fov_start = self.angle - self.fov/2
        for i in range(self.num_rays):
            self.rays[i].update(start=self.pos, angle=fov_start + i*self.ray_splits)
            
            
    def draw_rays(self):
        '''
        draw the rays coming from character
        '''
        for ray in self.rays:
            ray.draw()
        
    
    def draw(self):
        '''
        draw the character
        '''
        point1 = [self.pos[0] - (math.cos(self.angle+0.3))*self.size, 
                  self.pos[1] - (math.sin(self.angle+0.3))*self.size]
        point2 = [self.pos[0] - math.cos(self.angle)*self.size*.8, self.pos[1] - math.sin(self.angle)*self.size*.8]
        point3 = [self.pos[0] - (math.cos(self.angle-0.3))*self.size, 
                  self.pos[1] - (math.sin(self.angle-0.3))*self.size]
        
        pygame.draw.polygon(
            display,
            self.color,
            [self.pos, point1, point2, point3, self.pos]
        )
        if self.render_rays:
            self.draw_rays()
        
        
    def move(self, speed=0.5):
        '''
        move in the faced direction with number of pixels of speed
        collision detection uses the same ray marching algorithm
        after moving, update the rays
        '''
        collide_with_object = self.march_collision_detection(speed)
        if collide_with_object is False:
            self.pos[0] += math.cos(self.angle) * speed
            self.pos[1] += math.sin(self.angle) * speed
            
        else:
            #collided with object, move with the given depth
            dist_to_obj = collide_with_object[0]
            self.pos[0] += math.cos(self.angle) * dist_to_obj
            self.pos[1] += math.sin(self.angle) * dist_to_obj

        self.update_rays()
        return collide_with_object
            
            
    def march_collision_detection(self, max_dist):
        '''
        perform ray march, used for collision detection. The max_dist is the speed we are
        moving at. If the max_dist exceeds the sdf (i.e., we are colliding with an object), 
        then return the distance to the collided object
        
        If sdf exceeds max_dist, then we have not collided on our path, so return False 
        (i.e., no object hit)
        
        returns:
            False - if no object collided with
            dist, obj - if colliding with an object, return the distance that we are allowed to 
                travel and the object
        '''
        depth = 0
        p = self.pos
        for i in range(MAX_MARCH):
            dist, obj = self.obstacle_sdf(p)
            
            if dist < EPSILON:    
                #we have collided before passing the requisite distance
                return depth-2*EPSILON, obj
            
            if depth + dist > max_dist:
                #we have enough room to move on the desired path
                return False
            
            else:
                #we continue the march
                depth += dist
                p = p + np.array([np.cos(self.angle), np.sin(self.angle)]) * dist
            
        return depth, obj
    
        
    def rotate(self, angle=0.05):
        self.angle += angle
        self.angle = (self.angle + np.pi) % (2*np.pi) - np.pi
        self.update_rays()
        
    
    def ray_obs(self):
        '''
        Get all rays and their distances to objects
        normalize_depth: divide depth readings by value 
        '''
        ray_colors = []
        ray_depths = []
        for ray in self.rays:
#             ray_colors.append(colors_dict[ray.touched_obj.color])
            ray_colors.append(ray.touched_obj.color)
            ray_depths.append(ray.obj_dist)
            
#         if normalize_depth:
#             ray_depths = np.array(ray_depths) / normalize_depth
#         else:
#             ray_depths = np.array(ray_depths)
            
        ray_colors = np.array(ray_colors)
#         background_colors = np.full(ray_colors.shape, 0)
        ray_depths = np.clip(ray_depths, 0, self.max_depth) / self.max_depth
        visual = (1 - ray_depths.reshape(-1, 1)) * ray_colors / 255
        
#         return ray_depths, ray_colors
        return visual


        
        
        
def randomize_location_and_angle(character, goal=None, world_size=[300, 300], sdf_func=None, sep=True):
    '''
    create a random location and start direction for the character
    noting that we do not allow spawning into objects
    sep: if set to True, we will make sure character has a minimum distance away
        from the goal that is at least half the max distance possible from goal
        to end of window
    '''

    #max distance from goal to end of window
    max_goal_sep = dist(np.max([np.array(WINDOW_SIZE) - goal.center, goal.center], axis=0)) 
    searching = True
    while searching:
        pos = np.random.uniform(WINDOW_SIZE)
        goal_sep = dist(goal.center - pos)

        if sdf_func(pos)[0] > 0 and (not sep or goal_sep > max_goal_sep / 2):
            #position is okay
            searching = False
            
    character.pos = pos
    character.angle = np.random.uniform(6.28)
#     character.pos = np.array([100, 100])
#     character.angle = 0

    character.update_rays()
    
 


#
# Nav Environments
#

    
class GeneralNav(gym.Env):
    metadata = {"render.modes": ['rgb_array', 'human'], 'video.frames_per_second': 24}
    def __init__(self, num_rays=30, max_steps=200, num_objects=5,
                rew_structure='dist', give_heading=0, verbose=0, flat=True,
                world_gen_func=None, world_gen_params={}, world_size=[300, 300], skeleton=True):
        '''
        General Nav environment which can be used to test some general pygame things and see
        that all of the object and distance detection things are working
        
        When inheriting, should make sure to change the functions
            step(), reset(), get_observation(), generate_world()
        
        rew_structure: 'dist' - reward given based on distance to goal
                        'goal' - reward only given when goal reached
        give_heading: whether to additionally give a distance and direction to goal
        flat: whether to give observations in a flattened state
        world_gen_func: a function can be passed to manually create a world
            using some other rules. Note that it needs to generate objects, a goal, and
            set the agent position and heading
            The character will be passed as the argument
            
            
            
        '''
        super(GeneralNav, self).__init__()

        if 'pygame' not in globals():
            global pygame
            import pygame

        if not skeleton:
            print('generating general')
            self.total_rewards = 0

            self.give_heading = give_heading
            self.flat = flat
            if give_heading:
                self.observation_space = spaces.Box(low=0, high=1, shape=((num_rays + 1)*3,))
            else:
        #         self.observation_space = spaces.Box(low=0, high=1, shape=(num_rays*2,), dtype=np.float)
                self.observation_space = spaces.Box(low=0, high=1, shape=(num_rays*3,))
            self.action_space = spaces.Discrete(4) #turn left, forward, right as actions

            self.max_steps = max_steps
            self.current_steps = 0

            self.character = Character(max_depth=dist(world_size))

            self.num_objects = num_objects
            self.num_rays = num_rays
            self.rew_structure = rew_structure

            self.verbose = verbose
            self.objects = []

            self.world_gen_func = world_gen_func
            self.world_gen_params = world_gen_params
            self.world_size = world_size

            if self.world_gen_func is None:
                self.generate_world()
                # randomize_location_and_angle(self.character)
            else:
                self.world_gen_func(self.character, **self.world_gen_params)
            
    def step(self, action):
        reward = -1
        collide_with_object = False
        done = False
        info = {}
        if action == 0:
            self.character.rotate(-0.1)
        if action == 1:
            collide_with_object = self.character.move(10)
        if action == 2:
            self.character.rotate(0.1)
        if action == 3:
            pass

        if self.rew_structure == 'dist':
            goal = objects[-1]
            dist_to_goal = np.clip(dist(goal.center - self.character.pos), 0, 1000) / 1000
            reward = float(-dist_to_goal)

            
        if collide_with_object is not False:
            obj = collide_with_object[1]
            if obj.is_goal:
                if self.verbose:
                    print('goal reached!')
                reward = float(100)
                done = True
            else:
#                 reward = -10
                reward = float(-1)
        
        
        observation = self.get_observation()
        
        if self.current_steps > self.max_steps:
            done = True
        
        self.current_steps += 1
        self.total_rewards += reward
        if done and self.verbose:
            print('done, total_reward:{}'.format(self.total_rewards))
        return observation, reward, done, info
    
    def get_observation(self):
#         ray_depths, ray_colors = self.character.ray_obs()
#         return np.append(ray_depths, ray_colors)

        if self.give_heading > 0:
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
            
                        
            if self.flat:
                return np.array(obs.reshape(-1), dtype='float')
            else:
                return np.array(obs, dtype='float')
            
        else:
            if self.flat:
                return np.array(self.character.ray_obs().reshape(-1), dtype='float')
            else:
                return np.array(self.character.ray_obs(), dtype='float')
    
    def reset(self):
        self.generate_world()
    
    
    def generate_walls(self):
        self.objects.append(Box(np.array([0, 0]), np.array([1, self.world_size[1]]), color=(0, 255, 0)))
        self.objects.append(Box(np.array([0, 0]), np.array([self.world_size[0], 1]), color=(0, 255, 0)))
        self.objects.append(Box(np.array([0, self.world_size[1]]), np.array([self.world_size[0], 1]), color=(0, 255, 0)))
        self.objects.append(Box(np.array([self.world_size[0], 0]), np.array([1, self.world_size[1]]), color=(0, 255, 0)))

        

    def generate_box(self, pos=None, size=[10, 25], inside_window=True, color=(255, 255, 255), is_goal=False,
                is_visible=True, is_obstacle=True):
        '''
        Generate a box with width and height drawn randomly uniformly from size[0] to size[1]
        if inside_window is True, we force the box to stay inside the window
        '''
        box_size = np.random.uniform([size[0], size[0]], [size[1], size[1]])
        if pos is None:
            if inside_window:
                pos = np.random.uniform([box_size[0], box_size[1]], 
                                         [self.world_size[0] - box_size[0], self.world_size[1] - box_size[1]])
            else:
                pos = np.random.uniform(self.world_size)

        if inside_window:
            return Box(pos, box_size, color=color, is_goal=is_goal, is_visible=is_visible, is_obstacle=is_obstacle)
        else:
            return Box(pos, box_size, color=color, is_goal=is_goal, is_visible=is_visible, is_obstacle=is_obstacle)


    def generate_circle(self, pos=None, radius=[10, 25], inside_window=True, color=(255, 255, 255), is_goal=False,
                       is_visible=True, is_obstacle=True):
        circ_rad = np.random.uniform(radius[0], radius[1])
        if pos is None:
            if inside_window:
                pos = np.random.uniform([circ_rad, circ_rad], [self.world_size[0]-circ_rad, self.world_size[1]-circ_rad])
            else:
                pos = np.random.uniform(self.world_size)

        if inside_window:
            return Circle(pos, circ_rad, color=color, is_goal=is_goal, is_visible=is_visible, is_obstacle=is_obstacle)
        else:
            return Circle(pos, circ_rad, color=color, is_goal=is_goal, is_visible=is_visible, is_obstacle=is_obstacle)
        
    
    def generate_world(self):
        '''
        World generation should end up with a list of objects as self.objects
        Should end by calling 
        
        self.generate_walls (optional to include walls)
        
        self.visible_objects, self.obstacles = self.decompose_objects(self.objects)
        obstacle_sdf = self.get_sdf_func('obstacle')
        visible_sdf = self.get_sdf_func('visible')
        self.character.update_sdf_funcs(obstacle_sdf, visible_sdf)

        '''
        boxes = [self.generate_box() for i in range(5)]
        circles = [self.generate_circle() for i in range(5)]
        self.objects = boxes + circles
        self.generate_walls()
        self.visible_objects, self.obstacles, self.all_objects = self.decompose_objects(self.objects)
        obstacle_sdf = self.get_sdf_func('obstacle')
        visible_sdf = self.get_sdf_func('visible')
        self.character.update_sdf_funcs(obstacle_sdf, visible_sdf)
    
    
    def decompose_objects(self, objects):
        '''
        Take a list of objects and turn them into a dictionary
        of usable pieces
        We need to lists, one for visible objects (which vision rays
            will use for collision detection), and obstacle objects
            (which the player uses for collision detection).
        Goals are not inherently obstacles, so when making a goal, make sure
            to decided if it should have vision/collision detection included
        '''
        type_box = type(generate_box())
        type_circle = type(generate_circle())

        visible_objects = {'box_centers': [], 'box_sizes': [], 'boxes': [],
                          'circle_centers': [], 'circle_radii': [], 'circles': []}
        obstacles = {'box_centers': [], 'box_sizes': [], 'boxes': [],
                          'circle_centers': [], 'circle_radii': [], 'circles': []}
        all_objects = {'box_centers': [], 'box_sizes': [], 'boxes': [],
                          'circle_centers': [], 'circle_radii': [], 'circles': []}

        for obj in objects:
            if type(obj) == type_box:
                all_objects['box_centers'].append(obj.center)
                all_objects['box_sizes'].append(obj.size)
                all_objects['boxes'].append(obj)

                if obj.is_visible:
                    visible_objects['box_centers'].append(obj.center)
                    visible_objects['box_sizes'].append(obj.size)
                    visible_objects['boxes'].append(obj)
                if obj.is_obstacle:
                    obstacles['box_centers'].append(obj.center)
                    obstacles['box_sizes'].append(obj.size)
                    obstacles['boxes'].append(obj)

            elif type(obj) == type_circle:
                all_objects['circle_centers'].append(obj.center)
                all_objects['circle_radii'].append(obj.radius)
                all_objects['circles'].append(obj)

                if obj.is_visible:
                    visible_objects['circle_centers'].append(obj.center)
                    visible_objects['circle_radii'].append(obj.radius)
                    visible_objects['circles'].append(obj)
                if obj.is_obstacle:
                    obstacles['circle_centers'].append(obj.center)
                    obstacles['circle_radii'].append(obj.radius)
                    obstacles['circles'].append(obj)
            else:
                raise Exception('Invalid object not of type box or circle in objects')
        for key in visible_objects:
            if key not in ['boxes', 'circles']:
                visible_objects[key] = np.array(visible_objects[key])
        for key in obstacles:
            if key not in ['boxes', 'circles']:
                obstacles[key] = np.array(obstacles[key])
            
        return visible_objects, obstacles, all_objects
    
    
    def box_sdfs(self, p, objects):
        '''
        compute all the sdf functions for boxes using global variables
        box_centers
        box_sizes
        both are m x 2 arrays with each row representing a box
        '''
        box_centers = objects['box_centers']
        box_sizes = objects['box_sizes']
        if len(box_centers) > 0:
            offset = np.abs(p - box_centers) - box_sizes
            unsigned_dist = np.linalg.norm(np.clip(offset, 0, np.inf), axis=1)
            dist_inside_box = np.max(np.clip(offset, -np.inf, 0), axis=1)
            dists = unsigned_dist + dist_inside_box
            return dists
        else:
            return np.array([])
        
    def circle_sdfs(self, p, objects):
        '''
        compute all the sdf functions for circles using global variables
        circle_centers (m x 2 array)
        circle_radii   (m x 1 array)
        both arrays are 2 dimensional
        '''
        circle_centers = objects['circle_centers']
        circle_radii = objects['circle_radii']

        if len(circle_centers) > 0:
            return np.linalg.norm((circle_centers - p), axis=1) - circle_radii
        else:
            return np.array([])
        
    
    def scene_sdf(self, p, objects):
        '''
        Perform an sdf on the objects passed
        The objects passed should be those generated by the decompose_objects
            function
        '''
        box_dists = self.box_sdfs(p, objects)
        circle_dists = self.circle_sdfs(p, objects)

        dists = np.append(box_dists, circle_dists)
        min_dist = np.min(dists)
        obj_index = np.argmin(dists)
        obj_select_list = objects['boxes'] + objects['circles']
        
        return np.min(dists), obj_select_list[obj_index]

    
    def get_sdf_func(self, typ='visible'):
        '''
        Get an sdf function to be passed down to the character and rays
        '''
        if typ == 'visible':
            def sdf(p):
                return self.scene_sdf(p, self.visible_objects)
            return sdf
        elif typ == 'obstacle':
            def sdf(p):
                return self.scene_sdf(p, self.obstacles)
            return sdf
        elif typ == 'all':
            def sdf(p):
                return self.scene_sdf(p, self.all_objects)
            return sdf
                
        else:
            raise Exception('Invalid object type for sdf generator')
    
    
    def render(self, mode='rgb_array'):
        '''
        Render out the scene using pygame. If mode=='human', render it to the screen
        Otherwise only return an rgb_array of pixel colors using pygame
        '''
        if 'screen' not in globals():
            pygame.init()
            if mode == 'human':
                globals()['screen'] = pygame.display.set_mode([self.world_size[0], self.world_size[1] + 10])
            globals()['display'] = pygame.Surface([self.world_size[0], self.world_size[1] + 10])

        display.fill((0, 0, 0))
        
        self.character.draw()
        self.draw_character_view()
        for obj in self.objects:
            obj.draw()

        if mode == 'human':
            screen.blit(display, (0, 0))
            pygame.display.update()
            
        if mode == 'rgb_array':
            return pygame.surfarray.pixels3d(display)
        
    def draw_character_view(self):
        length = self.world_size[0] / self.num_rays
        colors = self.character.ray_obs() * 255
        for i in range(self.num_rays):
            rect = pygame.Rect([i * length, 300, length, 10])
            pygame.draw.rect(display, colors[i], rect)

    
class MorrisNav(GeneralNav):
    metadata = {"render.modes": ['rgb_array', 'human'], 'video.frames_per_second': 24}
    def __init__(self, num_rays=30, max_steps=None, give_heading=0, verbose=0,
                platform_visible=False, ep_struct=1, platform_size=10, world_size=[300, 300],
                platform_randomization=1, platform_randomization_spread=20,
                global_cues=1, platform_fixed_duration=10, character_sep=False, 
                reward_shift=0, platform_reward=100):
        '''
        rew_structure: 'dist' - reward given based on distance to goal
                        'goal' - reward only given when goal reached
        give_heading: whether to additionally give a distance and direction to goal
        platform_visible: whether platform should be made visible
        max_steps: how many steps an episode should last - default depends on episode structure
        !!
        ep_struct: important variable about what kind of test we will perform
            1: the platform position does not reset between episodes, episodes are 200 steps max
            2: the platform position resets each episode, and if the agent stays on a platform
                for a while, rewards will be given and position reset 
                (implement later)
            3: agent must stay on platform for 5 timesteps before reward is given and
                episode resets
            4: agent must explicitly perform an action to say when it is on the platform (not implemented)
        !!
        plaform_randomization: how the platform position will be randomized
            1: fixed positions in one of four quadrants
            2: some spot randomized close to the quadrant spots (given by platform_randomization_spread)
            3: uniform random
        global_cues: what global cues will be provided to the agent (not implemented)
            1: all walls colored differently
            2: all walls white with a "poster" hanging up
        
        observation space: each ray gives an rgb value depending on distance from object, so this
            gives num_rays*3 observations. Additionally a flag will be on/off depending on whether
            the agent is currently on a platform
            
        platform_fixed_time: once the agent reaches the plaform, it will not longer be allowed to 
            move forward, only rotate (mimic the "stay on platform and look around" phase). This controls
            how many timesteps this happens for

        character_sep: whether character should be forced to a randomized position far from platform

        reward_shift: value the reward should be centered on (e.g., -1 will make every time step give
         -1 reward, vs. 0 where the goal gives 1 reward)
        '''
        super(MorrisNav, self).__init__()

        if 'pygame' not in globals():
            global pygame
            import pygame


        self.total_rewards = 0
        
        self.give_heading = give_heading
        self.ep_struct = ep_struct
        self.platform_visible = platform_visible
        self.platform_size = platform_size
        self.platform_randomization = platform_randomization
        self.platform_randomization_spread = platform_randomization_spread
        self.world_size = world_size
        self.global_cues = global_cues
        self.platform_fixed_duration = platform_fixed_duration
        self.character_sep = character_sep
        self.reward_shift = reward_shift
        self.platform_reward = platform_reward

        self.num_rays = num_rays
        
        if give_heading:
            self.observation_space = spaces.Box(low=0, high=1, shape=((num_rays + 1)*3 + 1,))
        else:
    #         self.observation_space = spaces.Box(low=0, high=1, shape=(num_rays*2,), dtype=np.float)
            self.observation_space = spaces.Box(low=0, high=1, shape=(num_rays*3 + 1,))
        
        self.action_space = spaces.Discrete(4) #turn left, forward, right as actions
        
        if max_steps is None:
            if ep_struct == 1 or ep_struct == 3:
                self.max_steps = 200
            if ep_struct == 2:
                self.max_steps = 1000
        else:
            self.max_steps = max_steps

        if max_steps is not None:
            self.max_steps = max_steps

        self.current_steps = 0
        self.duration_on_platform = 0
        self.on_platform = False
        
        self.character = Character(max_depth=dist(world_size))
        
        self.verbose = verbose
        self.objects = []
        self.goal = None
        
        self.generate_world()
        
        
    def generate_world(self):
        self.objects = []
        if self.platform_randomization < 3:
            quadrant_locations = np.array([self.world_size[0] / 4, self.world_size[1] / 4])
            multipliers = np.array([1, 3])
            randoms = np.random.choice(2, size=(2))
            multipliers = multipliers[randoms] #get how much the x/y values should be multiplied by
            
            pos = quadrant_locations * multipliers
            
            if self.platform_randomization == 2:
                #add a spread to the platform location from quadrant position
                pos += np.random.uniform(-self.platform_randomization_spread, self.platform_randomization_spread,
                                         size=(2))
        elif self.platform_randomization == 3:
            pos = None
            
            
        platform = self.generate_box(pos=pos, size=[self.platform_size, self.platform_size], is_goal=True,
                         is_visible=self.platform_visible, is_obstacle=False)
        self.objects.append(platform)
        self.goal = platform
        self.generate_walls()
        
        self.visible_objects, self.obstacles, self.all_objects = self.decompose_objects(self.objects)
        obstacle_sdf = self.get_sdf_func('obstacle')
        visible_sdf = self.get_sdf_func('visible')
        self.character.update_sdf_funcs(obstacle_sdf, visible_sdf)
        
        
    def generate_walls(self):
        if self.global_cues == 1:
            self.objects.append(Box(np.array([0, 0]), np.array([1, self.world_size[1]]), color=(255, 0, 0)))
            self.objects.append(Box(np.array([0, 0]), np.array([self.world_size[0], 1]), color=(0, 255, 0)))
            self.objects.append(Box(np.array([0, self.world_size[1]]), np.array([self.world_size[0], 1]), color=(0, 0, 255)))
            self.objects.append(Box(np.array([self.world_size[0], 0]), np.array([1, self.world_size[1]]), color=(255, 255, 255)))
        elif self.global_cues == 2:
            pass

        

    def step(self, action):
        reward = 0
        collide_with_object = False
        done = False
        info = {}
        
        if action == 0:
            self.character.rotate(-0.1)
        if action == 1:
            if self.ep_struct >= 3 or not self.on_platform:
                #if on the platform, must now be fixed onto it
                collide_with_object = self.character.move(3)
        if action == 2:
            self.character.rotate(0.1)
        if action == 3:
            pass
            
        
        # if collide_with_object is not False:
        #     obj = collide_with_object[1]
                    
        if self.on_platform:
            self.duration_on_platform += 1
            if self.ep_struct <= 2:
                reward = self.platform_reward
            if self.duration_on_platform >= self.platform_fixed_duration:
                if self.ep_struct == 1:
                    #resetting episode in ep_struct 1
                    done = True
                elif self.ep_struct == 2:
                    #only reset position in ep_struct 2, episode concludes at end of time
                    self.reset_character()
                elif self.ep_struct == 3:
                    reward = self.platform_reward
                    done = True

        observation = self.get_observation()
        
        if self.current_steps > self.max_steps:
            done = True
                
        reward += self.reward_shift
        self.current_steps += 1
        self.total_rewards += reward
        if done and self.verbose:
            print('done, total_reward:{}'.format(self.total_rewards))
        return observation, reward, done, info
    
    def get_observation(self):
        '''
        Get observation reading the colors of the rays and also whether on platform or not
        '''
#         ray_depths, ray_colors = self.character.ray_obs()
#         return np.append(ray_depths, ray_colors)
        self.on_platform = np.all(np.abs(self.goal.center - self.character.pos) < self.goal.size)
        
        if self.give_heading > 0:
            raise Exception('Not implemented a give_heading > 0 condition for observation')
            #tell where the goal is distance and heading
            ray_obs = self.character.ray_obs()
            goal = self.goal
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
            
            #!! Add code to show when on top of platform
                        
            if self.flat:
                return np.array(obs.reshape(-1), dtype='float')
            else:
                return np.array(obs, dtype='float')
            
        else:
            obs = np.array(self.character.ray_obs().reshape(-1), dtype='float')
            obs = np.append(obs, np.array([self.on_platform * 1]))
            return obs
    
    def reset(self):
        if self.ep_struct == 2:
            self.generate_world()
        
        observation = self.get_observation()
        self.current_steps = 0
        self.total_rewards = 0
        self.on_platform = False
        self.duration_on_platform = 0
        randomize_location_and_angle(self.character, self.goal, self.world_size, self.get_sdf_func('all'), self.character_sep)
        return observation
    
    def reset_character(self):
        '''
        Reset position of the character, used for ep_struct 2
        '''
        self.on_platform = False
        self.duration_on_platform = 0
        randomize_location_and_angle(self.character, self.goal, self.world_size, self.get_sdf_func('all'), self.character_sep)
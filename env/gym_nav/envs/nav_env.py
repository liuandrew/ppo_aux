import numpy as np
import gym
from gym import spaces
import math

"""
NOTE: This code is older using ray marching for vision lines which is slow.
Use nav_env_flat for continuous navigation environment instead
"""

MAX_MARCH = 20
EPSILON = 0.1
DEG_TO_RAD = 0.0174533
WINDOW_SIZE = (200, 300) # Width x Height in pixels

def generate_box(pos=None, size=[10, 25], inside_window=True, color=(255, 255, 255), is_goal=False):
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

def generate_circle(pos=None, radius=[10, 25], inside_window=True, color=(255, 255, 255), is_goal=False):
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

def generate_boxes(num_boxes=5, size=[10, 25], is_goal=False, inside_window=True, color=(255, 255, 255)):
    centers = []
    sizes = []
    boxes = []
    for i in range(num_boxes):
        box = generate_box(size=size, color=color, is_goal=is_goal, inside_window=inside_window)
        centers.append(box.center)
        sizes.append(box.size)
        boxes.append(box)
        
    centers = np.array(centers)
    sizes = np.array(sizes)
    return boxes, centers, sizes

def generate_circles(num_circles=5, radius=[10, 25], is_goal=False, inside_window=True, color=(255, 255, 255)):
    centers = []
    radii = []
    circles = []
    for i in range(num_circles):
        circle = generate_circle(radius=radius, color=color, is_goal=is_goal, inside_window=inside_window)
        centers.append(circle.center)
        radii.append(circle.radius)
        circles.append(circle)
        
    centers = np.array(centers)
    radii = np.array(radii)
    return circles, centers, radii


def reset_objects():
    '''reset global object lists to be populated'''
    items = ['boxes', 'box_centers', 'box_sizes', 'circles', 'circle_centers',
            'circle_radii', 'objects']
    
    for item in items:
        globals()[item] = []
    

def add_box(box):
    '''add box to global boxes object for computation'''
    globals()['boxes'].append(box)
    
    if len(globals()['box_centers']) > 0:
        globals()['box_centers'] = np.vstack([box_centers, np.array([box.center])])
        globals()['box_sizes'] = np.vstack([box_sizes, np.array([box.size])])
    else:
        globals()['box_centers'] = np.array([box.center])
        globals()['box_sizes'] = np.array([box.size])
    globals()['objects'] = globals()['boxes'] + globals()['circles']
    
    
def add_circle(circle):
    '''add circle to global circles object for computation'''
    globals()['circles'].append(circle)
    if len(globals()['circle_centers']) > 0:
        globals()['circle_centers'] = np.vstack([circle_centers, np.array([circle.center])])
        globals()['circle_radii'] = np.vstack([circle_radii, np.array([circle.radius])])
    else:
        globals()['circle_centers'] = np.array([circle.center])
        globals()['circle_radii'] = np.array([circle.radius])

    globals()['objects'] = globals()['boxes'] + globals()['circles']
    
    
def add_walls():
    add_box(Box(np.array([0, 0]), np.array([1, WINDOW_SIZE[1]]), color=(0, 255, 0)))
    add_box(Box(np.array([0, 0]), np.array([WINDOW_SIZE[0], 1]), color=(0, 255, 0)))
    add_box(Box(np.array([0, WINDOW_SIZE[1]]), np.array([WINDOW_SIZE[0], 1]), color=(0, 255, 0)))
    add_box(Box(np.array([WINDOW_SIZE[0], 0]), np.array([1, WINDOW_SIZE[1]]), color=(0, 255, 0)))

    

def spaced_random_pos(sep=5):
    '''
    Find a spot that has a minimum separation from other objects in the scene
    '''
    while True:
        pos = np.random.uniform(WINDOW_SIZE)
        if scene_sdf(pos)[0] > sep:
            return pos



def generate_world(num_objects=5, min_goal_sep=15, color=(0, 255, 0)):
    reset_objects()
    '''generate obstacles'''
    boxes, box_centers, box_sizes = generate_boxes(num_objects, inside_window=False, color=color)
    circles, circle_centers, circle_radii = generate_circles(num_objects, inside_window=False, color=color)
    
    globals()['boxes'] = boxes
    globals()['box_centers'] = box_centers
    globals()['box_sizes'] = box_sizes
    globals()['circles'] = circles
    globals()['circle_centers'] = circle_centers
    globals()['circle_radii'] = circle_radii
    globals()['objects'] = boxes + circles
    
    #create walls around screen:
    add_walls()

    #create a goal, require it to be at least 30 units away from player
    searching = True
    while searching:
        pos = np.random.uniform(WINDOW_SIZE)
        if scene_sdf(pos)[0] > min_goal_sep:
            #position is okay
            searching = False
            
#     pos = np.array([500, 500])
    goal = generate_box(pos=pos, size=[15, 15], is_goal=True, color=(255, 0, 0))
    globals()['goal'] = goal
    add_box(goal)


    
def block_view_world(character, block_size=25, randomize_heading=0):
    '''
    Create a setting where the goal is perfectly blocked by a block
    randomize_heading:
        0 - always fixed
        1 - randomize headings but point agent in the right direction
        2 - randomize headings and point agent in random direction
    '''
#     print('call block view world')
    
    reset_objects()
    
    boxes, box_centers, box_sizes = generate_boxes(0)
    circles, circle_centers, circle_radii = generate_circles(0)
    
    #add a single block in the center of the screen
    add_box(Box(np.array([WINDOW_SIZE[0]/2, WINDOW_SIZE[1]/2]),
               np.array([block_size, block_size]), color=(0, 255, 0)))
    add_walls()
    
    base_size = 15
    base_x = 150
    base_y = 100
    base_radius = 88
    if randomize_heading > 0:
        angle = np.random.uniform(6.28)
        x = np.cos(angle) * base_radius
        y = np.sin(angle) * base_radius
        goal = Box(np.array([x + base_x, y + base_y]), np.array([base_size, base_size]), 
            is_goal=True, color=(255, 0, 0))
        globals()['goal'] = goal
        add_box(goal)
        
        angle2 = angle + 3.14
        x = np.cos(angle2) * base_radius
        y = np.sin(angle2) * base_radius
        character.pos = np.array([x + base_x, y + base_y])
        
        if randomize_heading > 1:
            character.angle = np.random.uniform(6.28)
        else:
            character.angle = angle
            
        character.update_rays()
        
    else:
        #add the goal
        goal = Box(np.array([WINDOW_SIZE[0] - 50, WINDOW_SIZE[1]/2]),
                   np.array([base_size, base_size]),
                   is_goal=True, color=(255, 0, 0))
        globals()['goal'] = goal
        add_box(goal)

        #set the agent position
        character.pos = np.array([50, WINDOW_SIZE[1]/2])
        character.angle = 0
        
        character.update_rays()

def dist(v):
    '''calculate length of vector'''
    return np.linalg.norm(v)

def scene_sdf(p):
#     closest_sdf = np.inf
#     closest = None
#     for obj in objects:
#         obj.draw()
        
#         sdf = obj.sdf(p)
#         if sdf < closest_sdf:
#             closest_sdf = sdf
#             closest = obj
#     return closest_sdf, closest
    box_dists = box_sdfs(p)
    circle_dists = circle_sdfs(p)
    
    dists = np.append(box_dists, circle_dists)
    min_dist = np.min(dists)
    obj_index = np.argmin(dists)
    
    #find which object sdf was closest to
    
    
    return np.min(dists), (boxes + circles)[obj_index]
#     return box_dists, circle_dists


def box_sdfs(p):
    '''
    compute all the sdf functions for boxes using global variables
    box_centers
    box_sizes
    both are m x 2 arrays with each row representing a box
    '''
    if len(box_centers) > 0:
        offset = np.abs(p - box_centers) - box_sizes
        unsigned_dist = np.linalg.norm(np.clip(offset, 0, np.inf), axis=1)
        dist_inside_box = np.max(np.clip(offset, -np.inf, 0), axis=1)
        dists = unsigned_dist + dist_inside_box
        return dists
    else:
        return np.array([])


def circle_sdfs(p):
    '''
    compute all the sdf functions for circles using global variables
    circle_centers (m x 2 array)
    circle_radii   (m x 1 array)
    both arrays are 2 dimensional
    '''
    if len(circle_centers) > 0:
        return np.linalg.norm((circle_centers - p), axis=1) - circle_radii
    else:
        return np.array([])
    

class Circle():
    def __init__(self, center, radius, color=(255, 255, 255), is_goal=False):
        self.center = center
        self.radius = radius
        self.color = color
        self.is_goal = is_goal
    
    def sdf(self, p):
        return dist(self.center - p) - self.radius
    
    def draw(self):
        pygame.draw.circle(display, self.color, self.center, self.radius)
        

class Box():
    def __init__(self, center, size, color=(255, 255, 255), is_goal=False):
        self.center = center
        self.size = size #this is a size 2 array for length and height
        self.color = color
        self.rect = pygame.Rect(center-size, size*2)
        self.is_goal = is_goal
        
    def sdf(self, p):
        offset = np.abs(p-self.center) - self.size
        unsigned_dist = dist(np.clip(offset, 0, np.inf))
        dist_inside_box = np.max(np.clip(offset, -np.inf, 0))
        return unsigned_dist + dist_inside_box
    
    def draw(self):
        pygame.draw.rect(display, self.color, self.rect)
        
        
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
            dist, obj = scene_sdf(p)
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
                fov=120*DEG_TO_RAD, num_rays=30, render_rays=True):
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
        
        self.rays = []
        
        fov_start = self.angle - self.fov/2
        for i in range(num_rays):
            self.rays.append(Ray(self.pos, fov_start + i*self.ray_splits))
            
#         print(len(self.rays))
#         print(self.num_rays)
    
    
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
            dist, obj = scene_sdf(p)
            
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
        
    
    def ray_obs(self, max_depth=1000):
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
        ray_depths = np.clip(ray_depths, 0, max_depth) / 1000
        visual = (1 - ray_depths.reshape(-1, 1)) * ray_colors / 255
        
#         return ray_depths, ray_colors
        return visual
        
        
        
        
def randomize_location_and_angle(character, sep=True):
    '''
    create a random location and start direction for the character
    noting that we do not allow spawning into objects
    sep: if set to True, we will make sure character has a minimum distance away
        from the goal that is at least half the max distance possible from goal
        to end of window
    '''

    #max distance from goal to end of window
    max_goal_sep = dist(np.max([np.array(WINDOW_SIZE) - goal.center, goal.center], axis=0)) 
    sep = True
    searching = True
    while searching:
        pos = np.random.uniform(WINDOW_SIZE)
        goal_sep = dist(globals()['goal'].center - pos)

        if scene_sdf(pos)[0] > 0 and (not sep or goal_sep > max_goal_sep / 2):
            #position is okay
            searching = False
            
    character.pos = pos
    character.angle = np.random.uniform(6.28)
#     character.pos = np.array([100, 100])
#     character.angle = 0

    character.update_rays()


class NavEnv(gym.Env):
    metadata = {"render.modes": ['rgb_array', 'human'], 'video.frames_per_second': 24}
    def __init__(self, num_rays=30, max_steps=200, num_objects=5,
                rew_structure='dist', give_heading=0, verbose=0, flat=True,
                world_gen_func=None, world_gen_params={}):
        '''
        rew_structure: 'dist' - reward given based on distance to goal
                        'goal' - reward only given when goal reached
        give_heading: whether to additionally give a distance and direction to goal
        flat: whether to give observations in a flattened state
        world_gen_func: a function can be passed to manually create a world
            using some other rules. Note that it needs to generate objects, a goal, and
            set the agent position and heading
            The character will be passed as the argument
        '''
        super(NavEnv, self).__init__()

        if 'pygame' not in globals():
            global pygame
            import pygame


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
        
        self.character = Character()
        
        self.num_objects = num_objects
        
        self.rew_structure = rew_structure
        
        self.verbose = verbose
        
        self.world_gen_func = world_gen_func
        self.world_gen_params = world_gen_params
        
        if self.world_gen_func is None:
            generate_world(self.num_objects)
            randomize_location_and_angle(self.character)
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
    
    def reset(self):
        if self.world_gen_func is None:
            generate_world(self.num_objects)
            randomize_location_and_angle(self.character)
        else:
            self.world_gen_func(self.character, **self.world_gen_params)
        
        observation = self.get_observation()
        self.current_steps = 0
        self.total_rewards = 0
        return observation
    
    def render(self, mode='rgb_array'):
        if 'screen' not in globals() or str(screen) == '<Surface(Dead Display)>':
            pygame.init()
            if mode == 'human':
                globals()['screen'] = pygame.display.set_mode(WINDOW_SIZE)
            globals()['display'] = pygame.Surface(WINDOW_SIZE)

        display.fill((0, 0, 0))
        
        self.character.draw()
        for obj in objects:
            obj.draw()

        if mode == 'human':
            screen.blit(display, (0, 0))
            pygame.display.update()
            
        if mode == 'rgb_array':
            return pygame.surfarray.pixels3d(display)
        
    def close(self):
        pygame.quit()
        
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
        


        
    
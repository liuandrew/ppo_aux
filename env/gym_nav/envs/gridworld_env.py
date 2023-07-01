import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import spaces

class GridworldNav(gym.Env):
    metadata = {"render.modes": ['rgb_array', 'human'], 'video.frames_per_second': 24}
    def __init__(self, view_width=2, max_steps=200, give_direction=False, world_gen_func={}, 
                world_size=20, give_dist=False, give_time=False, num_obstacles=0, goal_size=1,
                skeleton=True, goal_reward=1, reward_shaping=0, sub_goal_reward=0.01,
                wall_colors=1, task_structure=1, goal_wiggle=0, poster=False):
        '''
        General gridworld with 2d rays of vision. Agent gets to rotate or move forward

        view_width: how many rows to the left and right agent is able to see
        give_direction: include in observation currently faced direction
        give_dist: whether to include distances to objects seen in observation
        world_size: length and width of world
        num_obstacles: number of randomly generated obstacles
        goal_size: how big goal should be in length an width
        goal_reward: amount of reward earned in reaching goal

        reward_shaping: how reward should be given
            0: only when goal is reached
            1: always give additional reward inv prop to dist to goal
            2: when goal is in sight, give additional reward inv proportional to dist to goal
            3: when goal has been seen once, give additional reward inv prop
                to dist to goal
            (for 1-3, also give reward when goal reached)
        sub_goal_reward: max reward given by sub-task (from reward shaping)

        wall_colors: how many colors the walls should take (1, 2, 2.5, 3, 4)
            1: red, red, red, red
            2: red, green, red, green
                2.5 (alt configuration): red, red, green, green
            3: red, green, red, blue
                3.5 (alt configuration): red, red, green, blue
            4: red, green, blue, purple

        task_structure: the exact type of task given
            1: visible goal, resetting position every episode
            2: invisible goal, fixed position for goal

        goal_wiggle: whether goal should be moved from its fixed position randomly

        poster: set to a number to choose a position for the "poster"
        '''
        super(GridworldNav, self).__init__()
        
        self.object_to_idx = {
            'wall': 1,
            'goal': 2
        }
        self.color_to_idx = {
            'invisible': 0,
            'red': 1,
            'green': 2,
            'blue': 3,
            'yellow': 4,
            'purple': 5,
            'white': 6
        }
        self.idx_to_rgb = {
            1: np.array([0.9, 0, 0]),
            2: np.array([0, 0.9, 0]),
            3: np.array([0, 0, 0.9]),
            4: np.array([0.9, 0.9, 0]),
            5: np.array([0.9, 0, 0.9]),
            6: np.array([0.9, 0.9, 0.9])
        }
        self.action_keys = {
            0: 'left',
            1: 'forward',
            2: 'right',
            3: 'nothing'
        }
        
        self.current_steps = 0
        
        #generate the character icon
        self.char_icon = np.zeros([15, 15, 3])
        self.char_icon[2:14, 2:4] = [1, 1, 0]
        self.char_icon[3:13, 4:6] = [1, 1, 0]
        self.char_icon[4:12, 6:8] = [1, 1, 0]
        self.char_icon[5:11, 8:10] = [1, 1, 0]
        self.char_icon[6:10, 10:12] = [1, 1, 0]
        self.char_icon[7:9, 12:14] = [1, 1, 0]
        
        # if skeleton is False:
        #convention of world:
        # first index is y position (down is +1, up is -1)
        # second index is x position (left is -1, right is +1)
        self.world_size = [world_size, world_size]
        self.objects = np.zeros(self.world_size)
        self.visible = np.zeros(self.world_size)
        self.obstacles = np.zeros(self.world_size)
        self.num_obstacles = num_obstacles
        self.goal_size = goal_size
        self.goal_reward = goal_reward
        self.sub_goal_reward = sub_goal_reward
        self.reward_shaping = reward_shaping
        self.goal_seen = False #tracking whether goal seen yet
        self.wall_colors = wall_colors
        self.task_structure = task_structure
        self.goal_wiggle = goal_wiggle
        self.agent = [[0, 0], 0] #agent has a position and direction
        self.poster = poster
        #direction is 0: right, 1: up, 2: left, 3: down
        self.view_width = view_width
        self.max_steps = max_steps
        self.give_direction = give_direction
        self.give_dist = give_dist
        self.give_time = give_time

        total_width = view_width * 2 + 1

        observation_width = total_width
        if give_dist:
            observation_width = observation_width * 2
        if give_direction:
            observation_width += 1
        if give_time:
            observation_width += 1

        self.observation_space = spaces.Box(0, 6, shape=(observation_width,))

        self.action_space = spaces.Discrete(4)

        self.generate_world()
        self.randomize_agent_pos()

        
    def step(self, action):
        collision = False
        done = False
        reward = 0
        

        # -----Perform Action ------ #
        if action == 0:
            self.agent[1] = (self.agent[1] + 1) % 4
        elif action == 2:
            self.agent[1] = (self.agent[1] - 1) % 4
        elif action == 1:
            pos = self.agent[0].copy()
            if self.agent[1] == 0:
                pos[1] += 1
            elif self.agent[1] == 1:
                pos[0] -= 1
            elif self.agent[1] == 2:
                pos[1] -= 1
            elif self.agent[1] == 3:
                pos[0] += 1
                
            if pos[0] < 0 or pos[0] >= self.world_size[0] or \
                pos[1] < 0 or pos[1] >= self.world_size[1]:
                #cannot walk off edge of world
                pass
            elif self.obstacles[pos[0], pos[1]] == 0:
                self.agent[0] = pos
            else:
                collision = pos
        
            #check if reaching a goal
            if self.objects[pos[0], pos[1]] == 2:
                reward = self.goal_reward
                done = True


        #get observation                
        obs, colors = self.get_observation()

        #-------- Reward Shaping -------#
        #calc dist to goal
        y, x = self.agent[0]
        space_dists = np.abs(np.array([np.arange(self.world_size[0]) - y]).T) + \
            np.abs(np.array([np.arange(self.world_size[1]) - x]))
        dist_to_goal = np.min(space_dists[self.objects == 2])

        max_dist = 2 * self.world_size[0]

        goal_in_view = np.any(colors == 6)
        if goal_in_view:
            self.goal_seen = True

        #reward shaping 1: give reward based on distance away for goal
        if self.reward_shaping == 1:
            reward += (1 - (dist_to_goal / max_dist)) * self.sub_goal_reward

        #reward shaping 2: give reward based on whether goal is 
        #in sight and how far it is
        if self.reward_shaping == 2:
            if goal_in_view:
                reward += (1 - (dist_to_goal / max_dist)) * self.sub_goal_reward

        #reward shaping 3: give reward based on whether goal is seen
        #and once seen, give for total distance away
        if self.reward_shaping == 3:
            if self.goal_seen:
                reward += (1 - (dist_to_goal / max_dist)) * self.sub_goal_reward
        

        #--- Update Steps ---#
        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            done = True

        return obs, reward, done, {}
        
                
    def get_observation(self):
        '''
        Get observations based on vision lines. The agent sees to the left and right
        of where it is facing in a straight line. If the vision collides with an object (we assume
        it always does because there are walls, but without walls we would have to change it slightly)
        then we get a dist to the object and the color of the object
        '''
        #vision lines
        if self.agent[1] == 0:
            start = self.agent[0][1]
            end = self.world_size[1]
            left = self.agent[0][0] - self.view_width
            right = self.agent[0][0] + self.view_width
            left_idx = np.clip(left, 0, self.world_size[0])
            right_idx = np.clip(right, 0, self.world_size[0])
            left_right_idx = 0
            vis = self.visible[left_idx:right_idx+1, start:end] 
        elif self.agent[1] == 1:
            start = 0
            end = self.agent[0][0]
            left = self.agent[0][1] - self.view_width
            right = self.agent[0][1] + self.view_width
            left_idx = np.clip(left, 0, self.world_size[1])
            right_idx = np.clip(right, 0, self.world_size[1])
            left_right_idx = 1
            vis = np.rot90(self.visible[start:end+1, left_idx:right_idx+1], k=3)
        elif self.agent[1] == 2:
            start = 0
            end = self.agent[0][1]
            left = self.agent[0][0] + self.view_width
            right = self.agent[0][0] - self.view_width
            left_idx = np.clip(left, 0, self.world_size[0])
            right_idx = np.clip(right, 0, self.world_size[0])
            left_right_idx = 0
            vis = np.rot90(self.visible[right_idx:left_idx+1, start:end+1], k=2)
        elif self.agent[1] == 3:
            start = self.agent[0][0]
            end = self.world_size[0]
            left = self.agent[0][1] + self.view_width
            right = self.agent[0][1] - self.view_width
            left_idx = np.clip(left, 0, self.world_size[1])
            right_idx = np.clip(right, 0, self.world_size[1])
            left_right_idx = 1
            vis = np.rot90(self.visible[start:end, right_idx:left_idx+1], k=1)

        dists = np.argmax(vis > 0, axis=1)
        colors = vis[np.arange(vis.shape[0]), dists]

        if left < 0:
            dists = np.append([0]*-left, dists)
            colors = np.append([0]*-left, colors)
        if left >= self.world_size[left_right_idx]:
            dists = np.append([0]*(self.world_size[left_right_idx] + 1 - left), dists)
            colors = np.append([0]*(self.world_size[left_right_idx] + 1 - left), colors)
        if right < 0:
            dists = np.append(dists, [0]*-right)
            colors = np.append(colors, [0]*-right)
        if right >= self.world_size[left_right_idx]:
            dists = np.append(dists, [0]*(self.world_size[left_right_idx] + 1 - right))
            colors = np.append(colors, [0]*(self.world_size[left_right_idx] + 1 - right))
        
        obs = np.array(colors)
        if self.give_dist:
            obs = np.append(obs, dists)
        if self.give_direction:
            obs = np.append(obs, [self.agent[1]])
        if self.give_time:
            obs = np.append(obs, [self.current_steps])
        return obs, colors

    def find_empty_space(self, dist_from_others=0):
        '''
        Search for an empty space uniformly at random to populate with
        '''
        while True:
            y = np.random.randint(0, self.world_size[0])
            x = np.random.randint(0, self.world_size[1])
            if self.obstacles[y, x] == 0 and self.objects[y, x] == 0:
                if dist_from_others > 0:
                    y_range = np.clip([y-dist_from_others, y+dist_from_others+1], 
                        [0, 0], [self.world_size[0], self.world_size[0]])
                    x_range = np.clip([x-dist_from_others, x+dist_from_others+1], 
                        [0, 0], [self.world_size[1], self.world_size[1]])

                    if np.all(self.obstacles[y_range[0]:y_range[1], x_range[0]:x_range[1]] == 0):
                        return y, x

                else:
                    return y, x

    
    def reset(self):
        self.current_steps = 0
        self.generate_world()
        self.goal_seen = False
        self.randomize_agent_pos()
        obs, colors = self.get_observation()
        return obs
    
    def generate_world(self):
        '''
        Reset the world
        '''
        self.objects = np.zeros(self.world_size)
        self.visible = np.zeros(self.world_size)
        self.obstacles = np.zeros(self.world_size)
        
        self.generate_walls()
        
        #generate random obstacles
        for i in range(self.num_obstacles):
            y, x = self.find_empty_space()
            self.objects[y, x] = 1
            self.obstacles[y, x] = 1
            self.visible[y, x] = np.random.randint(1, 6)
            

        if self.task_structure == 1:
            #generate a goal with random position
            y, x = self.find_empty_space(self.goal_size - 1)
            self.objects[y:y+self.goal_size, x:x+self.goal_size] = 2
            self.obstacles[y:y+self.goal_size, x:x+self.goal_size] = 0
            self.visible[y:y+self.goal_size, x:x+self.goal_size] = 6

        if self.task_structure == 2:
            y, x = np.floor(np.array(self.world_size) * .75).astype(int)
            if self.goal_wiggle > 0:
                wiggle = np.random.choice(np.arange(-self.goal_wiggle, self.goal_wiggle + 1),
                    size=2)
                y = y + wiggle[0]
                x = x + wiggle[1]

            self.objects[y:y+self.goal_size, x:x+self.goal_size] = 2
            self.obstacles[y:y+self.goal_size, x:x+self.goal_size] = 0
            self.visible[y:y+self.goal_size, x:x+self.goal_size] = 0


        


    
    
    def generate_walls(self):
        '''
        Set walls to red color
        '''
        walls = []
        if self.wall_colors == 1:
            walls = ['red', 'red', 'red', 'red']
        elif self.wall_colors == 2:
            walls = ['red', 'green', 'red', 'green']
        elif self.wall_colors == 2.5:
            walls = ['red', 'red', 'green', 'green']
        elif self.wall_colors == 3:
            walls = ['red', 'green', 'red', 'blue']
        elif self.wall_colors == 3.5:
            walls = ['red', 'red', 'green', 'blue']
        elif self.wall_colors == 4:
            walls = ['red', 'green', 'blue', 'purple']

        #generate walls
        self.objects[:, 0] = self.object_to_idx['wall']
        self.objects[0, :] = self.object_to_idx['wall']
        self.objects[self.world_size[0]-1, :] = self.object_to_idx['wall']
        self.objects[:, self.world_size[1]-1] = self.object_to_idx['wall']
        
        #color walls red
        self.visible[:, 0] = self.color_to_idx[walls[0]]
        self.visible[0, :] = self.color_to_idx[walls[1]]
        self.visible[:, self.world_size[1]-1] = self.color_to_idx[walls[2]]
        self.visible[self.world_size[0]-1, :] = self.color_to_idx[walls[3]]
                
        #set walls as obstacles
        self.obstacles[:, 0] = 1
        self.obstacles[0, :] = 1
        self.obstacles[self.world_size[0]-1, :] = 1
        self.obstacles[:, self.world_size[1]-1] = 1

        if self.poster is not False:
            mid_point = int(np.floor(self.world_size[0] / 2))
            if self.poster == 0:
                self.visible[mid_point-1:mid_point+2, -1] = self.color_to_idx['yellow']
            elif self.poster == 1:
                self.visible[0, mid_point-1:mid_point+2] = self.color_to_idx['yellow']
            elif self.poster == 2:
                self.visible[mid_point-1:mid_point+2, 0] = self.color_to_idx['yellow']
            elif self.poster == 3:
                self.visible[-1, mid_point-1:mid_point+2] = self.color_to_idx['yellow']
        
        
        
    def randomize_agent_pos(self, heading=True):
        '''
        Randomize position of agent to position that is not an obstacle
        '''
        y, x = self.find_empty_space()

        self.agent[0] = [y, x]
        self.agent[1] = np.random.randint(0, 4)
        
        
    def render(self, mode='rgb_array'):
        window_size = [(self.world_size[0]) * 16, (self.world_size[1]) * 16]
        
        img = np.zeros(window_size + [3])

        #draw grid
        img[np.arange(0, window_size[0], 16), :, :] = 1
        img[:, np.arange(0, window_size[1], 16), :] = 1
        
        def color_block(x, y, rgb, img):
            img[y*16+1:(y+1)*16, x*16+1:(x+1)*16] = rgb
            return img
        
        #draw solid objects
        for i in range(self.world_size[0]):
            for j in range(self.world_size[1]):
                if self.visible[i, j] != 0:
                    # print(i)
                    # img[i*16+1:(i+1)*16, j*16+1:(j+1)*16] = self.idx_to_rgb[self.visible[i, j]]
                    img = color_block(j, i, self.idx_to_rgb[self.visible[i, j]], img)
                elif self.objects[i, j] != 0:
                    #draw invisible objects for viewer
                    img = color_block(j, i, [0.6, 0.6, 0.6], img)

        #draw agent
        y = self.agent[0][0]
        x = self.agent[0][1]
        img[y*16+1:(y+1)*16, x*16+1:(x+1)*16, :] = np.rot90(self.char_icon, k=self.agent[1])
        
        if mode == 'rgb_array':
            # return img.astype('uint8') * 255
            return (img * 255).astype('uint8')
        elif mode == 'human':
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

    def seed(self, seed=0):
        np.random.seed(seed)
            

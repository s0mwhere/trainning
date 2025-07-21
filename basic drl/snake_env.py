import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
from collections import deque

SNAKE_LEN_GOAL = 30*2
GRID_SIZE = 15

def collision_with_apple(apple_position, score):
	apple_position = [random.randrange(1,GRID_SIZE)*10,random.randrange(1,GRID_SIZE)*10]
	score += 1
	return apple_position, score

def collision_with_boundaries(snake_head):
	if snake_head[0]>=GRID_SIZE*10 or snake_head[0]<0 or snake_head[1]>=GRID_SIZE*10 or snake_head[1]<0 :
		return 1
	else:
		return 0

def collision_with_self(snake_position):
	snake_head = snake_position[0]
	if snake_head in snake_position[1:]:
		return 1
	else:
		return 0

def raytrac(i, snake_head):
	ray_dir = -1
	ray_close = 0
	if i[0]-i[1] == snake_head[0] - snake_head[1]:  #\
		if i[0]<snake_head[0]: 
			ray_dir = 7
			if i[0] + 10 == snake_head[0]: ray_close = 1
		else: 
			ray_dir = 3
			if i[0] - 10 == snake_head[0]: ray_close = 1
	elif i[0]+i[1] == snake_head[0] + snake_head[1]:	#/
		if i[0]<snake_head[0]: 
			ray_dir = 5
			if i[0] + 10 == snake_head[0]: ray_close = 1
		else: 
			ray_dir =1
			if i[0] - 10 == snake_head[0]: ray_close = 1
	elif i[0] == snake_head[0]:	#|
		if i[1] < snake_head[1]: 
			ray_dir = 0
			if i[1] + 10 == snake_head[1]: ray_close = 1
		else: 
			ray_dir = 4
			if i[1] - 10 == snake_head[1]: ray_close = 1
	elif i[1] == snake_head[1]:	#-
		if i[0] < snake_head[0]: 
			ray_dir = 6
			if i[0] + 10 == snake_head[0]: ray_close = 1
		else: 
			ray_dir = 2
			if i[0] - 10 == snake_head[0]: ray_close = 1
	return ray_dir, ray_close

def walldiago_trac(i):
	if i[0] == 0 : return 1
	elif i[0]+10 == GRID_SIZE*10: return 2
	else: return 0

def wallvertical_trac(i):
	if i[1] == 0 : return 1
	elif i[1]+10 == GRID_SIZE*10: return 2
	else: return 0

class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(SnekEnv, self).__init__()

        self.render_mode = render_mode
        self.clock = None
        self.window = None
        self.font = None
        self.dumb = (GRID_SIZE//2)*2

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        '''self.observation_space = spaces.Box(low=0, high=1000,
                                            shape=(6+SNAKE_LEN_GOAL,), dtype=np.int64)'''
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(8,3), dtype=np.int64)

    def step(self, action):
        self.reward = 0
               
        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down

        if action == 0 and self.prev_button_direction != 1:
            self.button_direction = 0
        elif action == 1 and self.prev_button_direction != 0:
            self.button_direction = 1
        elif action == 2 and self.prev_button_direction != 2:
            self.button_direction = 3
        elif action == 3 and self.prev_button_direction != 3:
            self.button_direction = 2
        else:
            self.button_direction = self.button_direction
        self.prev_button_direction = self.button_direction

        # Change the head position based on the button direction
        if self.button_direction == 1:
            self.snake_head[0] += 10
        elif self.button_direction == 0:
            self.snake_head[0] -= 10
        elif self.button_direction == 2:
            self.snake_head[1] += 10
        elif self.button_direction == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            '''self.dist = abs(self.snake_head[0]-self.apple_position[0]) + abs(self.snake_head[1]-self.apple_position[1])'''
            self.reward = self.score*4

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
            '''if self.dist < abs(self.snake_head[0]-self.apple_position[0]) + abs(self.snake_head[1]-self.apple_position[1]):
                self.reward = -1
            else: self.reward = 1
            self.dist = abs(self.snake_head[0]-self.apple_position[0]) + abs(self.snake_head[1]-self.apple_position[1])'''
            self.noapp += 1
            
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1 or self.noapp > 1000:
            self.terminated = True
            self.truncated = True
            self.reward = -100
        
        self.arr.fill(0)
        if wallvertical_trac(self.snake_head) == 1:
            self.arr[7][1], self.arr[0][1], self.arr[1][1] = 1, 1, 1
        if wallvertical_trac(self.snake_head) == 2:
            self.arr[3][1], self.arr[4][1], self.arr[5][1] = 1, 1, 1
        if walldiago_trac(self.snake_head) == 1:
            self.arr[5][1], self.arr[6][1], self.arr[7][1] = 1, 1, 1
        if walldiago_trac(self.snake_head) == 2:
            self.arr[1][1], self.arr[2][1], self.arr[3][1] = 1, 1, 1

        m, n = raytrac(self.apple_position, self.snake_head)
        if m != -1:
            self.arr[m][2] = 1
            self.arr[m][1] = n
        
        for i in self.snake_position[1:]:
            m, n =raytrac(i, self.snake_head)
            if m != -1 and self.arr[m][2] == 1:
                if (m==2 or m ==3 or m == 4) and (self.apple_position[0] < i[0] or self.apple_position[1] < i[1]):
                    continue
                if (m==6 or m ==7 or m == 0) and (self.apple_position[0] > i[0] or self.apple_position[1] > i[1]):
                    continue
                if m==1 and self.apple_position[0] < i[0]:
                    continue
                if m==5 and self.apple_position[0] > i[0]:
                    continue
                self.arr[m][0] = 1
                self.arr[m][2] = 0
                if self.arr[m][1] == 0: self.arr[m][1] = n
            elif m!= -1 and self.arr[m][2] != 1:
                self.arr[m][0] = 1
                if self.arr[m][1] == 0: self.arr[m][1] = n

        if self.render_mode == "human":
            self._render_frame()

        '''for _ in range(SNAKE_LEN_GOAL):
            self.prev_act.append(-1)
        for i in self.snake_position:
            self.prev_act.append(i[0])
            self.prev_act.append(i[1])

        self.observation = (self.snake_head[0], self.snake_head[1], self.apple_position[0], self.apple_position[1],
							len(self.snake_position), self.dist) + tuple(self.prev_act)'''
        self.observation = self.arr

        self.info = {}
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.snake_position = [[self.dumb*5,self.dumb*5],[self.dumb*5-10,self.dumb*5],[self.dumb*5-20,self.dumb*5]]
        self.apple_position = [random.randrange(1,GRID_SIZE)*10,random.randrange(1,GRID_SIZE)*10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [self.dumb*5,self.dumb*5]
        self.arr = np.zeros((8,3),dtype=int)
        '''self.dist = abs(self.snake_head[0]-self.apple_position[0]) + abs(self.snake_head[1]-self.apple_position[1])'''
        self.noapp = 0

        '''self.prev_act = deque([-1]*SNAKE_LEN_GOAL,maxlen=SNAKE_LEN_GOAL)
        for i in self.snake_position:
            self.prev_act.append(i[0])
            self.prev_act.append(i[1])
		
        self.observation = (self.snake_head[0], self.snake_head[1], self.apple_position[0], self.apple_position[1],
							len(self.snake_position), self.dist) + tuple(self.prev_act)'''
        self.observation = self.arr

        if self.render_mode == "human":
            self._render_frame()

        self.info = {}
        return self.observation, self.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((GRID_SIZE*10,GRID_SIZE*10))
            pygame.display.set_caption('snake')
            self.font = pygame.font.Font(None, 50)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.window.fill('white')

        text1_surf = self.font.render(f'{self.score}',False,(64,64,64))
        text1_rect = text1_surf.get_rect(center = (10,15))
        self.window.blit(text1_surf,text1_rect)

        # Display Apple
        pygame.draw.rect(self.window,'blue',(self.apple_position[0], self.apple_position[1],10,10))
        # Display Snake
        for position in self.snake_position:
            pygame.draw.rect(self.window,'red',(position[0],position[1],10,10))
        
        '''arr1=self.arr.tolist()
        text1_surf = self.font.render(f'uu{arr1[0]},ur{arr1[1]},rr{arr1[2]}',False,(64,64,64))
        text2_surf = self.font.render(f'dr{arr1[3]},dd{arr1[4]},dl{arr1[5]}',False,(64,64,64))
        text3_surf = self.font.render(f'll{arr1[6]},ul{arr1[7]}',False,(64,64,64))

        self.window.blit(text1_surf,(0,0))
        self.window.blit(text2_surf,(0,50))
        self.window.blit(text3_surf,(0,100))'''
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
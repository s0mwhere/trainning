import numpy as np
import pygame
import random
import time

GRID_SIZE = 50

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
	
def raytrac(i):
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

pygame.init()
screen = pygame.display.set_mode((GRID_SIZE*10,GRID_SIZE*10))
pygame.display.set_caption('snake')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)

# Initial Snake and Apple position
dumb = (GRID_SIZE//2)*2
snake_position = [[dumb*5,dumb*5],[dumb*5-10,dumb*5],[dumb*5-20,dumb*5]]
apple_position = [random.randrange(1,GRID_SIZE)*10,random.randrange(1,GRID_SIZE)*10]
score = 0
prev_button_direction = 1
button_direction = 1
snake_head = [dumb*5,dumb*5]

arr = np.zeros((8,3),dtype=int)
# 7  0  1	0: snake
# 6		2	1: dist
# 5  4  3	2: apple
while True:
	screen.fill('white')

	# Takes step after fixed time
	k=-1
	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			k=pygame.key.name(event.key)
			
	# 0-Left, 1-Right, 3-Up, 2-Down, q-Break
	# a-Left, d-Right, w-Up, s-Down

	if k == 'a' and prev_button_direction != 1:
		button_direction = 0
	elif k == 'd' and prev_button_direction != 0:
		button_direction = 1
	elif k == 'w' and prev_button_direction != 2:
		button_direction = 3
	elif k == 's' and prev_button_direction != 3:
		button_direction = 2
	elif k == 'q':
		break
	else:
		button_direction = button_direction
	prev_button_direction = button_direction

	# Change the head position based on the button direction
	if button_direction == 1:
		snake_head[0] += 10
	elif button_direction == 0:
		snake_head[0] -= 10
	elif button_direction == 2:
		snake_head[1] += 10
	elif button_direction == 3:
		snake_head[1] -= 10

	# Increase Snake length on eating apple
	if snake_head == apple_position:
		apple_position, score = collision_with_apple(apple_position, score)
		snake_position.insert(0,list(snake_head))

	else:
		snake_position.insert(0,list(snake_head))
		snake_position.pop()

	# On collision kill the snake and print the score
	if collision_with_boundaries(snake_head) == 1 or collision_with_self(snake_position) == 1:
		break

	pygame.draw.rect(screen,'blue',(apple_position[0],apple_position[1],10,10))
	# Display Snake

	if wallvertical_trac(snake_head) == 1:
		arr[7][1], arr[0][1], arr[1][1] = 1, 1, 1
	if wallvertical_trac(snake_head) == 2:
		arr[3][1], arr[4][1], arr[5][1] = 1, 1, 1
	if walldiago_trac(snake_head) == 1:
		arr[5][1], arr[6][1], arr[7][1] = 1, 1, 1
	if walldiago_trac(snake_head) == 2:
		arr[1][1], arr[2][1], arr[3][1] = 1, 1, 1

	m, n = raytrac(apple_position)
	if m != -1:
		arr[m][2] = 1
		if arr[m][1] == 0: arr[m][1] = n

	pygame.draw.rect(screen,'red',(snake_head[0],snake_head[1],10,10))
	for i in snake_position[1:]:
		pygame.draw.rect(screen,'red',(i[0],i[1],10,10))
		m, n =raytrac(i)
		if m != -1 and arr[m][2] == 1:
			if (m==2 or m ==3 or m == 4) and (apple_position[0] < i[0] or apple_position[1] < i[1]):
				continue
			if (m==6 or m ==7 or m == 0) and (apple_position[0] > i[0] or apple_position[1] > i[1]):
				continue
			if m==1 and apple_position[0] < i[0]:
				continue
			if m==5 and apple_position[0] > i[0]:
				continue
			arr[m][0] = 1
			arr[m][2] = 0
			if arr[m][1] == 0: arr[m][1] = n
		elif m!= -1 and arr[m][2] != 1:
			arr[m][0] = 1
			if arr[m][1] == 0: arr[m][1] = n

	arr1=arr.tolist()
	text1_surf = font.render(f'uu{arr1[0]},ur{arr1[1]},rr{arr1[2]}',False,(64,64,64))
	text2_surf = font.render(f'dr{arr1[3]},dd{arr1[4]},dl{arr1[5]}',False,(64,64,64))
	text3_surf = font.render(f'll{arr1[6]},ul{arr1[7]}',False,(64,64,64))

	screen.blit(text1_surf,(0,0))
	screen.blit(text2_surf,(0,50))
	screen.blit(text3_surf,(0,100))
		
	arr.fill(0)
	
	pygame.display.update()
	clock.tick(5)
    
pygame.quit()
exit()
#cv2.destroyAllWindows()
import pygame
import numpy as np
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
# Colors for 6 agents
# 1 start,2 outer circle, 3 agent, 4 inner circle, 5 end
# 1
#KHAKI = ((240, 230, 140), (255, 246, 143), (238, 230, 133), (205, 198, 115), (139, 134, 78))
# 2
#PALEGREEN = ((152, 251, 152), (154, 255, 154), (144, 238, 144), (124, 205, 124), (84, 139, 84))
# 3
#LIGHTPINK = ((255, 182, 193), (255, 174, 185), (238, 162, 173), (205, 140, 149), (139, 95, 101))
# 4
#ORCHID = ((218, 112, 214), (255, 131, 250), (238, 122, 233), (205, 105, 201), (139, 71, 137))
# 5
#LIGHTSKYBLUE = ((135, 206, 250), (176, 226, 255), (164, 211, 238), (141, 182, 205), (96, 123, 139))
# 6
#CORAL = ((255, 127, 80), (255, 114, 86), (238, 106, 80), (205, 91, 69), (139, 62, 47))
# 7
#DARKGOLDENROD = ((184, 134, 11), (255, 185, 15), (238, 173, 14), (205, 149, 12), (139, 101, 8))
# Array of agent's colors



class Color:
    def __init__(self, index, start_color, stop_color, path_color, impact_color, search_color):
        self.index = index
        self.start_color = start_color
        self.stop_color = stop_color
        self.path_color = path_color
        self.impact_color = impact_color
        self.search_color = search_color

    def chose_color(self):
        return COLORS[self.index]

#KHAKI = Color(0, (240, 230, 140), (255, 246, 143), (238, 230, 133), (205, 198, 115), (139, 134, 78))
salmon = Color(0, (250,128,114), (255,140,105), (238,130,98), (205,112,84), (139,76,57))
PALEGREEN = Color(1, (152, 251, 152), (154, 255, 154), (144, 238, 144), (124, 205, 124), (84, 139, 84))
LIGHTPINK = Color(2, (255, 182, 193), (255, 174, 185), (238, 162, 173), (205, 140, 149), (139, 95, 101))
ORCHID = Color(3, (218, 112, 214), (255, 131, 250), (238, 122, 233), (205, 105, 201), (139, 71, 137))
LIGHTSKYBLUE = Color(4, (135, 206, 250), (176, 226, 255), (164, 211, 238), (141, 182, 205), (96, 123, 139))
CORAL = Color(5, (255, 127, 80), (255, 114, 86), (238, 106, 80), (205, 91, 69), (139, 62, 47))
DARKGOLDENROD = Color(6, (184, 134, 11), (255, 185, 15), (238, 173, 14), (205, 149, 12), (139, 101, 8))

COLORS = [salmon, PALEGREEN, LIGHTPINK, ORCHID, LIGHTSKYBLUE, CORAL, DARKGOLDENROD]

class Robot:
    def __init__(self, start, end, curr, speed, color_index, priority, impact_rad, search_rad, path):
        self.start = start
        self.end = end
        self.curr = curr
        self.color_index = color_index
        self.speed = speed
        self.priority = priority
        self.impact_rad = impact_rad
        self.search_rad = search_rad
        self.impact_rad_points = self.generate_circle(self.curr, self.impact_rad)
        self.search_rad_points = self.generate_circle(self.curr, self.search_rad)
        self.path = []

    def fix_circle_points(self, input_list):
        seen = set()
        output_list = []

        for item in input_list:
            if item not in seen:
                if(not(on_board(item[0]) and on_board(item[1]))):
                    continue
                output_list.append(item)
                seen.add(item)

        return output_list

    def generate_circle(self, spot, radius=3, num_points=100):
        points = []
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = spot.row + radius * np.cos(theta)
        y = spot.col + radius * np.sin(theta)
        for x, y in zip(np.round(x).astype(int), np.round(y).astype(int)):
            points.append((x, y))
        return (self.fix_circle_points(points))

    def __lt__(self, other):
        return False

class Spot:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col

	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = TURQUOISE

	def make_path(self):
		self.color = PURPLE

	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbors(self, grid):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append(grid[self.row + 1][self.col])

		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])

		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])

		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])

	def check_point(self, robot_start_array, robot_end_array):
		#row, col = self.get_pos()
		for spot in robot_start_array:
			if self.row == spot.row and self.col == spot.col:
				return False
		for spot in robot_end_array:
			if self.row == spot.row and self.col == spot.col:
				return False
		return True

	def __lt__(self, other):
		return False


def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()


def algorithm(draw, grid, start, end):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start))
	came_from = {}
	g_score = {spot: float("inf") for row in grid for spot in row}
	g_score[start] = 0
	f_score = {spot: float("inf") for row in grid for spot in row}
	f_score[start] = h(start.get_pos(), end.get_pos())

	open_set_hash = {start}

	while not open_set.empty():
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			reconstruct_path(came_from, end, draw)
			end.make_end()
			return True

		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()

		draw()

		if current != start:
			current.make_closed()

	return False


def make_grid(rows, width):
	grid = []
	gap = width // rows
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			spot = Spot(i, j, gap, rows)
			grid[i].append(spot)

	return grid


def draw_grid(win, rows, width):
	gap = width // rows
	for i in range(rows):
		pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
		for j in range(rows):
			pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
	win.fill(WHITE)

	for row in grid:
		for spot in row:
			spot.draw(win)

	draw_grid(win, rows, width)
	pygame.display.update()


def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y, x = pos

	row = y // gap
	col = x // gap

	return row, col


def main(win, width):
    #ROWS = 20
    #R = Robot((5, 5), (1, 1), (5, 5), 1, 1, 1, 3, 4)
    MAX_ROBOTS = 7
    robot_index = 0
    grid = make_grid(ROWS, width)
    Robot_List = []
    robot_start_array = []
    robot_end_array = []
    mid_click = False
    start_end = True
    run = True
    show_rad = False
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not mid_click and robot_index < MAX_ROBOTS:
                    if start_end:
                        Robot_List.append(Robot(spot, spot, spot, 1, COLORS[robot_index], 1, 3, 4, None))
                        Spot.make_start(Robot_List[robot_index].start, Robot_List[robot_index].color_index)
                        start_end = False

                    else:
                        Robot_List[robot_index].end = spot
                        Spot.make_end(Robot_List[robot_index].end, Robot_List[robot_index].color_index)
                        start_end = True
                        robot_index += 1

                else:
                    if spot.check_point(Robot_List):
                        spot.make_barrier()
            elif pygame.mouse.get_pressed()[1]:  # Mid
                mid_click = True
                print("number of robots: " + str(robot_index))
            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                for i in range(robot_index):
                    if event.key == pygame.K_SPACE: #and robot_start_array[i] and robot_end_array[i]:
                        for row in grid:
                            for spot in row:
                                spot.update_neighbors(grid)

                        algorithm(lambda: draw(win, grid, ROWS, width), grid, Robot_List[i])

                    if event.key == pygame.K_c:
                        mid_click = False
                        start_end = True
                        robot_index = 0
                        grid = make_grid(ROWS, width)

                    if event.key == pygame.K_r:
                        grid = make_grid(ROWS, width)
                        #row, col, width, total_rows, robot
                        if not show_rad:
                            for robot in Robot_List:
                                for spot in robot.impact_rad_points:
                                    spot = Spot(spot[0],spot[1], width, ROWS, robot)
                                    spot.draw_impact_circle(robot.color_index)


    pygame.quit()

main(WIN, WIDTH)
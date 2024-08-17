import pygame
import numpy as np
from queue import PriorityQueue
import time
from probability import probability

ROWS = 20
WIDTH = 800
show_open_close = True
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")
SIMULATE = False

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
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

TOP_PERCENTAGE = 40

class points_to_look:
    def __init__(self, color, points, prob):
        self.color = color
        self.points = points
        self.prob = None

class Color:
    # For every robot there are 5 different sub-colors
    def __init__(self, index, start_color, stop_color, path_color, impact_color, search_color):
        self.index = index
        self.start_color = start_color
        self.stop_color = stop_color
        self.path_color = path_color
        self.impact_color = impact_color
        self.search_color = search_color

    def color_list(self): #set the color in array for future use
        return [self.start_color, self.stop_color, self.path_color, self.impact_color, self.search_color]

#Hard coded the colors by index, and then RGB for every sub-color
SALMON = Color(0, (250,128,114), (255,140,105), (238,130,98), (205,112,84), (139,76,57))
PALEGREEN = Color(1, (152, 251, 152), (154, 255, 154), (144, 238, 144), (124, 205, 124), (84, 139, 84))
LIGHTPINK = Color(2, (255, 182, 193), (255, 174, 185), (238, 162, 173), (205, 140, 149), (139, 95, 101))
ORCHID = Color(3, (218, 112, 214), (255, 131, 250), (238, 122, 233), (205, 105, 201), (139, 71, 137))
LIGHTSKYBLUE = Color(4, (135, 206, 250), (176, 226, 255), (164, 211, 238), (141, 182, 205), (96, 123, 139))
CORAL = Color(5, (255, 127, 80), (255, 114, 86), (238, 106, 80), (205, 91, 69), (139, 62, 47))
DARKGOLDENROD = Color(6, (184, 134, 11), (255, 185, 15), (238, 173, 14), (205, 149, 12), (139, 101, 8))

COLORS = [SALMON, PALEGREEN, LIGHTPINK, ORCHID, LIGHTSKYBLUE, CORAL, DARKGOLDENROD]
STOP_COLORS = [SALMON.stop_color, PALEGREEN.stop_color, LIGHTPINK.stop_color, ORCHID.stop_color, LIGHTSKYBLUE.stop_color, CORAL.stop_color, DARKGOLDENROD.stop_color]
class Robot:
    def __init__(self, start, end, curr, color_index, priority, impact_rad, search_rad, path):
        self.start = start #Start point of the robot
        self.end = end #Finish point of the path
        self.curr = curr #Current location of the robot
        self.color_index = color_index #Which color index belong to the robot
        self.priority = priority  #What is the robot index
        self.impact_rad = impact_rad #set the impact radius - how many spots
        self.search_rad = search_rad #Set the search radius - how many spots
        self.impact_rad_points = self.generate_circle(self.curr, self.impact_rad) #The points inside the impact radius
        self.impact_points_to_check = [] #Suspitions points inside the impact radius
        self.impact_points = self.points_inside_circle(self.curr, self.impact_rad) #Known points of other robot in the impaact radius
        self.search_rad_points = self.generate_circle(self.curr, self.search_rad) #The points inside the search radius
        self.search_points = self.points_inside_circle(self.curr, self.search_rad) #Known points of other robot in the search radius
        self.path = [] #The current robot path from current location to end
        self.impact_collision = [] #This are the Suspicion Points of another robot
        self.impact_lookat = [] #This value indicates the "raw" knowledge of spots with another robot in them
        self.points_to_avoid = [] #This value is the key component in avoiding collisions, it hold the points that we suspect there will be another robot in the future


    def fix_circle_points(self, input_list):
        #Given array of spots, filter out all the spots that are not on board
        #Gets the array of points
        seen = set()
        output_list = []

        for item in input_list:
            if item not in seen:
                if not(on_board(item[0]) and on_board(item[1])):
                    continue
                output_list.append(item)
                seen.add(item)

        return output_list

    def generate_circle(self, spot, radius=3, num_points=100):
        # Given radius and point generate circle, num_points by default is set to 100 and help creating circle
        points = []
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = spot.row + radius * np.cos(theta)
        y = spot.col + radius * np.sin(theta)
        for x, y in zip(np.round(x).astype(int), np.round(y).astype(int)):
            points.append((x, y))
        return self.fix_circle_points(points)

    def points_inside_circle(self, spot, radius=3):
        # Given point and radius get all the spots inside the circle
        points_inside = []
        min_row = int(spot.row - radius)
        max_row = int(spot.row + radius)
        min_col = int(spot.col - radius)
        max_col = int(spot.col + radius)
        if min_row < 0:
            min_row = 0
        if max_row > ROWS-1:
            max_row = ROWS-1
        if min_col < 0:
            min_col = 0
        if max_col > ROWS - 1:
            max_col = ROWS - 1
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if (row - spot.row) ** 2 + (col - spot.col) ** 2 <= (radius+1) ** 2:
                    points_inside.append((row, col))
        return points_inside

    def check_circle(self, search_list,Robot_List,priority):
        # Given search_list return all the suspition points
        return_list = {}
        for i, pos in enumerate(search_list):
            spot = get_spot(pos)
            if spot.color != WHITE and spot.color != BLACK and spot.color != RED and spot.color not in self.color_index.color_list() and spot.color not in STOP_COLORS:
                #return_list.append(spot)
                return_list[pos] = spot
            for robot in Robot_List:
                if priority == robot.priority:
                    continue
                if pos == robot.curr.get_pos():
                    spot.color =  robot.color_index.start_color
                    return_list[pos] = spot
        return list(return_list.values())

    def lookat(self, lookat_list, search_list):
        # Given the search_list and current robot that nearby, return the new robots nearby and their current location
        res = []
        for spot in search_list:
            if len(lookat_list) > 0:
                for i, spot_hist in enumerate(lookat_list):
                    if spot.color == spot_hist.color:
                        spot_hist.points.append(spot)
                        lookat_list[i] = spot_hist
                        res = lookat_list
                    else:
                        res.append(points_to_look(spot.color, [spot], None))
            else:
                res.append(points_to_look(spot.color, [spot], None))
        return res

    def __lt__(self, other):
        return False

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row # Spot row and colom number
        self.col = col
        self.x = row * width # The values in the grid that belong to the current spot
        self.y = col * width
        self.color = WHITE # Spot initial color
        self.neighbors = [] # All the spot's neighbors
        self.close = []  # Whether there are robot there or no
        self.open = []
        self.width = width # Spot size
        self.total_rows = total_rows

    def get_pos(self): # Return the spot row and colom
        return self.row, self.col

    def make_red(self): # Draw the spot in red - closed
        self.color = RED

    def draw_impact_circle(self, robot_color): # Draw impact circle by the robot's color index
        self.color = robot_color.impact_color

    def draw_search_circle(self, robot_color): # Draw search circle by the robot's color index
        self.color = robot_color.search_color

    def is_closed(self): # Check if the spot is closed - red
        return self.color == RED

    def is_open(self): # Check if the spot is open - green
        return self.color == GREEN

    def is_barrier(self): # Check if the spot is barried - black
        return self.color == BLACK

    def reset(self): # Reset the spot color to white
        self.color = WHITE

    def make_start(self, robot_color): # Draw the spot by color index - start color
        self.color = robot_color.start_color

    def make_closed(self): # Draw the spot by closed color - red
        self.close.append(self.get_pos())
        if show_open_close:
            self.color = RED

    def make_open(self): # Draw the spot by open color - green
        self.open.append(self.get_pos())
        if show_open_close:
            self.color = GREEN

    def make_barrier(self): # Set the spot to be barrier - black
        self.color = BLACK

    def make_end(self, robot_color): # Set the spot by color index to be end color
        self.color = robot_color.stop_color

    def make_path(self, robot_color): # Draw the spot to be path
        self.color = robot_color.path_color

    def draw(self, win): # Given the boarad and spot, this function use pygame library and draw the spot with new color
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self): # This function update for specific spot all the spots that are neighbors
        self.neighbors = []
        if self.row < self.total_rows - 1 and not GRID[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(GRID[self.row + 1][self.col])

        if self.row > 0 and not GRID[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(GRID[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not GRID[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(GRID[self.row][self.col + 1])

        if self.col > 0 and not GRID[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(GRID[self.row][self.col - 1])

    def check_point(self, Robot_List): # Given specific points and all the robots check if the spot is empty or not - False for not empty
        for robot in Robot_List:
            if self.row == robot.start.row and self.col == robot.start.col:
                return False
        for robot in Robot_List:
            if self.row == robot.end.row and self.col == robot.end.col:
                return False
        return True

    def __lt__(self, other):
        return False

def get_spot(pos): # Given spot row and colom - pos - return the spot object
    for row in GRID:
        for spot in row:
            if spot.row == pos[0] and spot.col == pos[1]:
                return spot
    return None

def h(p1, p2):  # Givn two points calculate the huristic value to evalute the spot for A*
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def on_board(x): # Check if the value (Can be row or colom) is on board or not
    return x>=0 and x< ROWS

def reconstruct_path(came_from, current, draw, Robot,update_grid):
    # Came from - new path from A*
    # current - robot current location
    # robot - the current robot
    # update grid - if its the first run (After space key) whether or not to draw now the whole path
    # This function update the robot new path and can draw the whole path
    path = []
    new_path = []
    if update_grid:
        new_path.append(Robot.curr)
    while current in came_from:
        current = came_from[current]
        path.append(current)
    for current in reversed(path[:-1]):
        new_path.append(current)
        if update_grid:
            current.make_path(Robot.color_index)
            draw()
    new_path.append(Robot.end)
    Robot.path = new_path

def calc_manhattan_dist(p1, p2): #Given two points calculate the manhatan distance between them
    return abs(p1.row - p2.row) + abs(p1.col - p2.col)

def algorithm(draw, Robot, update_grid):
    # robot the current robot
    # update_grid - if its the first run (after space key) whether or not to draw the whole path
    # Calculate the best available path to the end point
    #
    # initialize start point and end point and initilize the whole grid huristic to be inf expect the start location
    start = Robot.curr
    end = Robot.end
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in GRID for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in GRID for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    open_set_hash = {start}
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_set.get()[2]
        avoid = Robot.points_to_avoid
        current_po = current.get_pos()
        open_set_hash.remove(current)
        for row in GRID:
            for spot in row:
                if(g_score[spot] == float("inf") and f_score[spot] == float("inf")):
                    continue
        if (Robot.points_to_avoid):# Given point to avoid in the futre, build the path so that robot will avoid it
            man_dist = calc_manhattan_dist(current, Robot.curr)
            if (man_dist < len(Robot.points_to_avoid)):
                if  current.get_pos() in Robot.points_to_avoid[man_dist] or (current.get_pos() in Robot.points_to_avoid[man_dist-1] and Robot.priority != 0):
                    continue
        if current == end:
            reconstruct_path(came_from, end, draw,Robot,update_grid)
            if update_grid:
                end.make_end(Robot.color_index)
            return True
        for neighbor in current.neighbors:
            if (Robot.points_to_avoid):  # Given point to avoid in the futre, build the path so that robot will avoid it
                man_dist = calc_manhattan_dist(neighbor, Robot.start)
                if (man_dist < len(Robot.points_to_avoid)):
                    if neighbor.get_pos() in Robot.points_to_avoid[man_dist - 1]:
                        temp_g_score = float("inf")
                        continue
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    if update_grid:
                        neighbor.make_open()
        draw()
        if current != start and update_grid:
            current.make_closed()
    return False

def avoidInTheWay(robot):
    # given the robot this function check if the PTA and the path cross by place and time
    # This is in order to not change the current path if there is no reason to
    for turn in robot.points_to_avoid:
        for avoid in turn:
            avoid_cor =[avoid[0],avoid[1]]
            dist = abs(robot.curr.row - avoid_cor[0]) + abs(robot.curr.col - avoid_cor[1])
            if(dist >= len(robot.path)):
                continue
            if robot.path[dist].row == avoid_cor[0] and robot.path[dist].col == avoid_cor[1]:
                return True
            dist -=1 #Sometimes the other robot already did its move
            if robot.path[dist].row == avoid_cor[0] and robot.path[dist].col == avoid_cor[1]:
                return True
    return False

def calc_grad(points):
    # points array of points
    # this function calculate the gradient of the robot by the last 2 points
    last_point = points[-1]
    sec_to_last_point = points[-2]
    calc_row = last_point.row - sec_to_last_point.row
    calc_col = last_point.col - sec_to_last_point.col
    if calc_row == 1:
        return 'right'
    elif calc_row == -1:
        return 'left'
    elif calc_col == 1:
        return 'down'
    elif calc_col == -1:
        return 'up'
    elif calc_col == 0 and calc_row == 0:
        return 'STAY'
    else:
        return "UNKNOWN"
GRID = []

def make_grid(rows, width):
    # Rows - number of rows
    # width - Size of every spot in the grid
    # This function creats a 2-D grid - our board
    gap = width // rows
    for i in range(rows):
        GRID.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            GRID[i].append(spot)

def draw_grid(win, rows, width):
    # win - Window to draw on
    # rows - number of rows and coloms
    # width - width and height of every spot
    # This function draw the grid
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, rows, width):
    # win - Window to draw on
    # rows - number of rows and coloms
    # width - width and height of every spot
    # This function update the whole spots - first to white and than to the new color
    win.fill(WHITE)
    for row in GRID:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

def draw_blank(win, rows, width):
    # win - Window to draw on
    # rows - number of rows and coloms
    # width - width and height of every spot
    # This function draw all the spots to white if they are not black - barrier
    win.fill(WHITE)
    for row in GRID:
        for spot in row:
            if spot.color != BLACK:
                spot.color = WHITE
                spot.draw(win)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    # pos - where the user cliced
    # rows - number of rows and coloms
    # width - width and height of every spot
    # This function return the correct spot by row and colom given where on the window the user has clicked
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

def points_in_circle(center, radius):
    # center - center of the circle
    # radius - the radius of the circle
    # This function return all the spots in the circle
    cx, cy = center
    x = np.arange(cx - radius, cx + radius + 1)
    y = np.arange(cy - radius, cy + radius + 1)
    xx, yy = np.meshgrid(x, y)
    distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Create a mask to filter points within the circle
    mask = distances <= radius
    points = np.column_stack((xx[mask], yy[mask]))
    return points

def get_non_diagonal_neighbors(x, y):
    # x,y - rows and colom of spot
    # This function return all valid neighbors of spot by (row,colom)
    neighbors = [
        (x, y-1),  # Top
        (x-1, y),  # Left
        (x+1, y),  # Right
        (x, y+1)   # Bottom
    ]
    valid_neighbors = [(i, j) for i, j in neighbors if 0 <= i < 20 and 0 <= j < 20]
    return valid_neighbors

def update_rad(robot,Robot_List):
    # This function get a robot, and generate its radiud
    # Later the function search for other robots in the circle, and than check if already saw the robot or not to generate gradient
    # Finaly the function add PTA to the robot
    robot.impact_rad_points = robot.generate_circle(robot.curr, robot.impact_rad)
    robot.impact_points = robot.points_inside_circle(robot.curr, robot.impact_rad)
    robot.impact_collision = robot.check_circle(robot.impact_points,Robot_List,robot.priority)
    robot.impact_lookat = robot.lookat(robot.impact_lookat, robot.impact_collision)
    for i, spot in enumerate(robot.impact_lookat):
        death_spots = get_non_diagonal_neighbors(spot.points[-1].get_pos()[0],spot.points[-1].get_pos()[1])
        if len(spot.points) > 1:
            if calc_grad(spot.points) == "STAY":
                pos = spot.points[0].get_pos()
                points_to_avoid = [[pos], get_non_diagonal_neighbors(pos[0], pos[1])]
                robot.points_to_avoid = merge_points_to_avoid(points_to_avoid, robot.points_to_avoid)
                continue
                # In case there is gradient than get the top PTA by using probability
            robot.impact_lookat[i].prob = probability([0.75, 0.05, 0.1], calc_grad(spot.points),spot.points[-1].get_pos(), 3, TOP_PERCENTAGE)
            if (robot.impact_lookat[i].prob.blocked_in_future):
                robot.points_to_avoid = merge_points_to_avoid(robot.impact_lookat[i].prob.blocked_in_future,robot.points_to_avoid)
        if len(spot.points) == 1:
            pos = spot.points[0].get_pos()
            points_to_avoid = [[pos],get_non_diagonal_neighbors(pos[0], pos[1])]
            robot.points_to_avoid = merge_points_to_avoid(points_to_avoid,robot.points_to_avoid)
        robot.points_to_avoid[1] = list(set(robot.points_to_avoid[1]) | set(death_spots))

def merge_points_to_avoid(arr1,arr2):
    # Given two arrays of PTA with sub arrays, this fucntion merge the array in way to keep the order of the sub arrays-PTA
    if(not arr2):
        return arr1
    if(not arr1):
        return arr2
    max_length = max(len(arr1), len(arr2))
    merged_array = []
    for i in range(max_length):
        subarr1 = arr1[i] if i < len(arr1) else []
        subarr2 = arr2[i] if i < len(arr2) else []
        merged_subarr = list(subarr1) + list(subarr2)
        merged_array.append(merged_subarr)
    return merged_array

def main(win, width):
    # this function get the window and the width of the spots. And basicily run the whole process
    # THis function firstly initilize the grid, the robots the barrier and run all the process
    MAX_ROBOTS = 7
    robot_index = 0
    make_grid(ROWS, width)
    Robot_List = []
    mid_click = False
    start_end = True
    run = True
    show_rad = False
    SIMULATE = False
    timer_count = 0
    while run:
        draw(win, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if pygame.mouse.get_pressed()[0]:  # LEFT ->  initilize the robots initial place and end place
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = GRID[row][col]
                if spot.color != WHITE:
                    continue
                if not mid_click and robot_index < MAX_ROBOTS:
                    if start_end:
                        Robot_List.append(Robot(spot, False, spot, COLORS[robot_index], robot_index, 4, 6, None))
                        Spot.make_start(Robot_List[robot_index].start, Robot_List[robot_index].color_index)
                        start_end = False
                    else:
                        Robot_List[robot_index].end = spot
                        Spot.make_end(Robot_List[robot_index].end, Robot_List[robot_index].color_index)
                        start_end = True
                        robot_index += 1
                    GRID[row][col] = spot
                else:
                    if spot.check_point(Robot_List):
                        spot.make_barrier()
            elif pygame.mouse.get_pressed()[1]:  # Mid -> This section initilize the barrier spots
                mid_click = True
                print("number of robots: " + str(robot_index))
            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = GRID[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
            if event.type == pygame.KEYDOWN or SIMULATE: #This section start calaculate the first path by regular  A*
                if SIMULATE:
                    event.key = pygame.K_g
                for i in range(robot_index):
                    if event.key == pygame.K_SPACE:
                        print("Start calculating path")
                        start_time = time.time()
                        for row in GRID:
                            for spot in row:
                                spot.update_neighbors()
                        algorithm(lambda: draw(win, ROWS, width), Robot_List[i],True)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Elapsed time: {elapsed_time} seconds, for robot number: {i+1}/{robot_index}")
                    if event.key == pygame.K_c:
                        mid_click = False
                        start_end = True
                        robot_index = 0
                        make_grid(ROWS, width)
                    if event.key == pygame.K_r:  #This shows the radiuses
                        if not show_rad:
                            for robot in Robot_List:
                                for pos in robot.impact_rad_points:
                                    spot = GRID[pos[0]][pos[1]]
                                    spot.draw_impact_circle(robot.color_index)
                                for pos in robot.search_rad_points:
                                    spot = GRID[pos[0]][pos[1]]
                                    spot.draw_search_circle(robot.color_index)
                                draw(win, ROWS, width)
                    if event.key == pygame.K_s:
                        draw_blank(win, width, ROWS)
                        for robot in Robot_List:
                            print(robot.start.get_pos(), robot.end.get_pos())
                            robot.start.make_start(robot.color_index)
                            robot.end.make_start(robot.color_index)
                            for spot in robot.path:
                                spot.make_start(robot.color_index)
                                spot.make_path(robot.color_index)
                            draw(win, ROWS, width)
                if event.key == pygame.K_g: # For every press on 'g' this section moves the robot in one spot
                    draw_blank(win, width, ROWS)
                    all_finished = True
                    for robot in Robot_List: #This section check if all robots has reached to their end spot and if sor end the run
                        if robot.curr != robot.end:
                            all_finished = False
                        if timer_count == 0 :
                            robot.start.make_start(robot.color_index)
                    if all_finished:
                        print("All robots has got to their goal, the run is finished.")
                        exit()
                    timer_count += 1
                    print(f"time: {timer_count}")
                    for i, robot in enumerate(Robot_List):
                        update_rad(robot,Robot_List)
                        # if there are points to avoid look if there are in our robot path, if not exit
                        # if there are in robot path calculate new path
                        if robot.points_to_avoid:
                            val = avoidInTheWay(robot)
                            PTA = robot.points_to_avoid
                            if avoidInTheWay(robot):
                                ret_value = algorithm(lambda: draw(win, ROWS, width), Robot_List[i], False)
                                if Robot_List[i].path and Robot_List[i].path[0] != Robot_List[i].curr and timer_count == 1 or ret_value == False:
                                    Robot_List[i].path.insert(0, Robot_List[i].curr)
                            robot.points_to_avoid = ""  #Delete old PTA
                        Spot.reset(robot.curr)
                        robot.curr = robot.path[0]
                        Spot.make_start(robot.curr, robot.color_index)
                        Spot.make_end(robot.path[-1], robot.color_index)
                        if robot.curr == robot.end:
                            print(f"Robot {i+1} Ended course")
                            SIMULATE = False
                        if len(robot.path[1:]) > 0:
                            robot.path = robot.path[1:]
    pygame.quit()

main(WIN, WIDTH)
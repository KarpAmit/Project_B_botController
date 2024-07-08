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

TOP_PERCENTAGE = 30

class points_to_look:
    def __init__(self, color, points, prob):
        self.color = color
        self.points = points
        self.prob = None

class Color:
    def __init__(self, index, start_color, stop_color, path_color, impact_color, search_color):
        self.index = index
        self.start_color = start_color
        self.stop_color = stop_color
        self.path_color = path_color
        self.impact_color = impact_color
        self.search_color = search_color

    def color_list(self):
        return [self.start_color, self.stop_color, self.path_color, self.impact_color, self.search_color]

#KHAKI = Color(0, (240, 230, 140), (255, 246, 143), (238, 230, 133), (205, 198, 115), (139, 134, 78))
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
        self.impact_points_to_check = []
        self.impact_points = self.points_inside_circle(self.curr, self.impact_rad)
        self.search_rad_points = self.generate_circle(self.curr, self.search_rad)
        self.search_points = self.points_inside_circle(self.curr, self.search_rad)
        self.search_points_to_check = []
        self.path = []
        self.impact_collision = []
        self.search_collision = []
        self.impact_lookat = []
        self.search_lookat = []
        self.points_to_avoid = []


    def fix_circle_points(self, input_list):
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
        points = []
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = spot.row + radius * np.cos(theta)
        y = spot.col + radius * np.sin(theta)
        for x, y in zip(np.round(x).astype(int), np.round(y).astype(int)):
            points.append((x, y))
        return self.fix_circle_points(points)

    def points_inside_circle(self, spot, radius=3):
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

    def check_circle(self, search_list):
        return_list = []
        #print(type(search_list))
        for i, pos in enumerate(search_list):
            #print(i, pos, type(pos))
            spot = get_spot(pos)
            #print(spot.color, self.priority)
            if spot.color != WHITE and spot.color != BLACK and spot.color != RED and spot.color not in self.color_index.color_list() and spot.color not in STOP_COLORS:
                #print(self.priority, pos, spot.color)
                return_list.append(spot)
        return return_list

    def lookat(self, lookat_list, search_list):
        res = []
        print(f"lookat_list: {lookat_list}")
        print(f"search_list: {search_list}")
        for spot in search_list:
            if len(lookat_list) > 0:
                for i, spot_hist in enumerate(lookat_list):
                    if spot.color == spot_hist.color:
                        spot_hist.points.append(spot)
                        #print(f"robot_number: {self.priority} saw a robot go: {calc_grad(spot_hist.points)}")
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
        self.row = row
        self.col = col
        #self.robot = robot
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.close = []
        self.open = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def make_red(self):
        #print("IN RED")
        self.color = RED

    def draw_impact_circle(self, robot_color):

        self.color = robot_color.impact_color

    def draw_search_circle(self, robot_color):
        self.color = robot_color.search_color

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

    def make_start(self, robot_color):
        print(f"make start {self.row, self.col}")
        self.color = robot_color.start_color
        #30.0

    def make_closed(self):
        self.close.append(self.get_pos())
        if show_open_close:
            self.color = RED

    def make_open(self):
        self.open.append(self.get_pos())
        if show_open_close:
            self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self, robot_color):
        self.color = robot_color.stop_color

    def make_path(self, robot_color):
        #print("my color is:= " + str(self.color) + " in path " + str(robot_color.path_color))
        self.color = robot_color.path_color

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def spot_is_not_free(spot, robot, Robot_List):
        for other_robot in Robot_List:
            if other_robot != robot:
                if spot.row == other_robot.curr.row and spot.col == other_robot.curr.col:
                    return True
        return spot.is_barrier()

    def update_neighbors(self):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not GRID[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(GRID[self.row + 1][self.col])

        if self.row > 0 and not GRID[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(GRID[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not GRID[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(GRID[self.row][self.col + 1])

        if self.col > 0 and not GRID[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(GRID[self.row][self.col - 1])

    def check_point(self, Robot_List):
        # row, col = self.get_pos()

        for robot in Robot_List:
            if self.row == robot.start.row and self.col == robot.start.col:
                return False
        for robot in Robot_List:
            if self.row == robot.end.row and self.col == robot.end.col:
                return False
        return True
    """
    def __eq__(self, other):
        if isinstance(other, Spot):
            return self.color == other.color
        return False
    """
    def __lt__(self, other):
        return False

def get_spot(pos):
    for row in GRID:
        for spot in row:
            #print(type(spot.row))
            if spot.row == pos[0] and spot.col == pos[1]:
                return spot
    return None


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def on_board(x):
    return x>=0 and x< ROWS

def reconstruct_path(came_from, current, draw, Robot,update_grid):
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
    #draw()
    Robot.path = new_path
    #print(f"\n\nRobot {Robot.priority} start {Robot.curr.row},{Robot.curr.col}")
    #for iter in Robot.path:
    #    print(f"path {iter.row},{iter.col}")


def calc_manhattan_dist(p1, p2):
    return abs(p1.row - p2.row) + abs(p1.col - p2.col)

def algorithm(draw, Robot, update_grid):
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
        #
        current_loc = current.get_pos()
        #check 1,2
        open_set_hash.remove(current)
        # Debugging: Print g_score and f_score for every tile
        #print("before we got :\n")
        for row in GRID:
            for spot in row:
                if(g_score[spot] == float("inf") and f_score[spot] == float("inf")):
                    continue
        if (Robot.points_to_avoid):# Given point to avoid in the futre, build the path so that robot will avoid it
            man_dist = calc_manhattan_dist(current, Robot.curr)
            if (man_dist < len(Robot.points_to_avoid)):
                if current.get_pos() in Robot.points_to_avoid[man_dist]:
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

        draw()#Need to fix here the draw

        if current != start and update_grid:
            current.make_closed()

    return False
#3,5 [4,5 3,4 3,6] 5,5
def avoidInTheWay(robot):
    tmp = robot.curr.get_pos()
    for turn in robot.points_to_avoid:
        for avoid in turn:
            avoid_cor =[avoid[0],avoid[1]] #[list(t) for t in avoid]
            print(f"avoid is {avoid_cor} and {type(avoid_cor)}")
            #avoid_cor = [int(x.strip()) for x in tuple_string.split(",")]
            dist = abs(robot.curr.row - avoid_cor[0]) + abs(robot.curr.col - avoid_cor[1])
            avoing = robot.path[dist].get_pos()
            print(f"dist is {dist} point is {robot.path[dist].get_pos()}")
            if robot.path[dist].row == avoid_cor[0] and robot.path[dist].col == avoid_cor[1]:
                print(f"got a match for {avoid_cor}")
                return True
            else:
                print(f"point {avoid_cor} not in way")
    return False

def calc_grad(points):
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
    gap = width // rows
    for i in range(rows):
        GRID.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            GRID[i].append(spot)


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, rows, width):
    win.fill(WHITE)

    for row in GRID:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

def draw_blank(win, rows, width):
    win.fill(WHITE)

    for row in GRID:
        for spot in row:
            if spot.color != BLACK:
                spot.color = WHITE
                spot.draw(win)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def points_in_circle(center, radius):
    cx, cy = center
    x = np.arange(cx - radius, cx + radius + 1)
    y = np.arange(cy - radius, cy + radius + 1)
    xx, yy = np.meshgrid(x, y)
    distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Create a mask to filter points within the circle
    mask = distances <= radius
    points = np.column_stack((xx[mask], yy[mask]))

    return points

def update_collision(robot_list):
    search_color = [WHITE, BLACK, RED, GREEN]
    search_collision = []
    #impact_collision = []
    for robot in robot_list:
        search_color += robot.color_index.color_list()
        search_points = points_in_circle(robot.curr.get_pos(), robot.impact_rad)
        for spot in search_points:
            spot = GRID[spot[0]][spot[1]]
            if spot.color not in search_color:
                spot.make_red()
                robot.impact_collision.append(spot)
        search_color = [x for x in search_color if x not in robot.color_index.color_list()]

def get_non_diagonal_neighbors(x, y):
    neighbors = [
        (x, y-1),  # Top
        (x-1, y),  # Left
        (x+1, y),  # Right
        (x, y+1)   # Bottom
    ]
    valid_neighbors = [(i, j) for i, j in neighbors if 0 <= i < 20 and 0 <= j < 20]
    return valid_neighbors

def update_rad(robot):
    print(f"robot number {robot.priority + 1} has entered update_rad")
    robot.impact_rad_points = robot.generate_circle(robot.curr, robot.impact_rad)
    robot.impact_points = robot.points_inside_circle(robot.curr, robot.impact_rad)
    robot.impact_collision = robot.check_circle(robot.impact_points)
    #for point in robot.impact_collision:
    #    print(f"impact_collision: {point.get_pos()}")
    robot.impact_lookat = robot.lookat(robot.impact_lookat, robot.impact_collision)
    #print(f"impact list: {robot.impact_lookat}")
    for points in robot.impact_lookat:
        print(points.points[0].get_pos())
    for i, spot in enumerate(robot.impact_lookat):
        if len(spot.points) > 1:
            if calc_grad(spot.points) == "STAY":
                continue
            robot.impact_lookat[i].prob = probability([0.75, 0.05, 0.1], calc_grad(spot.points),spot.points[-1].get_pos(), 3, TOP_PERCENTAGE)
            if (robot.impact_lookat[i].prob.blocked_in_future):
                robot.points_to_avoid = robot.impact_lookat[i].prob.blocked_in_future
                print(f"robot {robot.priority} added {robot.points_to_avoid} ")
        if len(spot.points) == 1:
            pos = spot.points[0].get_pos()
            robot.points_to_avoid.append([pos])
            robot.points_to_avoid.append(get_non_diagonal_neighbors(pos[0], pos[1]))


    #robot.search_rad_points = robot.generate_circle(robot.curr, robot.search_rad)
    #robot.search_points = robot.points_inside_circle(robot.curr, robot.search_rad)
    #robot.search_collision = robot.check_circle(robot.search_points)
    #robot.search_lookat = robot.lookat(robot.search_lookat, robot.search_collision)
    #print(f"search_collision: {robot.search_collision}")

def main(win, width):
    #ROWS = 20
    #R = Robot((5, 5), (1, 1), (5, 5), 1, 1, 1, 3, 4)
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

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = GRID[row][col]
                if spot.color != WHITE:
                    continue
                if not mid_click and robot_index < MAX_ROBOTS:
                    if start_end:
                        Robot_List.append(Robot(spot, False, spot, 1, COLORS[robot_index], robot_index, 4, 6, None))
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
            elif pygame.mouse.get_pressed()[1]:  # Mid
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

            if event.type == pygame.KEYDOWN or SIMULATE:
                if SIMULATE:
                    event.key = pygame.K_g
                for i in range(robot_index):
                    if event.key == pygame.K_SPACE: #and robot_start_array[i] and robot_end_array[i]:
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

                    if event.key == pygame.K_r:
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
                        print("key S")
                        draw_blank(win, width, ROWS)
                        for robot in Robot_List:
                            print(robot.start.get_pos(), robot.end.get_pos())
                            robot.start.make_start(robot.color_index)
                            robot.end.make_start(robot.color_index)
                            for spot in robot.path:
                                spot.make_start(robot.color_index)
                                spot.make_path(robot.color_index)
                            draw(win, ROWS, width)

                if event.key == pygame.K_g:# or SIMULATE:
                    #SIMULATE = True
                    draw_blank(win, width, ROWS)
                    all_finished = True
                    for robot in Robot_List:
                        #robot.update_rad()
                        #update_rad(robot)
                        #time.sleep(2)
                        if robot.curr != robot.end:#Changed here so that when all robots will get to their destanation the run will end
                            all_finished = False
                        make_start_at = robot.path[0].get_pos()
                        print(f"robot {robot.priority} is printing {make_start_at} at start")
                        Spot.make_end(robot.path[-1], robot.color_index)
                        #Spot.make_start(robot.path[0], robot.color_index)
                    if all_finished:
                        print("All robots has got to their goal. Do you want to restart?")
                        exit()

                    timer_count += 1
                    print(f"time: {timer_count}")
                    for i, robot in enumerate(Robot_List):
                        update_rad(robot)
                        # if there are points to avoid look if there are in our robot path, if not exit
                        # if there are in robot path calculate new path
                        if robot.points_to_avoid:
                            print(f"\nPOINTS TO AVOID are {robot.points_to_avoid}")
                            if True :#avoidInTheWay(robot):
                                print("before")
                                for tile in robot.path:
                                    print(f"{tile.row},{tile.col}")
                                ret_value = algorithm(lambda: draw(win, ROWS, width), Robot_List[i], False)
                                if ret_value == False:
                                    print("False_MOVING = STAY!")
                                if Robot_List[i].path and Robot_List[i].path[0] != Robot_List[i].curr and timer_count == 1:
                                    Robot_List[i].path.insert(0, Robot_List[i].curr)
                                print(f"after ret_value {ret_value}\n")
                                for tile in robot.path:
                                    print(f"{tile.row},{tile.col}")
                            else:
                                print("\npoints to avoid not interuped")
                            robot.points_to_avoid = ""
                        print(f"for robot numer: {i} pos {robot.curr.row},{robot.curr.col} ")
                        if(len(robot.path) >=1):
                            print(f"next {robot.path[0].row},{robot.path[0].col}")
                        #robot.curr.reset()
                        robot.curr = robot.path[0]
                        make_start_at = robot.path[0].get_pos()
                        print(f"robot {i} is printing {make_start_at} at function and reseting {robot.path[0].get_pos()}")
                        Spot.make_start(robot.path[0], robot.color_index)
                        Spot.make_start(robot.curr, robot.color_index)
                        Spot.make_end(robot.path[-1], robot.color_index)
                        if robot.curr == robot.end:
                            print(f"Robot {i+1} Ended course")
                            SIMULATE = False
                        #next_spot = robot.path[0]
                        #
                        #for target_color in robot.impact_lookat:
                         #   if target_color.prob != None:
                          #      SIMULATE = False
                                ###print(f"Robot {i + 1} Got a robot in his course")
                        if len(robot.path[1:]) > 0:
                            robot.path = robot.path[1:]
                        print(f"{i} pos {robot.curr.row},{robot.curr.col} next {robot.path[0].row},{robot.path[0].col} end {robot.end.row},{robot.end.col}\n")


    pygame.quit()


main(WIN, WIDTH)

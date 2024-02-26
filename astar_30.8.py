import pygame
import numpy as np
from queue import PriorityQueue
import time
import probability as prob

ROWS = 20
WIDTH = 800
show_open_close = True
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

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

class points_to_look:
    def __init__(self, color, points):
        self.color = color
        self.points = points

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
        for spot in search_list:
            if len(lookat_list) > 0:
                for i, spot_hist in enumerate(lookat_list):
                    if spot.color == spot_hist.color:
                        spot_hist.points.append(spot)
                        #get_prob = prob([0.5, 0.1, 0.2] , calc_grad(spot_hist.points), spot_hist, 3)
                        #get_prob.
                        print(f"robot_number: {self.priority} saw a robot go: {calc_grad(spot_hist.points)}")
                        lookat_list[i] = spot_hist
                        res = lookat_list
                    else:
                        #print("new point")
                        res.append(points_to_look(spot.color, [spot]))
            else:
                #print("first point")
                res.append(points_to_look(spot.color, [spot]))
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
        #print("in start " + str(robot_color.start_color) + "pos = " + str(self.get_pos()))
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

def reconstruct_path(came_from, current, draw, Robot):
    path = []
    new_path = []
    while current in came_from:
    #while current in Robot.path:
        current = came_from[current]
        #current = current.reverse()
        #current.make_path(Robot.color_index)
        path.append(current)
    for current in reversed(path[:-1]):
        new_path.append(current)
        current.make_path(Robot.color_index)
        draw()
    Robot.path = new_path


def algorithm(draw, Robot):
    start = Robot.start
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
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw,Robot)
            end.make_end(Robot.color_index)
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

        draw()#Need to fix here the draw

        if current != start:
            current.make_closed()

    return False

def calc_grad(points):
    last_point = points[-1]
    sec_to_last_point = points[-2]
    calc_row = last_point.row - sec_to_last_point.row
    calc_col = last_point.col - sec_to_last_point.col
    if calc_row == 1:
        return "RIGHT"
    elif calc_row == -1:
        return "LEFT"
    elif calc_col == 1:
        return "DOWN"
    elif calc_col == -1:
        return "UP"
    elif calc_col == 0 and calc_row == 0:
        return "STAY"
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

def update_rad(robot):
    robot.impact_rad_points = robot.generate_circle(robot.curr, robot.impact_rad)
    robot.impact_points = robot.points_inside_circle(robot.curr, robot.impact_rad)
    robot.impact_collision = robot.check_circle(robot.impact_points)
    #print(f"impact_collision: {robot.impact_collision}")
    robot.impact_lookat = robot.lookat(robot.impact_lookat ,robot.impact_collision)
    robot.search_rad_points = robot.generate_circle(robot.curr, robot.search_rad)
    robot.search_points = robot.points_inside_circle(robot.curr, robot.search_rad)
    robot.search_collision = robot.check_circle(robot.search_points)
    robot.search_lookat = robot.lookat(robot.search_lookat, robot.search_collision)
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
                        Robot_List.append(Robot(spot, False, spot, 1, COLORS[robot_index], robot_index, 3, 6, None))
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

            if event.type == pygame.KEYDOWN:
                for i in range(robot_index):
                    if event.key == pygame.K_SPACE: #and robot_start_array[i] and robot_end_array[i]:
                        print("Start calculating path")
                        start_time = time.time()
                        for row in GRID:
                            for spot in row:
                                spot.update_neighbors()

                        algorithm(lambda: draw(win, ROWS, width), Robot_List[i])
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

                if event.key == pygame.K_g:
                    draw_blank(win, width, ROWS)
                    for robot in Robot_List:
                        robot.curr = robot.path[0]
                        Spot.make_start(robot.curr, robot.color_index)
                        Spot.make_end(robot.path[-1], robot.color_index)

                    timer_count += 1
                    print(f"time: {timer_count}")
                    for i, robot in enumerate(Robot_List):
                        #print(f"for robot numer: {i + 1} the path length is: {len(robot.path) - 1}")
                        robot.curr = robot.path[0]
                        Spot.make_start(robot.curr, robot.color_index)
                        Spot.make_end(robot.path[-1], robot.color_index)
                        update_rad(robot)
                        if len(robot.path[1:]) > 0:
                            robot.path = robot.path[1:]

    pygame.quit()


main(WIN, WIDTH)

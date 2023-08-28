import networkx as nx
import matplotlib.pyplot as plt
"""
def build_Graph(Size_x, Size_y):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(0, Size_x * Size_y)])
    for x in range(Size_x):
        for y in range(Size_y):
            if x>0 and y>0 and x<Size_x-1 and y<Size_x-1 :
                G.add_edges_from([(x + Size_x*y, x + Size_x*(y+1)), (x + Size_x*y, x+1 + Size_x*y), (x + Size_x*y, x-1 + Size_x*y), (x + Size_x*y, x + Size_x*(y-1))])
            elif x==0 and y==0:
                G.add_edges_from([(x + Size_x*y, x + Size_x*(y+1)), (x + Size_x*y, x+1 + Size_x*y)])
            elif x == Size_x-1 and y == Size_x-1:
                G.add_edges_from([(x + Size_x*y, x-1 + Size_x*y), (x + Size_x*y, x + Size_x*(y-1))])
            elif x == 0:
                G.add_edges_from([(x + Size_x * y, x + Size_x * (y + 1)), (x + Size_x * y, x + 1 + Size_x * y), (x + Size_x * y, x + Size_x * (y - 1))])
            elif y == 0:
                G.add_edges_from([(x + Size_x * y, x + Size_x * (y + 1)), (x + Size_x * y, x + 1 + Size_x * y),(x + Size_x * y, x + Size_x * (y - 1))])
            elif x == 10:
                G.add_edges_from([(x + Size_x*y, x + Size_x*(y+1)), (x + Size_x*y, x-1 + Size_x*y), (x + Size_x*y, x + Size_x*(y-1))])
            elif y == 10:
                G.add_edges_from([(x + Size_x * y, x + 1 + Size_x * y), (x + Size_x * y, x - 1 + Size_x * y), (x + Size_x * y, x + Size_x * (y - 1))])
    return G
G = build_Graph(11, 11)
nx.draw(G)
plt.show()
print(nx.astar_path(G,(0,0),(2,2),dist))
"""

import networkx as nx
import numpy as np
import random
import heapq

def build_dependency_matrix(size):
    mat = np.zeros((size*size,size*size))
    for x in range(size):
        for y in range(size):
            #print(mat)
            if x > 0 and y > 0 and x < size - 1 and y < size - 1:
                mat[x + size * y, x + size * (y - 1)] = 1
                mat[x + size * y, x-1 + size * y] = 1
                mat[x + size * y, x + size * (y + 1)] = 1
                mat[x + size * y, x + 1 + size * y] = 1

            elif x == 0 and y == 0:
                mat[x + size * y, x + size * (y + 1)] = 1
                mat[x + size * y, x + 1 + size * y] = 1
            elif x == size - 1 and y == size - 1:
                mat[x + size * y, x + size * (y - 1)] = 1
                mat[x + size * y, x - 1 + size * y] = 1
            elif x == 0 and y == size-1:
                mat[x + size * y, x + size * (y - 1)] = 1
                mat[x + size * y, x + 1 + size * y] = 1
            elif x == size - 1 and y == 0:
                mat[x + size * y, x + size * (y + 1)] = 1
                mat[x + size * y, x - 1 + size * y] = 1
            elif x == 0:
                mat[x + size * y, x + size * (y - 1)] = 1
                mat[x + size * y, x + size * (y + 1)] = 1
                mat[x + size * y, x + 1 + size * y] = 1
            elif y == 0:
                mat[x + size * y, x-1 + size * y] = 1
                mat[x + size * y, x + size * (y + 1)] = 1
                mat[x + size * y, x + 1 + size * y] = 1
            elif x == size-1:
                mat[x + size * y, x + size * (y - 1)] = 1
                mat[x + size * y, x-1 + size * y] = 1
                mat[x + size * y, x + size * (y + 1)] = 1
            elif y == size-1:
                mat[x + size * y, x + size * (y - 1)] = 1
                mat[x + size * y, x-1 + size * y] = 1
                mat[x + size * y, x + 1 + size * y] = 1
    return mat



def a_star(graph, start, goal, heuristic):
    open_list = [(0, start)]  # Priority queue of nodes to be explored, with their estimated costs
    closed_set = set()  # Set of nodes already explored
    g_score = {node: float('inf') for node in graph.nodes}  # Cost from start to node
    g_score[start] = 0

    while open_list:
        current_cost, current = heapq.heappop(open_list)

        if current == goal:
            path = [goal]
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed_set:
            continue

        closed_set.add(current)

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + 1  # Assuming each edge has a weight of 1

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))

    return None  # No path found

def heuristic(node, goal):
    return np.linalg.norm(np.array(node) - np.array(goal))

def colorize(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def create_wall(mat, walls):
    for node in walls:
        mat[node, :] = 0
    return mat

def inc_wall(walls,size):
    new_walls = walls.copy()
    for wall in walls:
        side = random.randint(0, 3)
        if side == 0 :
            if wall - size > 0:
                new_walls.append(wall - size)
        elif side == 1 :
            if wall % size != 0:
                new_walls.append(wall - 1)
        elif side == 2 :
            if wall + size < size * size -1:
                new_walls.append(wall + size)
        elif (wall + 1) % size != 0:
                new_walls.append(wall + 1)
    return new_walls

def Build_rand_walls_connect(start_node, goal_node, size,wall_count):
    walls = []
    for _ in range(wall_count):
        side = random.randint(0, 1)
        pick_row_col = random.randint(0, size * size - 1)
        pick_pos = [random.randint(0, size - 1) for _ in range(2)]
        min_pos = min(pick_pos)
        max_pos = max(pick_pos)
        for i in range(min_pos, max_pos+1):
            if side == 0:
                row = np.floor(pick_row_col / size).astype(int)
                walls.append(i + row * size)
            if side == 1:
                col = round(pick_row_col % size)
                walls.append(col + i * size)
    walls = list(set(walls))
    if start_node in walls:
        walls.remove(start_node)
    elif goal_node in walls:
        walls.remove(goal_node)
    return walls

size = 20
start_node = 0
goal_node = 1
random_positions = True
if random_positions:
    start_node = random.randint(0, size * size - 1)
    goal_node = random.randint(0, size * size - 1)
    while start_node == goal_node:
        start_node = random.randint(0, size * size - 1)
        goal_node = random.randint(0, size * size - 1)
walls = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Random_walls = False
Random_walls_inc = False
Wall_count = 17
Random_walls_list = [random.randint(0, size * size - 1) for _ in range(Wall_count)]
Build_rand_walls = True
if Random_walls:
    if Random_walls_inc:
        Random_walls_list = inc_wall(Random_walls_list,size)
    walls = Random_walls_list
elif Build_rand_walls:
    walls = Build_rand_walls_connect(start_node, goal_node, size, Wall_count)
came_from = {}
mat = build_dependency_matrix(size)
mat = create_wall(mat, walls)
G = nx.DiGraph(mat)
shortest_path = a_star(G, start_node, goal_node, heuristic)
#print("Shortest Path:", shortest_path)
show_mat = np.zeros((size,size))
flattened_array = show_mat.flatten().astype(int).astype(str)
if shortest_path is not None:
    for i in shortest_path:
        flattened_array[i] = 'P'
        flattened_array[i] = colorize(flattened_array[i], 34)
for i in walls:
    flattened_array[i] = 'W'
    flattened_array[i] = colorize(flattened_array[i], 31)
flattened_array[start_node] = 'S'
flattened_array[start_node] = colorize(flattened_array[start_node], 32)
flattened_array[goal_node] = 'E'
flattened_array[goal_node] = colorize(flattened_array[goal_node], 32)
reshaped_show_mat = flattened_array.reshape(size, size)
if shortest_path != None:
    for row in reshaped_show_mat:
        print(" ".join(row))
else:
    print("No Solotion")
    for row in reshaped_show_mat:
        print(" ".join(row))
print(1)

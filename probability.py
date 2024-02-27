from collections import deque

import numpy as np

class probability:
    def __init__(self, prob=[0.5, 0.1, 0.2], first_movement='up', starting_point=(10, 10),  turns_number= 5):
        self.forward = prob[0]
        self.backward = prob[1]
        self.sideways = prob[2]
        self.first_movement = first_movement
        self.starting_point = starting_point
        self.turns_number = turns_number

    def generate_grid(self, player_position, num_turns):
        grid = [[{} for _ in range(20)] for _ in range(20)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

        # Perform breadth-first search
        queue = deque([(player_position, 0, [])])  # (position, turns_taken, path)

        while queue:
            position, turns_taken, path = queue.popleft()

            if turns_taken > num_turns:
                continue

            if turns_taken not in grid[position[0]][position[1]]:
                grid[position[0]][position[1]][turns_taken] = []

            grid[position[0]][position[1]][turns_taken].append(path + [position])

            for dx, dy in directions:
                new_x, new_y = position[0] + dx, position[1] + dy
                if 0 <= new_x < 20 and 0 <= new_y < 20:
                    queue.append(((new_x, new_y), turns_taken + 1, path + [position]))

        return grid

    # Printing the grid
    def print_grid(self, grid):
        for i in range(20):
            for j in range(20):
                print(f"Position ({i}, {j}):")
                for turn, paths in grid[i][j].items():
                    print(f"Turn {turn}: {paths}")
                print()

    def get_prob(self, last_movement, positions):
        diraction = {
            'up': {
                'up': self.forward,
                'down': self.backward,
                'right': self.sideways,
                'left': self.sideways
            },
            'down': {
                'up': self.backward,
                'down': self.forward,
                'right': self.sideways,
                'left': self.sideways
            },
            'right': {
                'up': self.sideways,
                'down': self.sideways,
                'right': self.forward,
                'left': self.backward
            },
            'left': {
                'up': self.sideways,
                'down': self.sideways,
                'right': self.backward,
                'left': self.forward
            }
        }
        movements = {'(0, -1)': 'left', '(0, 1)': 'right', '(1, 0)': 'down', '(-1, 0)': 'up'}
        first_place = positions[0]
        prob = 0
        for position in positions[1:]:
            dx = position[0] - first_place[0]
            dy = position[1] - first_place[1]
            first_place = position
            if prob == 0:
                prob += diraction[last_movement][movements[str((dx, dy))]]
            else:
                prob *= diraction[last_movement][movements[str((dx, dy))]]
            last_movement = movements[str((dx, dy))]
        return prob

    def calc_prob(self, first_movement, grid, turns):
        prob_array = np.zeros((len(grid), len(grid), turns))
        for i in range(len(grid)):
            for j in range(len(grid)):
                for turn in range(turns):
                    if turn in grid[i][j].keys():
                        for way in grid[i][j][turn]:
                            prob_array[i][j][turn] += self.get_prob(first_movement, way)
        return prob_array

    def print_grid_with_colors(self, grid):
        for turn in range(grid.shape[2]):
            print(f"Turn {turn + 1}:")
            avg = self.get_above_avg(grid,turn)
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if grid[row, col, turn] > 0:
                        # Print in green if the value is greater than 0
                        print("\033[92m{:.2f}\033[0m".format(grid[row, col, turn]), end=" ")
                    else:
                        # Otherwise, print normally
                        print("{:.2f}".format(grid[row, col, turn]), end=" ")
                print()  # Newline for next row
            print()  # Empty line for next turn

    def get_above_avg(selfs,grid,turn):
        count =0
        to_ret = []
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if(grid[row, col, turn] != 0):
                    count +=1
        avg = 1/count#The sum of probabilities is 1 so we just need to count the non-zero
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if(grid[row, col, turn] >= avg):
                    to_ret.append([row,col,turn])#Collecting all the cells with probabilities above avg
        #Those print are for testing, and need to be removed before production
        print(f"sum if {sum} count is {count} avg is {1/count}")
        print(f"to_ret is {to_ret}")
        return to_ret
    def example(self):
        player_position = self.starting_point  # Example player position
        num_turns = self.turns_number
        #prob = self.prob
        movement = self.first_movement
        grid = self.generate_grid(player_position, num_turns)
        #self.print_grid(grid)


        #prob_array = np.zeros((len(grid), len(grid), num_turns))
        prob_array = self.calc_prob(movement, grid, num_turns)
        prob_array[player_position[0]][player_position[1]][0] = 1
        self.print_grid_with_colors(prob_array)

#FORWARD = 0.5
#BACKWARDS = 0.1
#SIDEWAYS = 0.2
#example()
prob = probability()
prob.example()
print(0)







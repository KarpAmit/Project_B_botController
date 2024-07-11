from collections import deque
import numpy as np
ROWS = 20

class probability:
    def __init__(self, prob = [0.75, 0.05, 0.1], first_movement='up', starting_point=(10, 10), turns_number=3,top_percentage=50):
        self.forward = prob[0]
        self.backward = prob[1]
        self.sideways = prob[2]
        self.first_movement = first_movement
        self.starting_point = starting_point
        self.turns_number = turns_number
        self.top_percentage = top_percentage
        self.next_turns_array = self.calc_prob(self.first_movement,
                                               self.generate_grid(self.starting_point, self.turns_number),
                                               self.turns_number)
        self.next_turns_array[self.starting_point[0]][self.starting_point[1]][0] = 1
        self.blocked_in_future = self.get_top_percentage(self.next_turns_array)
        # For debugging
        #print("Next Turns Array:")
        #self.print_grid(self.next_turns_array)

    def generate_grid(self, player_position, num_turns):
        grid = [[{} for _ in range(ROWS)] for _ in range(ROWS)]
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
                if 0 <= new_x < ROWS and 0 <= new_y < ROWS:
                    queue.append(((new_x, new_y), turns_taken + 1, path + [position]))
        return grid

    def print_grid(self, grid):
        turns_number = grid.shape[2]
        for turn in range(turns_number):
            print(f"Turn {turn}:")
            for i in range(ROWS):
                row = ""
                for j in range(ROWS):
                    cell_value = grid[i, j, turn]
                    row += f"{cell_value:<8.2f}"  # Adjust width as needed and format as float
                print(row)
            print()
            # Print additional information for the current turn
            print(f"Details for Turn {turn}:")
            for i in range(ROWS):
                for j in range(ROWS):
                    cell_value = grid[i, j, turn]
                    if cell_value > 0:  # Check if the cell value is greater than zero
                        print(f"Row: {i}, Col: {j}, Probability: {cell_value:.2f}")
            print()

    def get_prob(self, last_movement, positions):
        direction = {
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
            },
            'UNKNOWN': {
                'up': self.forward,
                'down': self.forward,
                'right': self.forward,
                'left': self.forward
            }
        }
        movements = {'(0, -1)': 'left', '(0, 1)': 'right', '(1, 0)': 'down', '(-1, 0)': 'up'}
        first_place = positions[0]
        prob = 1
        for position in positions[1:]:
            delta_rows = position[0] - first_place[0]
            delta_col = position[1] - first_place[1]
            first_place = position
            move = movements[str((delta_col, delta_rows))]
            prob *= direction[last_movement][move]
            last_movement = move
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

    def get_top_percentage(self, grid):
        non_zero = []
        for turn in range(self.turns_number):
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if grid[row, col, turn] > 0:
                        non_zero.append([row, col, turn, grid[row, col, turn]])
        sorted_array = sorted(non_zero, key=lambda x: x[3], reverse=True)
        num_elements = int(len(sorted_array) * (self.top_percentage / 100))
        while num_elements < len(sorted_array):
            if sorted_array[num_elements - 1][-1] == sorted_array[num_elements][-1]:
                num_elements += 1
            else:
                break
        top_elements = sorted_array[:num_elements]
        to_ret = [[] for _ in range(self.turns_number)]
        for elem in top_elements:
            turn = elem[2]
            to_ret[turn].append((elem[0], elem[1]))
        return to_ret

prob = probability()
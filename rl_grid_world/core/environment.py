"""
Nothing added.
"""

import numpy as np

class GridWorld:
    """
    from (0, 0) to (grid_size - 1, grid_size - 1)
    """

    STAR_VALUE = 5
    TRAP_VALUE = 10

    def __init__(self, grid_size = 5):
        """
        Initialize environment
        """
        self.grid_size = grid_size
        self.action_space = [0, 1, 2, 3]
        self.state = np.array([0, 0])
        self.goal = np.array([grid_size - 1, grid_size - 1])

        self.obstacles = np.zeros((grid_size, grid_size), dtype = bool)
        self.obstacles[1, 1] = True
        self.obstacles[2, 3] = True
        self.obstacles[3, 2] = True

        self.stars = np.zeros((grid_size, grid_size), dtype = bool)
        self.starsEaten = np.zeros((grid_size, grid_size), dtype = bool)
        self.stars[1, 3] = True
        self.stars[3, 3] = True

        self.traps = np.zeros((grid_size, grid_size), dtype = bool)
        self.trapsFallen = np.zeros((grid_size, grid_size), dtype = bool)
        self.traps[2, 0] = True
        self.traps[2, 4] = True

    def validblock(self, row, column):
        return (
            0 <= row < self.grid_size and
            0 <= column < self.grid_size and
            not self.obstacles[row, column]
        )
    
    def emptyblock(self, row, column):
        return (
            self.validblock(row, column) and
            not self.stars[row, column] and
            not self.traps[row, column] and
            not (row == 0 and column == 0) and
            not (row == self.grid_size - 1 and column == self.grid_size - 1)
        )
    
    def add_obstacle(self, row, column):
        if self.validblock(row, column):
            self.obstacles[row, column] = True
            return True
        else:
            return False
    
    def _check_star(self, row, column):
        return (self.stars[row, column] and not self.starsEaten[row, column])
    
    def add_star(self, row, column):
        if self.emptyblock(row, column):
            self.stars[row, column] = True
            return True
        else:
            return False
    
    def _check_trap(self, row, column):
        return (self.traps[row, column] and not self.trapsFallen[row, column])
    
    def add_trap(self, row, column):
        if self.emptyblock(row, column):
            self.traps[row, column] = True
            return True
        else:
            return False

    def reset(self):
        """
        reset the state, include of array starsEaten and trapFallen
        """
        self.state = np.array([0, 0])
        self.starsEaten = np.zeros((self.grid_size, self.grid_size), dtype = bool)
        self.trapsFallen = np.zeros((self.grid_size, self.grid_size), dtype = bool)
        return self.state.copy()

    def move(self, action):
        """
        return [next_state, reward(float), done(bool), info]
        info: [moved, eatstar, falltrap]
        """
        new_state = self.state.copy()

        if action == 0:
            new_state[0] -= 1
        elif action == 1:
            new_state[0] += 1
        elif action == 2:
            new_state[1] -= 1
        elif action == 3:
            new_state[1] += 1

        reward = -1
        done = False
        moved = self.validblock(new_state[0], new_state[1])
        eatstar = False
        falltrap = False

        if moved:
            self.state = new_state

            if np.array_equal(self.state, self.goal):
                done = True
            
            if self._check_star(self.state[0], self.state[1]):
                reward += self.STAR_VALUE
                eatstar = True
                self.starsEaten[self.state[0], self.state[1]] = True
            
            if self._check_trap(self.state[0], self.state[1]):
                reward -= self.TRAP_VALUE
                falltrap = True
                self.trapsFallen[self.state[0], self.state[1]] = True
        
        return self.state.copy(), reward, done, [moved, eatstar, falltrap]
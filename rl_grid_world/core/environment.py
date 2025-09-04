"""
增加了只能吃一次的奖励和惩罚
删掉了清空障碍的函数，如果需要重设置以后再加
"""

import numpy as np

class GridWorld:
    """
    智能体在 5x5 网格上从 (0,0) 到 (4,4)
    每走一步得到 -1 的奖励，到达获得 0 奖励并结束
    """

    STAR_VALUE = 5
    TRAP_VALUE = 10

    def __init__(self, grid_size = 5):
        """
        初始化环境: grid_size 表示网格大小
        """
        self.grid_size = grid_size
        self.action_space = [0, 1, 2, 3]
        self.state = np.array([0, 0])
        self.goal = np.array([grid_size - 1, grid_size - 1])

        # 设置障碍物
        self.obstacles = np.zeros((grid_size, grid_size), dtype = bool)
        self.obstacles[1, 1] = True
        self.obstacles[2, 3] = True
        self.obstacles[3, 2] = True

        # 设置奖励
        self.stars = np.zeros((grid_size, grid_size), dtype = bool)
        self.starsEaten = np.zeros((grid_size, grid_size), dtype = bool)
        self.stars[1, 3] = True
        self.stars[3, 3] = True

        # 设置陷阱
        self.traps = np.zeros((grid_size, grid_size), dtype = bool)
        self.trapsFallen = np.zeros((grid_size, grid_size), dtype = bool)
        self.traps[2, 0] = True
        self.traps[2, 4] = True

    def validblock(self, row, column):
        """
        是否是合法位置
        """
        return (
            0 <= row < self.grid_size and
            0 <= column < self.grid_size and
            not self.obstacles[row, column]
        )
    
    def emptyblock(self, row, column):
        """
        是否是空格
        """
        return (
            self.validblock(row, column) and
            not self.stars[row, column] and
            not self.traps[row, column] and
            not (row == 0 and column == 0) and
            not (row == self.grid_size - 1 and column == self.grid_size - 1)
        )
    
    def add_obstacle(self, row, column):
        """
        添加一个障碍物
        """
        if self.validblock(row, column):
            self.obstacles[row, column] = True
            return True
        else:
            return False
    
    def _check_star(self, row, column):
        return (self.stars[row, column] and not self.starsEaten[row, column])
    
    def add_star(self, row, column):
        """
        增加一个奖励
        """
        if self.emptyblock(row, column):
            self.stars[row, column] = True
            return True
        else:
            return False
    
    def _check_trap(self, row, column):
        return (self.traps[row, column] and not self.trapsFallen[row, column])
    
    def add_trap(self, row, column):
        """
        增加一个惩罚
        """
        if self.emptyblock(row, column):
            self.traps[row, column] = True
            return True
        else:
            return False

    def reset(self):
        """
        重置状态，包括奖励和惩罚是否被吃过
        """
        self.state = np.array([0, 0])
        self.starsEaten = np.zeros((self.grid_size, self.grid_size), dtype = bool)
        self.trapsFallen = np.zeros((self.grid_size, self.grid_size), dtype = bool)
        return self.state.copy()

    def move(self, action):
        """
        走一步 action(int) : 0-3
        返回 [next_state, reward(float), done(bool), info]
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
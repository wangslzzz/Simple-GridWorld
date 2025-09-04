import numpy as np

class GridWorld:
    """
    智能体在 5x5 网格上从 (0,0) 到 (4,4)
    每走一步得到 -1 的奖励，到达获得 0 奖励并结束
    """

    def __init__(self, grid_size = 5):
        """
        初始化环境: grid_size 表示网格大小
        """
        self.grid_size = grid_size
        self.action_space = [0, 1, 2, 3]
        self.state = np.array([0, 0])
        self.goal = np.array([grid_size - 1, grid_size - 1])

        # 设置
        self.obstacles = np.zeros((grid_size, grid_size), dtype = bool)
        self.obstacles[1, 1] = True
        self.obstacles[2, 3] = True
        self.obstacles[3, 2] = True

    def reachable(self, row, column):
        """
        是否是合法位置
        """
        return (
            0 <= row < self.grid_size and
            0 <= column < self.grid_size and
            not self.obstacles[row, column]
        )
    
    def add_obstacles(self, row, column):
        """
        添加一个障碍物
        """
        if self.reachable(row, column):
            self.obstacles[row, column] = True
            return True
        else:
            return False
    
    def clear_obstacles(self):
        """
        清空障碍物
        """
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype = bool)
    
    def reset(self):
        """
        重置状态
        """
        self.state = np.array([0, 0])
        return self.state.copy()

    def move(self, action):
        """
        走一步 action(int) : 0-3
        返回 [next_state, reward(float), done(bool), info]
        info: 用于调试，当前为空
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
        moved = self.reachable(new_state[0], new_state[1])

        if moved:
            self.state = new_state
            if np.array_equal(self.state, self.goal):
                done = True
        
        return self.state.copy(), reward, done, [moved]
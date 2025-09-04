"""
测试脚本: 随机运行一回合
"""

import numpy as np
from rl_grid_world.core.environment import GridWorld

def print_grid(env, state):
    """
    简单打印当前网格状态
    A: 智能体位置
    G: 目标位置
    X: 障碍物
    .: 空位置
    """
    for i in range(env.grid_size):
        row = []
        for j in range(env.grid_size):
            if np.array_equal([i, j], state):
                row.append('A')  # 智能体
            elif np.array_equal([i, j], env.goal):
                row.append('G')  # 目标
            elif env.obstacles[i, j]:
                row.append('#')  # 障碍物
            elif env.stars[i, j] and not env.starsEaten[i, j]:
                row.append('$')  # 奖励
            elif env.traps[i, j] and not env.trapsFallen[i, j]:
                row.append('*')  # 惩罚
            else:
                row.append('.')  # 空位置
        print(' '.join(row))
    print()

def main():
    env = GridWorld()

    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    crashed_time = 0
    star_count = 0
    trap_count = 0

    print(f"===Round start===\n")

    print_grid(env, state)

    while not done:
        action = np.random.choice(env.action_space)
        next_state, reward, done, info = env.move(action)

        total_reward += reward
        step_count += 1
        if info[0]:
            crashed_time += 1
        if info[1]:
            star_count += 1
        if info[2]:
            trap_count += 1

        state = next_state

        # 是否正常
        if step_count % 100 == 0:
            print(f"step {step_count}")
            print_grid(env, state)

        if step_count == 150:
            break
    
    print(f"reach to goal: {done}")
    print(f"step: {step_count}")
    print(f"total reward: {total_reward}")
    print(f"crashed time: {crashed_time}")
    print(f"star: {star_count}")
    print(f"trap: {trap_count}")

    print("\n===Round end!===")

if __name__ == "__main__":
    main()
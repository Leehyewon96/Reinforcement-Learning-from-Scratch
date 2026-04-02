import numpy as np

np.random.seed(0) # 시드 고정
rewards = []

for n in range(1,11): # 10번 플레이
    reward = np.random.rand() # 보상 (무작위 수로 시뮬레이션)
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)
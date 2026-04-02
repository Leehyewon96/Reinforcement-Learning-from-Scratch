import numpy as np

class Bandit:
    def __init__(self, arms = 10): # arms = 슬롯머신 대수
        self.rates = np.random.rand(arms) # 슬롯머신 각각의 승률 설정(무작위)
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        

bandit = Bandit()
Q = 0

for i in range(3):
    print(bandit.play(i))

for n in range(1, 11): # 10번 반복
    reward = bandit.play(0) # 0번째 슬롯머신 플레이
    Q += (reward - Q) / n # 가치 추정치 갱신
    print(Q)

print('------------------------------------------')

bandit = Bandit()
Qs = np.zeros(10) # 각 슬롯머신의 가치 추정치
ns = np.zeros(10) # 각 슬롯머신의 플레이 횟수

for n in range(10):
    action = np.random.randint(0, 10) # 무작위 행동(임의의 슬롯머신 선택)
    reward = bandit.play(action)

    ns[action] += 1 # action번째 슬롯머신을 플레이한 횟수 증가
    Qs[action] += (reward - Qs[action]) / ns[action] # action번째 슬롯머신의 가치 추정치 갱신
    print(Qs)

print('------------------------------------------')

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon # 무작위로 행동할 확률 (탐색 확률)
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward): # 슬롯머신의 가치 추정
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self): # 행동 선택
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs)) # 무작위 행동 선택
        return np.argmax(self.Qs) # 탐욕 행동 선택


import matplotlib.pyplot as plt # matplotlib 임포트

steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon)
total_reward = 0
total_rewards = [] # 보상 합
rates = [] # 승률

for step in range(steps):
    action = agent.get_action() # 행동선택
    reward = bandit.play(action) # 플레이하고 보상 받기
    agent.update(action, reward) # 행동에 대한 가치 추정치 갱신
    total_reward += reward

    total_rewards.append(total_reward) # 현재까지의 보상 합 저장
    rates.append(total_reward / (step + 1)) # 현재까지의 승률 저장

print(total_reward)

# 그래프 그리기 : 단계별 보상 총합
plt.ylabel('Total Reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()


# 그래프 그리기 : 단계별 승률
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()
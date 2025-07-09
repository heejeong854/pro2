import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# 환경 정의
class Env:
    def __init__(self):
        self.size = 5
        self.state = (0, 0)
        self.goal = (4, 4)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.size and 0 <= ny < self.size:
            self.state = (nx, ny)
            reward = 50 if self.state == self.goal else -1
            done = self.state == self.goal
        else:
            reward = -5
            done = False
        return self.state, reward, done

# 에이전트 정의
class Agent:
    def __init__(self):
        self.q_table = np.zeros((5, 5, 4))
        self.temp = 1.0

    def softmax(self, q_values):
        exp_q = np.exp(q_values / self.temp)
        return exp_q / np.sum(exp_q)

    def choose_action(self, state):
        x, y = state
        return np.random.choice(4, p=self.softmax(self.q_table[x, y]))

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        self.q_table[x, y, action] += 0.1 * (reward + 0.95 * np.max(self.q_table[nx, ny]) - self.q_table[x, y, action])
        self.temp = max(0.1, self.temp * 0.95)

# 시각화 함수
def plot_path(env, agent, ep):
    path = [env.reset()]
    for _ in range(30):
        action = agent.choose_action(path[-1])
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break
    fig, ax = plt.subplots(figsize=(4, 4))
    grid = np.zeros((5, 5))
    grid[env.goal] = 2
    for x, y in path:
        if grid[x, y] == 0:
            grid[x, y] = 1
    ax.imshow(grid, cmap='hot')
    ax.plot([y for _, y in path], [x for x, _ in path], 'b-o')
    st.pyplot(fig)
    plt.close(fig)

# 메인 앱 실행
def main():
    st.title("Q-learning 경로 최적화")
    episodes = int(st.number_input("에피소드 수", min_value=50, max_value=200, value=100))
    if st.button("학습 시작"):
        env = Env()
        agent = Agent()
        rewards = []
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            for _ in range(30):  # 충분한 탐색 시간 확보
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
            if ep % (episodes // 5) == 0 or ep == episodes - 1:
                st.write(f"에피소드 {ep+1}, 보상: {total_reward}")
                plot_path(env, agent, ep)

        # 보상 그래프
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(rewards)
        ax.set_title("에피소드별 총 보상")
        ax.set_xlabel("에피소드")
        ax.set_ylabel("총 보상")
        st.pyplot(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_cumulative_reward(rewards):
    cumulative_rewards = np.cumsum(rewards)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cumulative_rewards, color='green')
    ax.set_title("누적 보상 그래프 (Cumulative Reward)")
    ax.set_xlabel("에피소드")
    ax.set_ylabel("누적 보상")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

import gymnasium as gym
from PPO import * 
from reporter import *

def main():


    max_episode_num = 1000  # 최대 에피소드 설정
    env = gym.make("Pendulum-v1", g=9.81)
    agent = PPOagent(env)  # DDPG 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()


if __name__=="__main__":
    main()

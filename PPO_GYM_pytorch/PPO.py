import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, cat
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from save_torch_model import saveasONNX

from time import time, localtime, strftime
from reporter import *
import os
import sys
## PPO 액터 신경망
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.h1 = nn.Linear(state_dim, 64)
        self.h2 = nn.Linear(64,32)
        self.h3 = nn.Linear(32,16)
        self.mu = nn.Linear(16, action_dim)
        self.std = nn.Linear(16, action_dim)

    def forward(self, state):
        x = self.h1(state)
        x = self.relu(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.h3(x)
        x = self.relu(x)

        mu = self.mu(x)
        mu = self.tanh(mu)
        std = self.std(x)
        std = self.softplus(std)

        # 평균값을 [-action_bound, action_bound] 범위로 조정
        mu = mu*self.action_bound

        return [mu, std]


## PPO 크리틱 신경망
class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.relu = nn.ReLU()

        self.h1 = nn.Linear(state_dim, 64)
        self.h2 = nn.Linear(64,32)
        self.h3 = nn.Linear(32,16)
        self.v = nn.Linear(16,1)


    def forward(self, state):
        x = self.h1(state)
        x = self.relu(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.h3(x)
        x = self.relu(x)
        v = self.v(x)
    
        return v


## PPO 에이전트 클래스
class PPOagent(object):

    def __init__(self, env):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.GAE_LAMBDA = 0.9
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.05
        self.EPOCHS = 5

        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = env.observation_space.shape[0]
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        # 행동의 최대 크기
        self.action_bound = env.action_space.high[0]
        # 표준편차의 최솟값과 최댓값 설정
        self.std_bound = [1e-2, 1.0]

        # 액터 신경망 및 크리틱 신경망 생성
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)


        # 옵티마이저
        self.Actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.ACTOR_LEARNING_RATE)
        self.Critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr =self.CRITIC_LEARNING_RATE)

        # self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = [-1e4]
        self.all_epi_reward = []


    ## 로그-정책 확률밀도함수 계산
    def log_pdf(self, mu, std, action):
        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])               # clip the value, std_bound[0] : 0.01, std_bound[1] : 1
        var = std.pow(2)
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * torch.log(var * 2 * np.pi)
        return torch.sum(log_policy_pdf, dim=1, keepdims=True)


    ## 액터 신경망으로 정책의 평균, 표준편차를 계산하고 행동 샘플링
    def get_policy_action(self, state):
        state =state.reshape(-1)
        mu_a, std_a = self.actor(torch.as_tensor(state, device = self.device).type(torch.float32))
        if torch.any(torch.isnan(mu_a)) or torch.any(torch.isnan(std_a)):
            self.reporter.info("Nan Occurs in get_policy_action")
            self.reporter.info("mu_a : ", mu_a)
            self.reporter.info("std_a : ", std_a)
            mu_a = torch.nan_to_num(mu_a, nan=1)
            std_a = torch.nan_to_num(std_a, nan=1e-2)
            sys.exit()
        mu_a = mu_a.detach().cpu().numpy()
        std_a = std_a.detach().cpu().numpy()
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return mu_a, std_a, action


    ## GAE와 시간차 타깃 계산
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets


    ## 배치에 저장된 데이터 추출
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack


    ## 액터 신경망 학습
    def actor_learn(self, log_old_policy_pdf, states, actions, gaes):
        #print("gaes : ", gaes)
        #gaes = gaes.detach().cpu().numpy()
        #gaes = torch.tensor(gaes).to(self.device)

        self.actor.train()
        mu, std = self.actor(torch.as_tensor(states, device = self.device).type(torch.float32))
        log_pdf = self.log_pdf(mu, std, actions)
        ratio = torch.exp(log_pdf - log_old_policy_pdf)
        clipped_ratio = torch.clamp(ratio, 1.0-self.RATIO_CLIPPING, 1.0+self.RATIO_CLIPPING)
        surrogate = -torch.min(ratio * gaes, clipped_ratio * gaes)
        loss = torch.mean(surrogate)

        self.Actor_optimizer.zero_grad()
        loss.backward()
        self.Actor_optimizer.step()


    ## 크리틱 신경망 학습
    def critic_learn(self, states, td_targets):
        #print("states : ", states)
        #print("td_targets : ", td_targets)
        #states = states.detach().cpu().numpy()
        #td_targets = td_targets.detach().cpu().numpy()
        #td_targets = torch.tensor(td_targets, device=self.device)

        self.critic.train()
        #td_hat = self.critic(torch.as_tensor(states, device=self.device).type(torch.float32))
        td_hat = self.critic(states).type(torch.float32)
        advantage = td_hat - td_targets
        loss = torch.mean(torch.square(advantage))

        self.Critic_optimizer.zero_grad()
        loss.backward()
        self.Critic_optimizer.step()


    ## 에이전트 학습
    def train(self, max_episode_num):

        cur_time = strftime("%m%d_%I%M%p", localtime(time()))
        log_name = cur_time + ".log"
        reporter = reporter_loader("info", log_name)
        onnx_path = "./savemodel/"  + cur_time + "/"
        os.makedirs(onnx_path)

        # 배치 초기화
        batch_state, batch_action, batch_reward = [], [], []
        batch_log_old_policy_pdf = []

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            # 에피소드 초기화
            timestep, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state, info = self.env.reset()

            while not done:

                # 환경 가시화
                #self.env.render()
                # 이전 정책의 평균, 표준편차를 계산하고 행동 샘플링
                mu_old, std_old, action = self.get_policy_action(state)
                # 행동 범위 클리핑
                action = np.clip(action, -self.action_bound, self.action_bound)
                # 이전 정책의 로그 확률밀도함수 계산
                var_old = std_old ** 2
                log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)
                # 다음 상태, 보상 관측
                next_state, reward, done, truncated, _ = self.env.step(action)

                if done or truncated:
                    print(done, truncated)
                    break

                # shape 변환
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])
                log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1, 1])
                # 학습용 보상 설정
                train_reward = (reward + 8) / 8
                # 배치에 저장
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(train_reward)
                batch_log_old_policy_pdf.append(log_old_policy_pdf)

                # 배치가 채워질 때까지 학습하지 않고 저장만 계속
                if len(batch_state) < self.BATCH_SIZE:
                    # 상태 업데이트
                    state = next_state
                    episode_reward += reward[0]
                    timestep += 1
                    continue

                # 배치가 채워지면, 학습 진행
                # 배치에서 데이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)
                # 배치 비움
                batch_state, batch_action, batch_reward, = [], [], []
                batch_log_old_policy_pdf = []
                # GAE와 시간차 타깃 계산
                next_v_value = self.critic(torch.as_tensor(next_state, device = self.device, dtype=torch.float32))
                v_values = self.critic(torch.as_tensor(states, device = self.device, dtype=torch.float32)) 
                gaes, y_i = self.gae_target(rewards, v_values.detach().cpu().numpy(), next_v_value.detach().cpu().numpy(), done)

                # 에포크만큼 반복
                for _ in range(self.EPOCHS):
                    # 액터 신경망 업데이트
                    self.actor_learn(torch.as_tensor(log_old_policy_pdfs, device = self.device, dtype=torch.float32),
                                     torch.as_tensor(states, device = self.device, dtype=torch.float32),
                                     torch.as_tensor(actions, device = self.device, dtype=torch.float32),
                                     torch.as_tensor(gaes, device = self.device, dtype=torch.float32))
                    # 크리틱 신경망 업데이트
                    self.critic_learn(torch.as_tensor(states, device = self.device, dtype=torch.float32),
                                      torch.as_tensor(y_i, device = self.device, dtype=torch.float32))

                # 다음 에피소드를 위한 준비
                state = next_state
                episode_reward += reward[0]
                timestep += 1

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep+1, 'Time: ', timestep, 'Reward: ', episode_reward)

            if episode_reward > self.save_epi_reward[-1]:           # -1로 하면 자기 자신과 비교하게 됨.
                
                saveasONNX(self.actor, self.critic, self.device, reporter, ep+1, onnx_path, episode_reward)
                print(" [ Save improved model ] ")
                self.save_epi_reward.append(episode_reward)
            
            self.all_epi_reward.append(episode_reward)
            #print("self.save_epi_reward : ", self.save_epi_reward)
            #print("self.save_epi_reward[-1] : ", self.save_epi_reward[-1])
            print("episode_reward : ", episode_reward)

            if (ep+1) % 1 == 0:
            # agent.Actor_model.save_weights('DDPG', save_format='tf')
                plt.plot(self.all_epi_reward, "r",label = "reward")

                if ep == 0:
                    plt.legend()
                # plt.show(block = False)
                plt.savefig('./PPO.png')


        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        #print(self.save_epi_reward)
    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()


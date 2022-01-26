from select import select
from dotmap import DotMap
import torch
import numpy as np
import random
import gym
import time
import agents
import math
import os
import matplotlib.pyplot as plt


def store_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, is_terminal: bool) -> None:
    """
    Store a replay if this agent supports experience replays
    """
    raise NotImplementedError("Agent does not support experience replays")
    
## Parameters
# General
seed = 42
device = "cpu" #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
selected_agent = "StandardSAC"
fixed_time = time.time()

# Environment
environment_name = "BipedalWalker-v3"
video_every = 10
max_episodes = 20000
max_timesteps = 5000

# Setup environment
env = gym.make(environment_name)
env = gym.wrappers.Monitor(env, f"./{selected_agent}/{fixed_time}/videos/", video_callable=lambda ep_id: ep_id%video_every == 0, force=True)
obs_dim = env.observation_space.shape[0] # type: ignore
act_dim = env.action_space.shape[0] # type: ignore

# Reproducability
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

# Logging
ep_reward = 0
reward_list = []
plot_data = []
log_folder = f"./{selected_agent}/{fixed_time}/logs/"
os.makedirs(log_folder)
graph_folder = f"./{selected_agent}/{fixed_time}/graphs/"
os.makedirs(graph_folder)

log_f = open(f"{log_folder}agent-log.txt", "w+")
training_log = open(f"{log_folder}training-log-0.txt", "w+")
plot_interval = 10

def log(message, epoch, timestep):
    training_log.write(f"[Epoch {epoch}:{timestep}] {message}\n")

# Agent
agents = {
    "StandardTD3": {
        "agent": agents.StandardTD3Agent,
        "parameters": DotMap({
            "buffer_size": 1000000,
            "lr": 3e-4,
            "noise_sigma": 0.2,
            "tau": 0.005,
            "replay_batch_size": 256,
            "noise_clip": 0.5,
            "gamma": 0.99,
            "policy_update_frequency": 2,
            "exploration_noise": 0.1,
            "warmup_function": lambda episode: max(0, 300 - episode)
        }),
        "training": {
            "priority_buffer": False,
            "pre_training_warmup": 0
        }
    },
    "PrioritisedTD3": {
        "agent": agents.PrioritisedTD3Agent,
        "parameters": DotMap({
            "leaf_count": 20,
            "lr": 3e-4,
            "noise_sigma": 0.2,
            "tau": 0.005,
            "replay_batch_size": 256,
            "noise_clip": 0.5,
            "gamma": 0.99,
            "policy_update_frequency": 2,
            "exploration_noise": 0.1,
            "warmup_function": lambda episode: max(0, 300 - episode)
        }),
        "training": {
            "priority_buffer": True,
            "pre_training_warmup": 0
        }
    },
    "StandardSAC": {
        "agent": agents.StandardSACAgent,
        "parameters": DotMap({
            "buffer_size": 1000000,
            "gradient_update_steps": 1,
            "lr": 3e-4,
            "tau": 0.005,
            "gamma": 0.99,
            "start_steps": 10000,
            "replay_batch_size": 256,
            "target_update_interval": 1,
            "warmup_function": lambda episode: 0,
            "log_interval": 10
        }),
        "training": {
            "priority_buffer": False,
            "pre_training_warmup": 10000
        }
    }
}

# Initialise the selected agent
agent = agents[selected_agent]["agent"](obs_dim, act_dim, device, agents[selected_agent]["parameters"], env)

if selected_agent == "StandardSAC":
    agent.run()

exit()

# Complete warmup steps to get information about environment if needed
if agents[selected_agent]["training"]["pre_training_warmup"] != 0:
    warmup_steps = agents[selected_agent]["training"]["pre_training_warmup"]
    done = False
    accumulated_steps = 0
    state = env.reset()

    print("Warmup is enabled, starting...")

    while accumulated_steps < warmup_steps:
        if accumulated_steps % 1000 == 0:
            print(accumulated_steps)

        if done:
            state = env.reset()
            done = False
            
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if agents[selected_agent]["training"]["priority_buffer"]:
            agent.store_memory(state, action, reward, next_state, done, None)
        else:
            agent.store_memory(state, action, reward, next_state, done)
            
        accumulated_steps += 1
    
    print("Completing final run...")

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if agents[selected_agent]["training"]["priority_buffer"]:
            agent.store_memory(state, action, reward, next_state, done, None)
        else:
            agent.store_memory(state, action, reward, next_state, done)

        accumulated_steps += 1

    print(f"MBPO Warmed up! (acc: {accumulated_steps})")
else:
    print("Warmup is disabled")

# Timing
total_times = dict()
episode_start_time = None

for episode in range(1, max_episodes + 1):
    if episode % 10 == 0:
        training_log = open(f"{log_folder}training-log-{episode // 10}.txt", "w+")
    
    # Time logging
    training_log.flush()
    log(f"Summary of time shares so far:", episode, 0)
    total_acc_time_sum = sum(total_times.values())
    log(f"Total accumulated time is {total_acc_time_sum}s", episode, 0)

    for key in total_times.keys():
        pp = 100 * total_times[key] / total_acc_time_sum
        log(f"{key}: {total_times[key]:.4f}s ({pp:.1f}%)", episode, 0)
        
    print(f"Starting episode {episode}")
    eps_start_time = time.time()
    state = env.reset()
    warmup_it = agent.parameters.warmup_function(episode) # max(0, int(1000 * (math.log(1000) - math.log(agent.warmup_iters)) / math.log(1000))) #max(0, agent.warmup_iters - episode)
    is_eval = episode % plot_interval == 0

    #epsilon_greedy = max(reverse_sigmoid(episode, 0.18, 20), reverse_sigmoid(episode, 0.08, 20))
    epsilon_greedy = 0 if episode >= 250 else 1 - (1 / (1 + math.exp(-0.06 * (episode - 80))))
    
    t_time = time.time()
    accumulated_times = dict()

    for t in range(max_timesteps):
        if len(accumulated_times.keys()) != 0:
            total_time = sum([x[0] for x in accumulated_times.values()]) + 1e-5
            log(f"Total time on last step was {total_time}s", episode, t)

            for key in accumulated_times.keys():
                if key not in total_times.keys():
                    total_times[key] = 0
                    total_times[key] += accumulated_times[key][0]
                    percentage = 100 * accumulated_times[key][0] / total_time
                    log(f"{key}: {accumulated_times[key][0]:.4f}s ({percentage:.1f}%) (Info: {accumulated_times[key][1]})", episode, t)
        
        accumulated_times = dict()
        log(f"Starting new timestep, spent {time.time() - t_time}s on previous", episode, t)
        t_time = time.time()

        track_time = time.time()
        action = agent.sample_action(state)
        accumulated_times["sampling"] = [time.time() - track_time, f"Policy sample: {t >= warmup_it}"]

        track_time = time.time()
        # take action in environment and get r and s'
        next_state, reward, done, _ = env.step(action)
        accumulated_times["env_step"] = [time.time() - track_time, None]

        track_time = time.time()

        if agents[selected_agent]["training"]["priority_buffer"]:
            agent.store_memory(state, action, reward, next_state, done, None)
        else:
            agent.store_memory(state, action, reward, next_state, done)

        accumulated_times["store_replay"] = [time.time() - track_time, None]

        state = next_state
        ep_reward += reward
        
        if t==(max_timesteps-1):
            print("Max timesteps")

        # stop iterating when the episode finished
        if done or t==(max_timesteps-1):
            break
    
    agent.train()
    
    # append the episode reward to the reward list
    reward_list.append(ep_reward)

    # do NOT change this logging code - it is used for automated marking!
    log_f.write('episode: {}, reward: {}\n'.format(episode, ep_reward))
    log_f.flush()
    ep_reward = 0
    
    # print reward data every so often - add a graph like this in your report
    if is_eval:
        # disp.clear_output(wait=True)
        plot_data.append([episode, np.array(reward_list).mean(), np.array(reward_list).std()])
        reward_list = []
        # plt.rcParams['figure.dpi'] = 100
        plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:grey')
        plt.fill_between([x[0] for x in plot_data], [x[1]-x[2] for x in plot_data], [x[1]+x[2] for x in plot_data], alpha=0.2, color='tab:grey')
        plt.xlabel('Episode number')
        plt.ylabel('Episode reward')
        plt.savefig(f"{graph_folder}graph-{episode}.png")

exit()

for episode in range(1, max_episodes + 1):
    if episode % 10 == 0:
        training_log = open(f"{log_folder}training-log-{episode // 10}.txt", "w+")
    
    # Time logging
    training_log.flush()
    log(f"Summary of time shares so far:", episode, 0)
    total_acc_time_sum = sum(total_times.values())
    log(f"Total accumulated time is {total_acc_time_sum}s", episode, 0)

    for key in total_times.keys():
        pp = 100 * total_times[key] / total_acc_time_sum
        log(f"{key}: {total_times[key]:.4f}s ({pp:.1f}%)", episode, 0)
        
    print(f"Starting episode {episode}")
    eps_start_time = time.time()
    state = env.reset()
    warmup_it = agent.parameters.warmup_function(episode) # max(0, int(1000 * (math.log(1000) - math.log(agent.warmup_iters)) / math.log(1000))) #max(0, agent.warmup_iters - episode)
    is_eval = episode % plot_interval == 0

    #epsilon_greedy = max(reverse_sigmoid(episode, 0.18, 20), reverse_sigmoid(episode, 0.08, 20))
    epsilon_greedy = 0 if episode >= 250 else 1 - (1 / (1 + math.exp(-0.06 * (episode - 80))))
    
    t_time = time.time()
    accumulated_times = dict()

    for t in range(max_timesteps):
        if len(accumulated_times.keys()) != 0:
            total_time = sum([x[0] for x in accumulated_times.values()]) + 1e-5
            log(f"Total time on last step was {total_time}s", episode, t)

            for key in accumulated_times.keys():
                if key not in total_times.keys():
                    total_times[key] = 0
                    total_times[key] += accumulated_times[key][0]
                    percentage = 100 * accumulated_times[key][0] / total_time
                    log(f"{key}: {accumulated_times[key][0]:.4f}s ({percentage:.1f}%) (Info: {accumulated_times[key][1]})", episode, t)
        
        accumulated_times = dict()
        log(f"Starting new timestep, spent {time.time() - t_time}s on previous", episode, t)
        t_time = time.time()

        track_time = time.time()

        if t < warmup_it or np.random.uniform() < epsilon_greedy:
            # print("Randomised action")
            action = env.action_space.sample()
        else:
            # select the agent action
            sampled = agent.sample_action(state)
            action = (sampled + np.random.normal(0, agent.parameters.exploration_noise, size=act_dim)).clip(-1, 1)

        accumulated_times["sampling"] = [time.time() - track_time, f"Policy sample: {t >= warmup_it}"]

        track_time = time.time()
        # take action in environment and get r and s'
        next_state, reward, done, _ = env.step(action)
        accumulated_times["env_step"] = [time.time() - track_time, None]

        track_time = time.time()

        if agents[selected_agent]["training"]["priority_buffer"]:
            agent.store_memory(state, action, reward, next_state, done, None)
        else:
            agent.store_memory(state, action, reward, next_state, done)

        accumulated_times["store_replay"] = [time.time() - track_time, None]

        state = next_state
        ep_reward += reward

        if t >= warmup_it:
            # track_time = time.time()
            agent.train()
            # accumulated_times["agent_train_total"] = [time.time() - track_time, None]
            accumulated_times = {**accumulated_times, **agent.get_log()}
            agent.clear_log()
        
        if t==(max_timesteps-1):
            print("Max timesteps")

        # stop iterating when the episode finished
        if done or t==(max_timesteps-1):
            break
    
    # append the episode reward to the reward list
    reward_list.append(ep_reward)

    # do NOT change this logging code - it is used for automated marking!
    log_f.write('episode: {}, reward: {}\n'.format(episode, ep_reward))
    log_f.flush()
    ep_reward = 0
    
    # print reward data every so often - add a graph like this in your report
    if is_eval:
        # disp.clear_output(wait=True)
        plot_data.append([episode, np.array(reward_list).mean(), np.array(reward_list).std()])
        reward_list = []
        # plt.rcParams['figure.dpi'] = 100
        plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:grey')
        plt.fill_between([x[0] for x in plot_data], [x[1]-x[2] for x in plot_data], [x[1]+x[2] for x in plot_data], alpha=0.2, color='tab:grey')
        plt.xlabel('Episode number')
        plt.ylabel('Episode reward')
        plt.savefig(f"{graph_folder}graph-{episode}.png")
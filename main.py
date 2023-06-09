import torch
import numpy as np
import argparse
from lib.normalization import Normalization, RewardScaling
from lib.replaybuffer import ReplayBuffer,ReplayBufferV2, ReplayBufferV3, ReplayBufferV2_
from agents.hppo_mine import HPPO
from environment import lifecycle_env
import torch
from datasets import DataSet

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

def evaluate_policy(args, env, agent, state_norm):
    times = 1000
    evaluate_reward = 0
    mu = 0
    single_re = []
    # infu=0
    for _ in range(times):
        s = env.reset()
        _s = state_norm(s)
        done = False
        re = 0
        episode_reward = 0
        while not done:
            a,p = agent.evalate(_s)  # We use the deterministic policy during the evaluating
            if agent.ac_type == 'normal' or agent.ac_type == 'stuT':
                param = (p[a*10:a*10+10]+1)/2
            elif agent.ac_type == 'beta' or agent.ac_type=='gamma':
                param = p[a:(a+1)*2]
            elif agent.ac_type == 'F':
                param = p[a:(a+1)] / 2
            s_, r, done = env.step(s, (a, (param)))
            _s = state_norm(s_)
            episode_reward += r
            mu += param[0]
            # infu += param[1]
            s = s_
            # print(a, r)
        single_re.append(episode_reward)
        evaluate_reward += episode_reward
    # print('infu:', infu/ times)
    return evaluate_reward / times


def main(args, number, seed):
    env = lifecycle_env(DataSet(), 20, 1, [20.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,], [1.0,65.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])
    env_evaluate = lifecycle_env(DataSet(), 20, 1, [20.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,], [1.0,65.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])

    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.state_dim
    args.action_dim = env.action_dim
    args.param_dim = env.param_dim
    args.max_action = 1
    print("env={}".format('Life Cycle cn'))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    device = 1
    replay_buffer = ReplayBufferV2_(args, device)
    agent = HPPO(
        action_dim=env.action_dim,
        state_dim=env.state_dim,
        param_dim=env.param_dim,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        max_train_steps=args.max_train_steps,
        gamma=args.gamma,
        ac_type=args.distribution_type,
        torch_device=device,
    )
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization

    while total_steps < args.max_train_steps:
        s = env.reset()
        _s = state_norm(s)
        episode_steps = 0
        done = False
        reward_scaling.reset()
        while not done:
            episode_steps += 1
            a, a_logprob, p, p_logprob = agent.choose_action(_s)  # Action and the corresponding log probability
            if agent.ac_type == 'normal' or agent.ac_type == 'stuT':
                param = (p[a*10:a*10+10]+1)/2
            elif agent.ac_type == 'beta' or agent.ac_type=='gamma':
                param = p[a:(a+1)*2]
            elif agent.ac_type == 'F':
                param = p[a:(a+1)] / 2
            s_, r, done = env.step(s, (a, param))
            # print(param)
            _s_ = state_norm(s_)
            r_ = reward_scaling(r)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done:
                dw = True
            else:
                dw = False

            replay_buffer.store(_s, a, a_logprob, p, p_logprob, r_, _s_, dw, done)
            s = s_
            _s = _s_
            total_steps += 1
            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.batch_size == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                # Save the rewards
                # if evaluate_num % args.save_freq == 0:
                #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=9e4, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--distribution_type", type=str, default="normal", help="the distribution of ppo continue")
    args = parser.parse_args()    
    main(args, number=1, seed=0)
import torch
import numpy as np
import argparse
from lib.normalization import Normalization, RewardScaling
from lib.replaybuffer import ReplayBuffer,ReplayBufferV2, ReplayBufferV3, ReplayBufferV2_
from agents.hppo_mine import HPPO
from environment import lifecycle_env
import torch
from datasets import DataSet

now_reward = -1000000
ind_1 = [31.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,]
ind_2 = [20.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,]
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
    # global now_reward
    times = 500
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
                param = p[a*10:a*10+10]
            elif agent.ac_type == 'F':
                param = p[a:(a+1)] / 2
            s_, r, done = env.step(s, (a, (param)))
            _s = state_norm(s_)
            episode_reward += r
            mu += param[0]
            # infu += param[1]
            s = s_
            # print(a, r)
        # single_re.append(episode_reward)
        evaluate_reward += episode_reward
    env.record_df.to_csv('./state.csv')
    return evaluate_reward / times


def main(args, number, seed):
    env = lifecycle_env(DataSet(), 25, 1, [25.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,], [1.0,65.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])
    env_evaluate = lifecycle_env(DataSet(), 25, 1, [25.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,], [1.0,65.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])

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
    device = 2
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
    agent.load('./model_behavior/models')
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    # state_norm.running_ms.n = 1591533
    # state_norm.running_ms.mean = np.array([44.013186028817785, 0.9600533950600115, 1562.352058102224, 0.9702936206610618, 2.0, 21527.610066487272, 64202.282596850026, 0.2721929108601513, 243636.88070965503, 6393.39566239046, 35275.78437277017, 210153.36066392314, 683299.8619776118, 0.274740404534331, 0.013140499212051161, 70760.20773127167])
    # state_norm.running_ms.S = np.array([353286153.27798074, 35330.309462488076, 1.5739824634729714e+16, 30440.225694195247, 0.0, 1.4561299158420646e+16, 5266029191979867.0, 315288.94224374654, 4.986939998212595e+16, 951045154152520.0, 6.4192058970218424e+16, 2.1689328276608467e+17, 3.715525474670262e+18, 8048099887.748394, 11077450.415637562, 992447580114931.0])
    # state_norm.running_ms.std = np.array( [14.89894386941037, 0.14899301094210007, 99447.09944640756, 0.13829806701664252, 0.0, 95651.59981847912, 57521.97739953014, 0.4450886767124487, 177014.66496896933, 24445.15213861969, 200831.96103390714, 369160.50124459766, 1527925.9172860868, 71.11133877535111, 2.6382265304037325, 24971.576704237934])
    evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
    evaluate_rewards.append(evaluate_reward)
    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))

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
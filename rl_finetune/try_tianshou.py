import gymnasium as gym
import Callienv.envs.tools as tools

# from gymnasium.wrappers.record_video import RecordVideo  ##

import torch
import time
import numpy as np
import random
import os
import json
import utils

from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv 
from tianshou.policy import SACPolicy
import tianshou as ts

from tianshou.trainer import offpolicy_trainer  

from MLP.model import My_MLP, My_Siren
from tianshou.utils.net.common import Net, DataParallelNet
from tianshou.utils.net.continuous import ActorProb, Critic  

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="env.is_vector_env")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(("Continuous SAC using Tianshou, "
        "for the meaning of some hyper-parameters, "
        "refer to the documentation of Tianshou."
    ))
    # 区分epoch, episode, step
    # Network parameters
    parser.add_argument('--actor_network_shape', default=(128, 256), type=list)
    parser.add_argument('--critic_network_shape', default=(128, 256, 256), type=list)
    parser.add_argument('--learn_fourier', default = False, type=bool)
    parser.add_argument('--sigma', default = 0.1, type=float)
    parser.add_argument('--train_B', default=True, type=bool)
    parser.add_argument('--fourier_dim', default=256, type=int)
    parser.add_argument('--concatenate_fourier', default= True, type=bool)

    # SAC policy
    parser.add_argument('--mu_std_net', default=(64,), type=list)
    parser.add_argument('--actor_lr', default=3e-5, type=float)
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--tau',  default=0.005, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--target_entropy_ratio', default=0.98, type=float)
    parser.add_argument('--buffer_size', default=2**20, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # simulator env nums
    # notice The number of train/test envs must be a divisor of the number of training/test data!
    parser.add_argument('--train_env_num', default=4, type=int)
    parser.add_argument('--test_env_num', default=4, type=int)

    # render mode 这里表示的是录视频
    parser.add_argument('--test_render_mode', default='rgb_array', type=str)
    parser.add_argument('--train_render_mode', default='rgb_array', type=str)

    # data dirs
    ## both img and skel are in the same folder
    parser.add_argument('--train_data_dir',  default=None, type=str)
    parser.add_argument('--test_data_dir',  default=None, type=str)

    # save dirs
    # parser.add_argument('--save_video_dir', default='./result/demo/', type=str)
    parser.add_argument('--save_model_dir', default='./result/models/demo/', type=str)
    parser.add_argument('--save_control_dir', default='./result/demo/arrays/', type=str)
    parser.add_argument('--test_save_dir', default='./result/demo/test/', type=str)
    parser.add_argument('--save_visualize_dir', default='./result/demo/vis/', type=str)
    parser.add_argument('--logdir',  default=None, type=str)

    # tool properties
    parser.add_argument('--which_tool', default='brush', type=str)
    parser.add_argument('--tool_property_dir',  default=None, type=str) ##file recording tool property

    # training
    ## training iterations
    parser.add_argument('--image_iter', default=10, type=int)
    parser.add_argument('--start_update', default=1, type=int)
    parser.add_argument('--update', default=1, type=int)
    

    parser.add_argument('--max_epoch',  default=150, type=int)
    parser.add_argument('--step_per_epoch',  default=10000, type=int) #一个epoch 最多collect多少个transitions
    parser.add_argument('--batch_size',  default=2**11, type=int)
    parser.add_argument('--resume_from_log',  default=None)

    ## step_per_collect & update_per_step: the former hyper-param is used in sampling while the latter is for updating the model.
    parser.add_argument('--step_per_collect',  default=4, type=int)   #相当于走4个step，更新一次网络参数
    parser.add_argument('--update_per_step',  default=2, type=float)

    # testing
    parser.add_argument('--episode_per_test',  default=10, type=int)
    
    args, unknown = parser.parse_known_args()
    
    return args
    

args = parse_args()
# make dirs

# if not os.path.exists(args.save_video_dir):
#     os.makedirs(args.save_video_dir)
if not os.path.exists(args.save_model_dir):
    os.makedirs(args.save_model_dir)
if not os.path.exists(args.save_control_dir):
    os.makedirs(args.save_control_dir)
if not os.path.exists(args.save_visualize_dir):
    os.makedirs(args.save_visualize_dir)
if not os.path.exists(args.test_save_dir):
    os.makedirs(args.test_save_dir)
    
now = int(round(time.time()*1000))
now = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(now/1000))
# save_video_dir = args.save_video_dir+now+'/'
save_model_dir = args.save_model_dir+now+'.pth'

## load tool property: json file
print(f"load tool {args.which_tool} ...")
# 读取文件内容
with open(args.tool_property_dir) as f:
    tp = json.load(f)

if args.which_tool == 'brush':
    tool = tools.Writing_Brush(tp["r_min"], tp["r_max"],
                                tp["l_min"], tp["l_max"],
                                tp["theta_min"], tp["theta_max"],
                                tp["theta_step"])
elif args.which_tool == 'ellipse':
    tool = tools.Ellipse(tp["r_min"], tp["r_max"],
                                tp["l_min"], tp["l_max"],
                                tp["theta_min"], tp["theta_max"],
                                tp["theta_step"])
    
elif args.which_tool == 'marker':
    tool = tools.Chisel_Tip_Marker(tp["r_min"], tp["r_max"],
                                tp["l_min"], tp["l_max"],
                                tp["theta_min"], tp["theta_max"],
                                tp["theta_step"])

# 这里用的是npy数据型，也就是一个裸露的numpy数据 npz数据是一个压缩数组
# 这里返回的是匹配的数量
train_img_num = utils.count_file_num(args.train_data_dir)
test_img_num = utils.count_file_num(args.test_data_dir)

# CalliEnv-v0 这是自定义环境名称
env = gym.make('CalliEnv-v0',tool = tool,
                # 这是训练数据
                folder_path = args.train_data_dir,
                # 表示只跑单环境
                env_num = 1,
                # 随机挑选一张作为训练集
                env_rank=(0,train_img_num),
                render_mode = args.train_render_mode,
                output_path = args.save_control_dir,
                visualize_path = None,
                # 返回的是5元组而不是4元组
                # new_step_api = True  # 移除：Gymnasium 0.29.0默认使用新API
                )

print("make train envs...")
train_envs = SubprocVectorEnv([lambda i=i: 
    gym.make('CalliEnv-v0',
             tool=tool,
             folder_path=args.train_data_dir,
             output_path=args.save_control_dir,
             visualize_path=args.save_visualize_dir,
             env_num=args.train_env_num,
             env_rank=(int(train_img_num / args.train_env_num * i), train_img_num),
             render_mode=args.train_render_mode,
             image_iter=args.image_iter,
             start_update=args.start_update,
             update=args.update,
             ema_gamma=0.9,
             seed=args.seed + i  
             ) for i in range(args.train_env_num)])
# SubprocVectorEnv 同时启动多个子进程
# 在 gym.make() 中为每个环境设置独立种子
# train_envs = SubprocVectorEnv([lambda i=i: RecordVideo(
#     gym.make('CalliEnv-v0',
#              tool=tool,
#              folder_path=args.train_data_dir,
#              output_path=args.save_control_dir,
#              visualize_path=args.save_visualize_dir,
#              env_num=args.train_env_num,
#              env_rank=(int(train_img_num / args.train_env_num * i), train_img_num),
#              render_mode=args.train_render_mode,
#              image_iter=args.image_iter,
#              start_update=args.start_update,
#              update=args.update,
#              ema_gamma=0.9,
#              seed=args.seed + i  
#              ),
#     video_folder=save_video_dir,
#     name_prefix='trainvids_' + str(i)
# ) for i in range(args.train_env_num)])

print("make test envs...")
test_envs = DummyVectorEnv([lambda i=i: 
    gym.make('CalliEnv-v0',
             tool=tool,
             folder_path=args.test_data_dir,
             output_path=args.test_save_dir,
             visualize_path=None,
             env_num=args.test_env_num,
             env_rank=(int(test_img_num / args.test_env_num * i), test_img_num),
             render_mode=args.test_render_mode,
             image_iter=args.image_iter,
             start_update=args.start_update,
             update=args.update,
             ema_gamma=0.9,
             seed=args.seed + 1000 + i 
             ) for i in range(args.test_env_num)])
# test_envs = DummyVectorEnv([lambda i=i: RecordVideo(
#     gym.make('CalliEnv-v0',
#              tool=tool,
#              folder_path=args.test_data_dir,
#              output_path=args.test_save_dir,
#              visualize_path=None,
#              env_num=args.test_env_num,
#              env_rank=(int(test_img_num / args.test_env_num * i), test_img_num),
#              render_mode=args.test_render_mode,
#              image_iter=args.image_iter,
#              start_update=args.start_update,
#              update=args.update,
#              ema_gamma=0.9,
#              seed=args.seed + 1000 + i 
#              ),
#     video_folder=save_video_dir,
#     name_prefix='testvids_' + str(i)
# ) for i in range(args.test_env_num)])

# 设置全局随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

observe_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]

## if adopt fourier version
f_kwargs: Dict[str, Any] = {
                "sigma": args.sigma,
                "train_B": args.train_B,
                "fourier_dim": args.fourier_dim,
                "concatenate_fourier":args.concatenate_fourier
            }


actor_net = My_MLP(observe_shape, hidden_sizes=args.actor_network_shape,device=device,\
                   learn_fourier=args.learn_fourier, **f_kwargs).to(device)
critic_net = My_MLP(observe_shape,action_shape,hidden_sizes=args.critic_network_shape,concat=True,\
                    device=device, learn_fourier=args.learn_fourier, **f_kwargs).to(device)

actor = ActorProb(actor_net, action_shape, args.mu_std_net, device=device).to(device)
critic_1 = Critic(critic_net, device=device).to(device)
critic_2 = Critic(critic_net, device=device).to(device)

actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
critic1_optim = torch.optim.Adam(critic_1.parameters(), lr=args.critic_lr)
critic2_optim = torch.optim.Adam(critic_2.parameters(), lr=args.critic_lr)

## entropy
target_entropy = args.target_entropy_ratio * torch.log(torch.tensor(float(action_shape)))
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optim = torch.optim.Adam([log_alpha], lr=args.actor_lr)
alpha = (target_entropy, log_alpha, alpha_optim)

## SAC policy
policy = SACPolicy(actor,
                   actor_optim,
                   critic_1,
                   critic1_optim,
                   critic_2,
                   critic2_optim,
                   tau = args.tau,
                   gamma = args.gamma,
                   alpha=alpha
                )

## replay_buffer
buf = VectorReplayBuffer(args.buffer_size, len(train_envs))

## collector
# 移除exploration_noise参数（在0.5.1中可能不兼容）
train_collector = Collector(policy, train_envs, buf)  # , exploration_noise=True
train_collector.collect(n_step=((2**9)*args.train_env_num), random=True)
test_collector = Collector(policy, test_envs)  # , exploration_noise=False

## logger
log_path = os.path.join(args.logdir, now)
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)
print("start training...")

step_per_collect = (args.step_per_collect * args.train_env_num)
assert step_per_collect % args.train_env_num == 0

result = offpolicy_trainer(policy=policy,
                        train_collector=train_collector,
                        test_collector=test_collector,
                        max_epoch=args.max_epoch,
                        step_per_epoch = args.step_per_epoch,  
                        step_per_collect= step_per_collect,
                        episode_per_test= args.episode_per_test,
                        batch_size=args.batch_size,
                        update_per_step = args.update_per_step,
                        resume_from_log= args.resume_from_log,
                        logger=logger,
                        verbose= True,
                        show_progress= True,
                        test_in_train= True
                        )
print(f'Finished training! Use {result["duration"]}')
torch.save(policy.state_dict(), save_model_dir)
policy.eval()
collector = Collector(policy, test_envs)  # , exploration_noise=True
result = test_collector.collect(n_episode=1, render=1/30)
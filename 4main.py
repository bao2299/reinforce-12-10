import sys
import os
import time
from collections import deque
import numpy as np
import torch
import copy
import logging
import os

import numpy as np

from shutil import copyfile
from acktr import algo, utils
from acktr.utils import get_possible_position, get_rotation_mask
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from evaluation import evaluate
from tensorboardX import SummaryWriter
from unified_test import unified_test
from gym.envs.registration import register

def main(args):
    # input arguments about environment
    if args.test:
        test_model(args)
    else:
        train_model(args)

def test_model(args):
    assert args.test is True
    model_url = args.load_dir + args.load_name
    unified_test(model_url, args)

def train_model(args):


    # 初始化日志目录和文件路径
    log_dir = os.path.join(os.getcwd(), "consolelogs")  # 当前目录下的 logs 文件夹
    log_file = os.path.join(log_dir, "1consoleout.log")  # 日志文件路径

    # 自动创建日志目录（如果不存在）
    os.makedirs(log_dir, exist_ok=True)


        # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 写入日志文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )



    custom = "2experiment_2" 
    time_now = time.strftime('%Y.%m.%d-%H-%M', time.localtime(time.time()))
    env_name = args.env_name
    
    if args.device != 'cpu':
        torch.cuda.set_device(torch.device(args.device))

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_path = args.save_dir
    load_path = args.load_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    data_path = os.path.join(save_path, custom)
    try:
        os.makedirs(data_path)
    except OSError:
        pass

    log_dir = './log'
    log_dir = os.path.expanduser(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device(args.device)
    envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, log_dir, device, True, args=args)

    # 初始化 actor_critic
    if args.pretrain:
        model_pretrained, ob_rms = torch.load(os.path.join(load_path, args.load_name))
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args})
        load_dict = {k.replace('module.', ''): v for k, v in model_pretrained.items()}
        actor_critic.load_state_dict(load_dict)
        setattr(utils.get_vec_normalize(envs), 'ob_rms', ob_rms)
    else:
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args})
    print(actor_critic)
    print("Rotation:", args.enable_rotation)
    actor_critic.to(device)
    

    # 备份代码
    copyfile('main.py', os.path.join(data_path, 'main.py'))
    # 其他备份...

    # 初始化 agent
    if args.algorithm == 'a2c':
        agent = algo.ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            args.invalid_coef,
            args.lr,
            args.eps,
            args.alpha,
            max_grad_norm=0.5,
            reinforce=args.reinforce,
            args=args
        )
    elif args.algorithm == 'acktr':
        agent = algo.ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            args.invalid_coef,
            acktr=True,
            reinforce=args.reinforce,
            args=args
        )

    # 初始化 RolloutStorage
    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              can_give_up=False,
                              enable_rotation=args.enable_rotation,
                              pallet_size=args.container_size[0])

    obs = envs.reset()

    # print("\nAfter 初始化reset, observations:")
    # for env_id in range(args.num_processes):
    #     print(f"\nEnvironment初始化 {env_id} observation:")
    #     print("Shape:", obs[env_id].shape)
    #     print("Values:", obs[env_id])


    location_masks = []
    for observation in obs:
        if not args.enable_rotation:
            box_mask = get_possible_position(observation, args.container_size)
        else:
            box_mask = get_rotation_mask(observation, args.container_size)
        location_masks.append(box_mask)
    location_masks = torch.FloatTensor(location_masks).to(device)


    rollouts.obs[0].copy_(obs)
    rollouts.location_masks[0].copy_(location_masks)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_ratio = deque(maxlen=10)
    start = time.time()

    tbx_dir = './runs'
    if not os.path.exists('{}/{}/{}'.format(tbx_dir, env_name, custom)):
        os.makedirs('{}/{}/{}'.format(tbx_dir, env_name, custom))
    if args.tensorboard:
        writer = SummaryWriter(logdir='{}/{}/{}'.format(tbx_dir, env_name, custom))

    j = 0
    index = 0
    total_step =0 
    done_flags = [False] * args.num_processes 
    done_flags_mask = [False] * args.num_processes # 用于标记每个环境的done状态
    completed_obs_storage_masks = [False] * args.num_processes 
    completed_location_storage_mask = [False] * args.num_processes


    completed_obs_storage = torch.zeros(1, args.num_processes, *envs.observation_space.shape)
    completed_masks = torch.ones(1, args.num_processes, 1)
    completed_bad_masks = torch.ones(1, args.num_processes, 1)
    # 初始化一个计数器
    change_count = 0

        # 创建location_masks存储张量，形状要和rollouts中的location_masks匹配
    if args.enable_rotation:
        completed_location_storage = torch.zeros(1, args.num_processes, 2 * args.container_size[0]**2)
    # elif args.can_give_up:
    #     completed_location_storage = torch.zeros(1, args.num_processes, args.container_size**2 + 1)
    else:
        completed_location_storage = torch.zeros(1, args.num_processes, args.container_size[0]**2)


    


    while j<100000:
        j += 1
        for step in range(150):
            with torch.no_grad():

                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], location_masks)

            location_masks = []
            # print("Before step:", type(envs))  # 打印 envs 的类型
            # print("Envs class:", envs.__class__.__name__)
            obs, reward, done, infos = envs.step(action)

            # 更新done_flags
            # 遍历所有并行环境
            for env_id in range(len(done)):
                # 检查这个环境是否完成
                if done[env_id] == True  :
                    # 如果完成了，标记这个环境的done_flag为True
                    done_flags[env_id] = True

                   
                    

                # 新增一个循环来存储observation
            for env_id in range(len(done)):
                if done[env_id] == True  and  completed_obs_storage_masks[env_id]== False and  done_flags[env_id] == True:
                   

                                # 打印当前环境的观察值
                    print(f"\nEnvironment {env_id} observation at done=True:")
                    print("Shape:", obs[env_id].shape)  # 打印形状
                    print("Values:", obs[env_id])       # 打印值
                    # 存储对应环境的observation
                    completed_obs_storage[0, env_id].copy_(obs[env_id])
                    #             # 打印存储后的 completed_obs_storage 中的值，确认存储是否成功
                    # print(f"compute--Stored observation for environment {env_id}:")
                    # print("Shape:", completed_obs_storage[0, env_id].shape)  # 打印形状
                    # print("Values:", completed_obs_storage[0, env_id])       # 打印值

                    completed_obs_storage_masks[env_id]= True


            for i, info in enumerate(infos):
                if 'reset_observation' in info:
                    print(f"Worker {i} reset observation:", info['reset_observation'])
                                        
            for i in range(len(infos)):
                if 'episode' in infos[i].keys():
                    episode_rewards.append(infos[i]['episode']['r'])
                    episode_ratio.append(infos[i]['ratio'])
            

            for observation in obs:
                if not args.enable_rotation:
                    box_mask = get_possible_position(observation, args.container_size)
                else:
                    box_mask = get_rotation_mask(observation, args.container_size)
                location_masks.append(box_mask)
            location_masks = torch.FloatTensor(location_masks).to(device)


            
    


             # 如果环境还未结束，则插入rollout
            for i in range(args.num_processes):

                if  done_flags[i] == False:
                    
                        # 获取当前并行环境的数据
                    masks = torch.FloatTensor([[0.0] if done[i] else [1.0]]).squeeze()
                    bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in infos[i].keys() else [1.0]]).squeeze()

                    rollouts.insert(i, 
                                    obs[i], 
                                    recurrent_hidden_states[i], 
                                    action[i], 
                                    action_log_prob[i], 
                                    value[i], 
                                    reward[i], 
                                    masks, 
                                    bad_masks, 
                                    location_masks[i])
                    
                    
                # rollout 最后一次插入 当这个并行环境结束  
                if done_flags[i] == True and done_flags_mask[i] == False:
                       # 获取当前并行环境的数据
                    masks = torch.FloatTensor([[0.0] if done[i] else [1.0]]).squeeze()
                    bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in infos[i].keys() else [1.0]]).squeeze()

                    rollouts.insert(i, 
                                    obs[i], 
                                    recurrent_hidden_states[i], 
                                    action[i], 
                                    action_log_prob[i], 
                                    value[i], 
                                    reward[i], 
                                    masks, 
                                    bad_masks, 
                                    location_masks[i])
                    done_flags_mask[i] = True
                    completed_masks[0,i].copy_(masks)
                    completed_bad_masks[0,i].copy_(bad_masks)
                    change_count += 1  # 增加计数器
                    print(f"done_mask 被修改的次数: {change_count}")
                    completed_bad_masks[0,i].copy_(bad_masks)
                    completed_masks[0,1].copy_(masks)

                     # 新增一个循环来同时存储和location_masks
                for env_id in range(len(done)):
                    if done[env_id] == True and done_flags[env_id] == True and completed_location_storage_mask[env_id]== False:
                        
                    
                        # 存储对应环境的location_mask
                        completed_location_storage[0, env_id].copy_(location_masks[env_id])
                        completed_location_storage_mask[env_id] = True


                
             # 2. 检查是否所有并行环境都完成了采样和插入
            if all(done_flags):



                # 假设 done_flags 已经存在，且你想将所有元素修改为 False
                for i in range(len(done_flags)):
                    done_flags[i] = False

                # 将 done_mask 列表的所有元素修改为 False
                for i in range(len(done_flags_mask)):
                    done_flags_mask[i] = False

                # 将 completed_obs_storage_masks 列表的所有元素修改为 False
                for i in range(len(completed_obs_storage_masks)):
                    completed_obs_storage_masks[i] = False

                # exit()

                break

            #如果所有的done 都是true了 不在继续并行采样了

            total_step += 1 * args.num_processes


            
                
                
                
        # 对于rein force 算法不需要获取nextvalue 所以随便
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        
        print('第j次更细',j)
        j= j +1 
       
        
        print("rollouts.step:", rollouts.step)  # 直接打印列表
        # 退出程序
        

        #重新初始化 rollouts
        rollouts = RolloutStorage(args.num_steps,
                                  args.num_processes,
                                  envs.observation_space.shape,
                                  envs.action_space,
                                  actor_critic.recurrent_hidden_state_size,
                                  can_give_up=False,
                                  enable_rotation=args.enable_rotation,
                                  pallet_size=args.container_size[0])
       
       #把completed_obs_storage 重新赋值给新的初始化后的roll out 

        rollouts.obs[0].copy_(completed_obs_storage[0])
        # rollouts.masks[0].copy_(completed_masks[0])
        # rollouts.bad_masks[0].copy(completed_bad_masks[0])
        rollouts.location_masks[0].copy_(completed_location_storage[0])
        

        # rollouts.obs[0].copy_(obs)
        # rollouts.location_masks[0].copy_(location_masks)
        rollouts.to(device)

        if args.save_model:
            if (j % args.save_interval == 0) and args.save_dir != "":
                torch.save([
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(data_path, env_name + time_now + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            #total_num_steps = (j + 1) * args.num_processes * args.num_steps
            total_num_steps = total_step
            end = time.time()
            index += 1


            log_info_1 = (
                "The algorithm is {}, the recurrent policy is {}\nThe env is {}, the version is {}".format(
                    args.algorithm, False, env_name, custom
                )
            )
            log_info_2 = (
                "Updates {}, num timesteps {}, FPS {}\n"
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                "The dist entropy {:.5f}, The value loss {:.5f}, the action loss {:.5f}\n"
                "The mean space ratio is {}".format(
                    j, total_num_steps, int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss,
                    action_loss, np.mean(episode_ratio)
                )
            )

            # 打印到控制台和日志文件
            logging.info(log_info_1)
            logging.info(log_info_2)





            if args.tensorboard:
                writer.add_scalar('The average rewards', np.mean(episode_rewards), j)
                writer.add_scalar("The mean ratio", np.mean(episode_ratio), j)
                writer.add_scalar('Distribution entropy', dist_entropy, j)
                writer.add_scalar("The value loss", value_loss, j)
                writer.add_scalar("The action loss", action_loss, j)
                writer.add_scalar('Probability loss', prob_loss, j)
                writer.add_scalar("Mask loss", graph_loss, j)

def registration_envs():
    register(
        id='Bpp-v0',  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',  # Expalined in envs/__init__.py
    )



if __name__ == "__main__":
    registration_envs()
    args = get_args()
    main(args)
import sys
import os
import time
from collections import deque
import numpy as np
import torch
import copy

from shutil import copyfile
from acktr import algo, utils
from acktr.utils import get_possible_position, get_rotation_mask
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from acktr.DYRolloutStorage import DynamicRolloutStorage
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
    rollouts = DynamicRolloutStorage(
                              args.num_processes,
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              can_give_up=False,
                              enable_rotation=args.enable_rotation,
                              pallet_size=args.container_size[0])

    obs = envs.reset()
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
    while True:
        j += 1
        for step in range(150):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], location_masks)

            location_masks = []
            obs, reward, done, infos = envs.step(action)
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

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks,
                            location_masks)
            total_step = total_step + 1
            
            if done == True:
                break
                

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        print("第j次更新",j)

        # 重新初始化 rollouts
        rollouts = DynamicRolloutStorage(
                                  args.num_processes,
                                  envs.observation_space.shape,
                                  envs.action_space,
                                  actor_critic.recurrent_hidden_state_size,
                                  can_give_up=False,
                                  enable_rotation=args.enable_rotation,
                                  pallet_size=args.container_size[0])
        obs = envs.reset()
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
            print(
                "The algorithm is {}, the recurrent policy is {}\nThe env is {}, the version is {}".format(
                    args.algorithm, False, env_name, custom))
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                "The dist entropy {:.5f}, The value loss {:.5f}, the action loss {:.5f}\n"
                "The mean space ratio is {}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss, np.mean(episode_ratio)))

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
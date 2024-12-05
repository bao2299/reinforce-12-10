import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

# the shape of observation: batch * cpu * length
class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size,can_give_up, enable_rotation, pallet_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        # self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps+1 , num_processes, 1)


        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
         # 打印初始状态，看看这些张量在创建时的内容是什么
        #print("Initial obs:", self.obs)
        #print("Initial rewards:", self.rewards)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        if enable_rotation:
            self.location_masks = torch.zeros(num_steps+1, num_processes, 2 * pallet_size**2)
        elif can_give_up:
            self.location_masks = torch.zeros(num_steps+1, num_processes, pallet_size**2 +1)
        else:
            self.location_masks = torch.zeros(num_steps+1, num_processes, pallet_size**2)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = [0] * num_processes
        self.lossmask = torch.zeros(num_steps, num_processes, 1)
        self.lossmask[0, :, :] = 1

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.location_masks = self.location_masks.to(device)
        self.lossmask = self.lossmask.to(device)

    # def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
    #            value_preds, rewards, masks, bad_masks, location_masks):
    #     self.obs[self.step + 1].copy_(obs)
    #     self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
    #     self.actions[self.step].copy_(actions)
    #     self.action_log_probs[self.step].copy_(action_log_probs)
    #     self.value_preds[self.step].copy_(value_preds)
    #     self.rewards[self.step].copy_(rewards)
    #     self.masks[self.step + 1].copy_(masks)
    #     self.bad_masks[self.step + 1].copy_(bad_masks)
    #     self.location_masks[self.step + 1].copy_(location_masks)
    #     #self.step = (self.step + 1) % self.num_steps
    #     self.step = (self.step + 1)



    def insert(self, env_idx, obs, recurrent_hidden_states, actions, action_log_probs,
           value_preds, rewards, masks, bad_masks, location_masks,lossmask):
    # 对应于指定的并行环境
        current_step = self.step[env_idx]
        self.obs[current_step + 1, env_idx].copy_(obs)
        self.recurrent_hidden_states[current_step + 1, env_idx].copy_(recurrent_hidden_states)
        self.actions[current_step, env_idx].copy_(actions)
        self.action_log_probs[current_step, env_idx].copy_(action_log_probs)
        self.value_preds[current_step, env_idx].copy_(value_preds)
        self.rewards[current_step, env_idx].copy_(rewards)
        self.masks[current_step + 1, env_idx].copy_(masks)
        self.bad_masks[current_step + 1, env_idx].copy_(bad_masks)
        self.location_masks[current_step + 1, env_idx].copy_(location_masks)
        # 插入 lossmask 的值
        self.lossmask[current_step +1 , env_idx].copy_(lossmask)
        # 更新该环境的步数
        self.step[env_idx] += 1



    def reset_trajectory(self, i):
        """
        清空指定轨迹的数据，将其重置为初始值。
        
        参数:
            i (int): 并行环境的索引。
        """
        # 重置轨迹相关的数据
        self.obs[:, i].fill_(0)  # 初始值为0
        self.recurrent_hidden_states[:, i].fill_(0)  # 初始值为0
        self.rewards[:, i].fill_(0)  # 初始值为0
        self.value_preds[:, i].fill_(0)  # 初始值为0
        self.returns[:, i].fill_(0)  # 初始值为0
        self.action_log_probs[:, i].fill_(0)  # 初始值为0
        self.actions[:, i].fill_(0)  # 初始值为0
        self.masks[:, i].fill_(1)  # 初始值为1
        self.bad_masks[:, i].fill_(1)  # 初始值为1

        # 根据初始化时的情况重置 location_masks
        if hasattr(self, 'enable_rotation') and self.enable_rotation:
            self.location_masks[:, i].fill_(0)  # 初始值为0
        elif hasattr(self, 'can_give_up') and self.can_give_up:
            self.location_masks[:, i].fill_(0)  # 初始值为0
        else:
            self.location_masks[:, i].fill_(0)  # 初始值为0

        self.lossmask[:, i].fill_(0)  # 初始值为0
        self.lossmask[0, i].fill_(1)  # 在初始步设置 loss 掩码为1，符合初始化时的状态

        # 重置当前轨迹的步数
        self.step[i] = 0


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.location_masks[0].copy_(self.location_masks[-1])
    
    def after_update1(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        
        # Reset masks[0] and bad_masks[0] to their initialized values (1)
        self.masks[0].fill_(1.0)
        self.bad_masks[0].fill_(1.0)
        
        self.location_masks[0].copy_(self.location_masks[-1])
        self.step = 0


    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        reinforce=True): # 这里一定要改成true 如果想用reinforce算法的话
        if reinforce:
            '''
            # 蒙特卡洛方法：累计轨迹总回报
            print('查看每条轨迹走了多少步')
            print(self.step)
            self.returns[self.step - 1] = self.rewards[self.step - 1]

            #for step in reversed(range(self.step - 1)):
        # 如果 bad_masks 为 0.0，说明该状态不应对未来回报有影响
                #self.returns[step] = (self.rewards[step] + gamma * self.returns[step + 1]) * self.bad_masks[step]

            # 从倒数第二个时间步开始反向遍历，使用 range(self.step - 2, -1, -1)
            for step in range(self.step - 2, -1, -1):
        # 累计当前时间步的折扣回报
                self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1]
                 # 对累积回报进行标准化处理
            valid_returns = self.returns[:self.step]  # 只对有效的部分进行标准化
            returns_mean = valid_returns.mean()
            returns_std = valid_returns.std() + 1e-5  # 防止标准差为0导致的除0错误
            self.returns[:self.step] = (valid_returns - returns_mean) / returns_std
            '''

                # 遍历每个并行环境，单独计算每个环境的 returns
            for i in range(len(self.step)):
                current_step = self.step[i]  # 获取当前并行环境的步数
                if current_step > 0:  # 确保步数有效
                    # 将最后一个时间步的 returns 赋值为当前时间步的 reward
                    self.returns[current_step - 1, i] = self.rewards[current_step - 1, i]
                    
                    # 从倒数第二个时间步开始反向遍历，计算累计回报
                    for step in range(current_step - 2, -1, -1):
                        self.returns[step, i] = self.rewards[step, i] + gamma * self.returns[step + 1, i]

            # 对累积回报进行标准化处理
            for i in range(len(self.step)):
                
                valid_returns = self.returns[:self.step[i], i]  # 只对有效的部分进行标准化
                returns_mean = valid_returns.mean()
                returns_std = valid_returns.std() + 1e-5  # 防止标准差为0导致的除0错误
                self.returns[:self.step[i], i] = (valid_returns - returns_mean) / returns_std
        else:
            # 原有的 TD 或 GAE 方法
            if use_proper_time_limits:
                if use_gae:
                    self.value_preds[-1] = next_value
                    gae = 0
                    for step in reversed(range(self.rewards.size(0))):
                        delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
                else:
                    self.returns[-1] = next_value
                    for step in reversed(range(self.rewards.size(0))):
                        self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
            else:
                if use_gae:
                    self.value_preds[-1] = next_value
                    gae = 0
                    for step in reversed(range(self.rewards.size(0))):
                        delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
                else:
                    self.returns[-1] = next_value
                    for step in reversed(range(self.rewards.size(0))):
                        self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

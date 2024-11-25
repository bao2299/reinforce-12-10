import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class DynamicRolloutStorage:
    def __init__(self, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, can_give_up, enable_rotation, pallet_size):
        # 初始化一个最小容量（初始容量为 1）
        initial_capacity = 1
        self.obs = torch.zeros(initial_capacity + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            initial_capacity + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(initial_capacity, num_processes, 1)
        self.value_preds = torch.zeros(initial_capacity + 1, num_processes, 1)
        self.returns = torch.zeros(initial_capacity + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(initial_capacity, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(initial_capacity, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.masks = torch.ones(initial_capacity + 1, num_processes, 1)

        if enable_rotation:
            self.location_masks = torch.zeros(initial_capacity + 1, num_processes, 2 * pallet_size ** 2)
        elif can_give_up:
            self.location_masks = torch.zeros(initial_capacity + 1, num_processes, pallet_size ** 2 + 1)
        else:
            self.location_masks = torch.zeros(initial_capacity + 1, num_processes, pallet_size ** 2)

        self.bad_masks = torch.ones(initial_capacity + 1, num_processes, 1)

        self.capacity = initial_capacity  # 跟踪当前存储的容量
        self.step = 0  # 跟踪当前插入的位置

    def _expand_storage(self):
        # 扩展到能够容纳新插入步的容量
        new_capacity = self.capacity + 1

        # 扩展每个存储张量
        self.obs = self._expand_tensor(self.obs, new_capacity + 1)
        self.recurrent_hidden_states = self._expand_tensor(self.recurrent_hidden_states, new_capacity + 1)
        self.rewards = self._expand_tensor(self.rewards, new_capacity)
        self.value_preds = self._expand_tensor(self.value_preds, new_capacity + 1)
        self.returns = self._expand_tensor(self.returns, new_capacity + 1)
        self.action_log_probs = self._expand_tensor(self.action_log_probs, new_capacity)
        self.actions = self._expand_tensor(self.actions, new_capacity)
        self.masks = self._expand_tensor(self.masks, new_capacity + 1)
        self.location_masks = self._expand_tensor(self.location_masks, new_capacity + 1)
        self.bad_masks = self._expand_tensor(self.bad_masks, new_capacity + 1)

        self.capacity = new_capacity

    def _expand_tensor(self, tensor, new_size):
        old_size = tensor.size(0)
        new_tensor = torch.zeros(new_size, *tensor.size()[1:], device=tensor.device)
        new_tensor[:old_size] = tensor
        return new_tensor

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

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, location_masks):
        # 检查是否需要扩展容量
        if self.step + 1 >= self.capacity:
            self._expand_storage()

        # 插入新数据
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.location_masks[self.step + 1].copy_(location_masks)

        # 更新插入位置
        self.step += 1

    def after_update(self):
        # 将上一次存储的最后一步复制到起始位置
        self.obs[0].copy_(self.obs[self.step])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[self.step])
        self.masks[0].copy_(self.masks[self.step])
        self.bad_masks[0].copy_(self.bad_masks[self.step])
        self.location_masks[0].copy_(self.location_masks[self.step])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        reinforce=True): # 这里一定要改成true 如果想用reinforce算法的话
        if reinforce:
            # 蒙特卡洛方法：累计轨迹总回报
            print('!!!查看每条轨迹走了多少步!!!')
            print(self.step)
            self.returns[self.step - 1] = self.rewards[self.step - 1]

            #for step in reversed(range(self.step - 1)):
        # 如果 bad_masks 为 0.0，说明该状态不应对未来回报有影响
                #self.returns[step] = (self.rewards[step] + gamma * self.returns[step + 1]) * self.bad_masks[step]

            for step in reversed(range(self.step - 1)):

                self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1]

                 # 对累积回报进行标准化处理
            valid_returns = self.returns[:self.step]  # 只对有效的部分进行标准化
            returns_mean = valid_returns.mean()
            returns_std = valid_returns.std() + 1e-5  # 防止标准差为0导致的除0错误
            self.returns[:self.step] = (valid_returns - returns_mean) / returns_std


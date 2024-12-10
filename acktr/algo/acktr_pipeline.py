import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from acktr.algo.kfac import KFACOptimizer

class ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 invaild_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 reinforce=False,
                 args=None):

        self.actor_critic = actor_critic
        self.acktr = acktr
        self.reinforce = reinforce
        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm
        self.loss_func_value = nn.MSELoss(reduction='none')
        self.loss_func_graph = nn.MSELoss(reduction='none')


        # self.loss_func = nn.MSELoss(reduce=False, size_average=True)
        self.entropy_coef = entropy_coef
        self.args = args

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
                    # 获取仅包含 actor 参数的列表（排除 critic 参数）
            actor_params = [p for n, p in actor_critic.named_parameters() if 'critic' not in n and 'critic_linear' not in n]
            # 初始化优化器，只优化 actor 参数
            self.optimizer = optim.RMSprop(actor_params, lr, eps=eps, alpha=alpha)
            # self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
            
        if reinforce:
            # 为 Critic 参数单独设置优化器
            critic_params = [p for n, p in actor_critic.named_parameters() if 'critic' in n or 'critic_linear' in n]
            self.critic_optimizer = optim.RMSprop(critic_params, lr=1e-4 ,eps=eps,alpha=alpha)
        

        actor_params_ids = [id(p) for p in self.optimizer.param_groups[0]['params']]
        critic_params_ids = [id(p) for p in self.critic_optimizer.param_groups[0]['params']]
        shared_params = set(actor_params_ids) & set(critic_params_ids)
        print(f"共享的参数数量: {len(shared_params)}")  # 如果大于 0，则存在共享问题
       
        # # 打印 Actor 优化器包含的参数名称和形状
        # print("Actor Optimizer Parameters:")
        # for name, param in actor_critic.named_parameters():
        #     if id(param) in actor_params_ids:  # 使用 id(param) 来进行比较
        #         print(f"Parameter Name: {name}, Shape: {param.shape}")

        # # 打印 Critic 优化器包含的参数名称和形状
        # print("\nCritic Optimizer Parameters:")
        # for name, param in actor_critic.named_parameters():
        #     if id(param) in critic_params_ids:  # 使用 id(param) 来进行比较
        #         print(f"Parameter Name: {name}, Shape: {param.shape}")

        # exit()


    def update(self, rollouts, j, wandb):
        # 如果总步数太少则跳过更新
        # if sum(rollouts.step) <= 1:
        #     print(f"!!!Skipping update: 当前_num_steps = {rollouts.step}")
        #     return 0, 0, 0, 0, 0

        T = 80
        # 获取维度信息
        # rollouts.obs: [T+1, num_processes, *obs_shape]
        # rollouts.actions:-T, num_processes, action_dim]
        # rollouts.rewards: [50, num_processes, 1]
        # rollouts.location_masks: [51, num_processes, mask_size]
        obs_shape = rollouts.obs[:T].size()[2:]  
        action_shape = rollouts.actions[:T].size()[-1]
        num_steps, num_processes, _ = rollouts.rewards[:T].size()
        mask_size = rollouts.location_masks[:T].size()[-1]
      

      # 打印完整的形状
        # print("Complete shapes:")
        # print(f"rollouts.obs shape: {rollouts.obs.size()}")
        # print(f"rollouts.actions shape: {rollouts.actions.size()}")
        # print(f"rollouts.rewards shape: {rollouts.rewards.size()}")
        # print(f"rollouts.location_masks shape: {rollouts.location_masks.size()}")

        # 打印提取的形状
        print("\nExtracted shapes:")
        print(f"obs_shape: {obs_shape}")
        print(f"action_shape: {action_shape}")
        print(f"num_steps: {num_steps}, num_processes: {num_processes}")
        print(f"mask_size: {mask_size}")
        realnumprocess = num_processes
        for i, step in enumerate(rollouts.step):
            if step == 1:
                print(f"Found step == 1 for environment {i}")
                # return 0,0,0,0,0
            
        
        # Evaluate actions
        # 输入形状:
        # obs: [T * num_processes, *obs_shape]
        # recurrent_hidden_states: [num_processes, hidden_state_size]
        # masks: [T * num_processes, 1]
        # actions: [T * num_processes, action_shape]
        # location_masks: [T * num_processes, mask_size]
        obs_input = rollouts.obs[:T].view(-1, *obs_shape)
        masks_input = rollouts.masks[:T].view(-1, 1)
        actions_input = rollouts.actions[:T].view(-1, action_shape)
           
        location_masks_input = rollouts.location_masks[:T].view(-1, mask_size)

        lossmask_input = rollouts.lossmask[:T].view(-1, 1) # [T * N, 1]
        # 将各输入与lossmask相乘（除 recurrent_hidden_states[0] 外）
        obs_input = obs_input * lossmask_input     # obs与lossmask相乘
        # masks_input = masks_input * lossmask_input   # masks是 [T*N,1]，与lossmask_input [T*N,1] 直接相乘
        actions_input = actions_input * lossmask_input   # actions [T*N, action_dim] 与 [T*N,1] 会广播
        location_masks_input = location_masks_input * lossmask_input  # [T*N, mask_size]与[T*N,1] 也可直接相乘

        
            
        values, action_log_probs, dist_entropy, _, bad_prob, pred_mask = self.actor_critic.evaluate_actions(
            obs_input,
            rollouts.recurrent_hidden_states[0].view(-1, 256),# 如果是rnn 记得改成256
            masks_input,
            actions_input,
            location_masks_input,
            lossmask_input
        )
        
        # 打印 的形状
        # print("values:\n", values)
        # print("values shape:", values.shape)
        # print("action_log_probs:\n", action_log_probs)
        # print("action_log_probs.shale",action_log_probs.shape)
        # print("dist_entropy:\n", dist_entropy)
        # print("dist_entropy shape:", dist_entropy.shape)
        # print("bad_prob:\n", bad_prob)
        # print("bad_prob shape:", bad_prob.shape)
        # print("pred_mask:\n", pred_mask)
        # print("predmask shape:", pred_mask.shape)
        # print('rolloutlossmsk：',rollouts.lossmask)
        # print('rollouts.returns',rollouts.returns)
        # print('rollouts.reward',rollouts.rewards)
        

        


       
        # 输出重塑
        # values: [T, num_processes, 1]
        # action_log_probs: [T, num_processes, 1]
        values = values.view(T, num_processes, 1)
        action_log_probs = action_log_probs.view(T, num_processes, 1)

        # 计算损失
        if not self.reinforce:
            # advantages: [T, num_processes, 1]
            advantages = rollouts.returns[:T] - values
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
        else:
            # advantages: [T, num_processes, 1]
            print("reninforce = true")
            advantages = rollouts.returns[:T]
            action_loss = -(advantages.detach()* action_log_probs)
            action_loss = action_loss * rollouts.lossmask
            action_loss = action_loss.sum()/realnumprocess

            critic_loss = self.loss_func_value(
                values.view(-1, 1),  # [T * num_processes, 1]
                rollouts.returns[:T].view(-1, 1)  # [T * num_processes, 1]
            )

            critic_loss = critic_loss.view(T, num_processes, 1)
            # 算出来的损失去成lossmask
            critic_loss = critic_loss * rollouts.lossmask 
            # 总损失去除以轨迹数
            critic_loss = critic_loss.sum()/realnumprocess
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
           
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()  # 清除 critic 梯度
            print('critirc loss 可以更新没有问题')
            
            
            value_loss = critic_loss

        # Mask相关损失计算
        mask_len = self.args.container_size[0] * self.args.container_size[1]
        if self.args.enable_rotation:
            mask_len *= 2

        # pred_mask: [T, num_processes, mask_len]
        # mask_truth: [T, num_processes, mask_len]
        pred_mask = pred_mask.view(T, num_processes, mask_len)
        mask_truth = rollouts.location_masks[0:T]
        graph_loss = self.loss_func_graph(pred_mask, mask_truth)
        graph_loss = graph_loss * rollouts.lossmask
        graph_loss = graph_loss.sum()/realnumprocess

        dist_entropy = dist_entropy.view(T, num_processes, 1)
        dist_entropy = dist_entropy * rollouts.lossmask
        dist_entropy = dist_entropy.sum()/realnumprocess

        prob_loss = bad_prob
        prob_loss = prob_loss.view(T, num_processes, 100)
        prob_loss = prob_loss * rollouts.lossmask
        prob_loss = prob_loss.sum()/realnumprocess
        
        print("所有损失打码码成功")
        

        

        # 优化步骤
        self.optimizer.zero_grad()
        loss = action_loss
        
        # if not self.reinforce:
        #     loss += value_loss * self.value_loss_coef
        
        loss += prob_loss * self.invaild_coef
        loss -= dist_entropy * self.entropy_coef
        loss += graph_loss * 4.0

        loss.backward()

        if not self.acktr:
            # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.max_grad_norm)

        self.optimizer.step()
        print("参数更新成功")
        
      
        # 每10次迭代记录到wandb
        if j % 10 == 0:
            wandb.log({
                "value_loss": value_loss.item(),
                "action_loss": action_loss.item(), 
                "dist_entropy": dist_entropy.item(),
                "prob_loss": prob_loss.item(),
                "graph_loss": graph_loss.item()
            })

        return (
            value_loss.item(),
            action_loss.item(),
            dist_entropy.item(),
            prob_loss.item(),
            graph_loss.item()
        )

    def compute_returns(self, rollouts, next_value, gamma, gae_lambda):
        if self.reinforce:
            rollouts.compute_returns(next_value, False, gamma, gae_lambda, use_proper_time_limits=True)
        else:
            rollouts.compute_returns(next_value, True, gamma, gae_lambda, use_proper_time_limits=True)

    def check_nan(self, model, index):
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f'index {index} encountered NaN in gradients!')
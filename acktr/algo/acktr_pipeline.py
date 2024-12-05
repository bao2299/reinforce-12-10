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
       
        # 打印 Actor 优化器包含的参数名称和形状
        print("Actor Optimizer Parameters:")
        for name, param in actor_critic.named_parameters():
            if id(param) in actor_params_ids:  # 使用 id(param) 来进行比较
                print(f"Parameter Name: {name}, Shape: {param.shape}")

        # 打印 Critic 优化器包含的参数名称和形状
        print("\nCritic Optimizer Parameters:")
        for name, param in actor_critic.named_parameters():
            if id(param) in critic_params_ids:  # 使用 id(param) 来进行比较
                print(f"Parameter Name: {name}, Shape: {param.shape}")

        exit()


    def update(self, rollouts, j, wandb):
        # 如果总步数太少则跳过更新
        # if sum(rollouts.step) <= 1:
        #     print(f"!!!Skipping update: 当前_num_steps = {rollouts.step}")
        #     return 0, 0, 0, 0, 0

        
        # 获取维度信息
        # rollouts.obs: [51, num_processes, *obs_shape]
        # rollouts.actions: [50, num_processes, action_dim]
        # rollouts.rewards: [50, num_processes, 1]
        # rollouts.location_masks: [51, num_processes, mask_size]
        obs_shape = rollouts.obs[:51].size()[2:]  
        action_shape = rollouts.actions[:50].size()[-1]
        num_steps, num_processes, _ = rollouts.rewards[:50].size()
        mask_size = rollouts.location_masks[:51].size()[-1]

        realnumprocess = num_processes
        for i, step in enumerate(rollouts.step):
            if step == 1:
                print(f"Found step == 1 for environment {i}")
                realnumprocess = num_processes-1
                rollouts.lossmask[:, i].fill_(0)
            

        # Evaluate actions
        # 输入形状:
        # obs: [50 * num_processes, *obs_shape]
        # recurrent_hidden_states: [num_processes, hidden_state_size]
        # masks: [50 * num_processes, 1]
        # actions: [50 * num_processes, action_shape]
        # location_masks: [50 * num_processes, mask_size]
        values, action_log_probs, dist_entropy, _, bad_prob, pred_mask = self.actor_critic.evaluate_actions(
            rollouts.obs[:50].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:50].view(-1, 1),
            rollouts.actions[:50].view(-1, action_shape),
            rollouts.location_masks[:50].view(-1, mask_size)
        )

        # 打印 的形状
        print("action_log_probs.shale",action_log_probs.shape)
        print("dist_entropy shape:", dist_entropy.shape)
        print("bad_prob shape:", bad_prob.shape)
        print("predmask shape:", pred_mask.shape)
        


       
        # 输出重塑
        # values: [50, num_processes, 1]
        # action_log_probs: [50, num_processes, 1]
        values = values.view(50, num_processes, 1)
        action_log_probs = action_log_probs.view(50, num_processes, 1)

        # 计算损失
        if not self.reinforce:
            # advantages: [50, num_processes, 1]
            advantages = rollouts.returns[:50] - values
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
        else:
            # advantages: [50, num_processes, 1]
            print("reninforce = true")
            advantages = rollouts.returns[:50]
            action_loss = -(advantages.detach()* action_log_probs)
            action_loss = action_loss * rollouts.lossmask
            action_loss = action_loss.sum()/realnumprocess

            critic_loss = self.loss_func_value(
                values.view(-1, 1),  # [50 * num_processes, 1]
                rollouts.returns[:50].view(-1, 1)  # [50 * num_processes, 1]
            )

            critic_loss = critic_loss.view(50, num_processes, 1)
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

        # pred_mask: [50, num_processes, mask_len]
        # mask_truth: [50, num_processes, mask_len]
        pred_mask = pred_mask.view(50, num_processes, mask_len)
        mask_truth = rollouts.location_masks[0:50]
        graph_loss = self.loss_func_graph(pred_mask, mask_truth)
        graph_loss = graph_loss * rollouts.lossmask
        graph_loss = graph_loss.sum()/realnumprocess

        dist_entropy = dist_entropy.view(50, num_processes, 1)
        dist_entropy = dist_entropy * rollouts.lossmask
        dist_entropy = dist_entropy.sum()/realnumprocess

        prob_loss = bad_prob
        prob_loss = prob_loss.view(50, num_processes, 100)
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
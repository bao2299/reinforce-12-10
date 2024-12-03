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
        self.loss_func = nn.MSELoss(reduction='mean')
        # self.loss_func = nn.MSELoss(reduce=False, size_average=True)
        self.entropy_coef = entropy_coef
        self.args = args

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

        if reinforce:
            # 为 Critic 参数单独设置优化器
            critic_params = [p for n, p in actor_critic.named_parameters() if 'critic' in n]
            self.critic_optimizer = optim.RMSprop(critic_params, lr=1e-3)

    def update(self, rollouts,j , wandb):
        

        # 获取维度信息
        num_steps, num_processes, _ = rollouts.rewards.size()
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        mask_size = rollouts.location_masks.size()[-1]
        
        # 初始化累计损失
        value_losses = []
        action_losses = []
        dist_entropies = []
        prob_losses = []
        graph_losses = []

        # 遍历每个并行环境
        for env_idx in range(num_processes):
            # 获取当前环境的有效步数
            environment_step = rollouts.step[env_idx]
            
            if environment_step <= 1:
                print(f"!!!Skipping update for environment {env_idx}: 当前_num_steps = {environment_step}")
                continue

            # 获取当前环境的数据，保持三维结构 [num_steps, 1, feature_dim]
            obs = rollouts.obs[:environment_step, env_idx:env_idx+1]  
            recurrent_hidden_states = rollouts.recurrent_hidden_states[0, env_idx:env_idx+1]
            masks = rollouts.masks[:environment_step, env_idx:env_idx+1]
            actions = rollouts.actions[:environment_step, env_idx:env_idx+1]
            location_masks = rollouts.location_masks[:environment_step, env_idx:env_idx+1]
            
            # 重塑数据以保持与原始代码相同的维度结构
            values, action_log_probs, dist_entropy, _, bad_prob, pred_mask = self.actor_critic.evaluate_actions(
                obs.view(-1, *obs_shape),  # [steps * 1, *obs_shape]
                recurrent_hidden_states.view(-1, self.actor_critic.recurrent_hidden_state_size),  # [1, hidden_size]
                masks.view(-1, 1),  # [steps * 1, 1]
                actions.view(-1, action_shape),  # [steps * 1, action_dim]
                location_masks.view(-1, mask_size)  # [steps * 1, mask_size]
            )

            # 重塑输出维度为 [num_steps, 1, 1]
            values = values.view(environment_step, 1, 1)
            action_log_probs = action_log_probs.view(environment_step, 1, 1)

            # 计算优势和损失
            if not self.reinforce:
                advantages = rollouts.returns[:environment_step, env_idx:env_idx+1] - values
                value_loss = advantages.pow(2).mean()
                action_loss = -(advantages.detach() * action_log_probs).mean()
            else:

                advantages = rollouts.returns[:environment_step, env_idx:env_idx+1]
                action_loss = -(advantages * action_log_probs).mean()

                critic_loss = self.loss_func(
                    values,
                    rollouts.returns[:environment_step, env_idx:env_idx+1]
                )
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()
                
                value_loss = critic_loss

            # 处理mask相关的损失
            mask_len = self.args.container_size[0] * self.args.container_size[1]
            if self.args.enable_rotation:
                mask_len *= 2
            
            pred_mask = pred_mask.view(environment_step, 1, mask_len)
            mask_truth = location_masks.view(environment_step, 1, mask_len)
            
            graph_loss = self.loss_func(pred_mask, mask_truth)
            dist_entropy = dist_entropy.mean()
            prob_loss = bad_prob.mean()

            # ACKTR特定的Fisher信息矩阵计算
            if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
                self.actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = torch.randn(values.size())
                if values.is_cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
                fisher_loss = pg_fisher_loss + vf_fisher_loss + graph_loss * 1e-8
                
                self.optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.optimizer.acc_stats = False

            # 优化步骤
            self.optimizer.zero_grad()
            loss = action_loss
            
            if not self.reinforce:
                loss += value_loss * self.value_loss_coef

            loss += prob_loss * self.invaild_coef
            loss -= dist_entropy * self.entropy_coef
            loss += graph_loss * 4.0

            loss.backward()

            if not self.acktr:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

            self.optimizer.step()

            if j % 10 == 0 and env_idx == 0:  # 每隔10个环境记录一次

                # 在每次环境更新后记录到 wandb
                wandb.log({
                    "env_index": env_idx,
                    "value_loss": value_loss.item(),
                    "action_loss": action_loss.item(),
                    "dist_entropy": dist_entropy.item(),
                    "prob_loss": prob_loss.item(),
                    "graph_loss": graph_loss.item()
                })
                

            # 记录损失
            value_losses.append(value_loss.item())
            action_losses.append(action_loss.item())
            dist_entropies.append(dist_entropy.item())
            prob_losses.append(prob_loss.item())
            graph_losses.append(graph_loss.item())

        # 返回所有环境的平均损失
        return (
            sum(value_losses) / len(value_losses) if value_losses else 0,
            sum(action_losses) / len(action_losses) if action_losses else 0,
            sum(dist_entropies) / len(dist_entropies) if dist_entropies else 0,
            sum(prob_losses) / len(prob_losses) if prob_losses else 0,
            sum(graph_losses) / len(graph_losses) if graph_losses else 0,
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
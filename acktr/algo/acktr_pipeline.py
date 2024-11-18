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
                 acktr=False,   # 为 true 时启用 ACKTR 自然梯度优化器。使用 KFACOptimizer 来优化模型。适用于 Actor-Critic 模式。
                                # 为 false 时使用常规优化器（如 RMSprop）。仍然支持 Actor-Critic 和 Reinforce with Baseline 模式，根据 reinforce 的值决定。

                 reinforce=False,  # 为 true 时开启 Reinforce with Baseline，false 时使用 Actor-Critic 网络

                 args=None):

        self.actor_critic = actor_critic
        self.acktr = acktr
        self.reinforce = reinforce


        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm

        self.loss_func = nn.MSELoss(reduction='mean')
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


    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        mask_size = rollouts.location_masks.size()[-1]

        # Evaluate actions
        values, action_log_probs, dist_entropy, _, bad_prob, pred_mask = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            rollouts.location_masks[:-1].view(-1, mask_size)
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        if not self.reinforce:
            # Actor-Critic 损失计算
            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
        else:
            # Reinforce 损失计算：使用回报的平均值作为baseline 这个平均回报已经在在storag的compute_return 中处理了
            advantages = rollouts.returns[:-1]
            '''
            # Reinforce 损失计算：直接使用 critic 的值作为 baseline
            advantages = rollouts.returns[:-1].view(-1) - values.detach()
            '''
            action_loss = -(advantages * action_log_probs).mean()

            # Critic 网络的独立优化
            critic_loss = self.loss_func(values.view(-1, 1), rollouts.returns[:-1].view(-1, 1))

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # 在 reinforce 模式下，直接将 value_loss 赋值为 critic_loss
            value_loss = critic_loss

        # Mask-related losses
        mask_len = self.args.container_size[0] * self.args.container_size[1]
        mask_len = mask_len * (1 + self.args.enable_rotation)
        pred_mask = pred_mask.reshape((num_steps, num_processes, mask_len))
        mask_truth = rollouts.location_masks[0:num_steps]
        graph_loss = self.loss_func(pred_mask, mask_truth)
        dist_entropy = dist_entropy.mean()
        prob_loss = bad_prob.mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()  # detach

            fisher_loss = pg_fisher_loss + vf_fisher_loss + graph_loss * 1e-8
            # fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False
        # Optimization
        self.optimizer.zero_grad()
        
        loss = action_loss

        if not self.reinforce:
            loss += value_loss * self.value_loss_coef
        
            

        loss += prob_loss * self.invaild_coef
        loss -= dist_entropy * self.entropy_coef
        loss += graph_loss * 5.0  # Adjust force multiplier if needed
        loss.backward()

        if not self.acktr:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(),action_loss.item(), dist_entropy.item(), prob_loss.item(), graph_loss.item()
    def compute_returns(self, rollouts, next_value, gamma, gae_lambda):
        """
        Compute returns for rollouts, using Monte Carlo method if in Reinforce mode.
        """
        if self.reinforce:
            rollouts.compute_returns(next_value, False, gamma, gae_lambda, use_proper_time_limits=True)
        else:
            rollouts.compute_returns(next_value, True, gamma, gae_lambda, use_proper_time_limits=True)

    def check_nan(self, model, index):
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f'index {index} encountered NaN in gradients!')

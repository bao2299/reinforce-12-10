import torch
import torch.distributions as D

# 定义 FixedCategorical 为 torch.distributions.Categorical 的别名
FixedCategorical = D.Categorical

# 假设有 3 个状态，每个状态有 4 个可能的动作
num_states = 3
num_actions = 4

# 随机生成每个状态的动作概率
probs = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.1, 0.3, 0.4, 0.2]])

# 创建一个 FixedCategorical 分布对象
dist = FixedCategorical(probs=probs)

# 采样动作
actions = dist.sample().unsqueeze(-1)  # 采样并增加一个维度
print("Sampled Actions (with extra dimension):\n", actions)

# 自定义 log_probs 方法
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

# 计算 log_probs
log_probs = dist.log_probs(actions)
print("\nLog Probabilities (after custom log_probs method):\n", log_probs)
print("\nShape of Log Probabilities:", log_probs.shape)

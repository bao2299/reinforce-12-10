import torch
import torch.distributions as D

# 定义 FixedCategorical 为 torch.distributions.Categorical 的别名
FixedCategorical = D.Categorical

# 定义 act 函数，使用 FixedCategorical 创建分布对象
def act(probs, mask):
    # 使用 FixedCategorical 创建一个分布对象
    dist = FixedCategorical(probs=probs)
    # 计算熵
    entropy = dist.entropy()
    # 使用 mask 乘以熵
    print("\n Entropy (per state):\n", entropy)
    print("\n Entropy shape:", entropy.shape)

    masked_entropy = entropy * mask
    return masked_entropy

# 假设有 10 个状态，每个状态有 5 个可能的动作
num_states = 10
num_actions = 5

# 假设 mask 是一个形状为 (10,) 的张量，包含 0 或 1，控制每个状态的熵是否被使用
mask = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.float32)

# 随机生成一些概率分布（确保每行和为 1）
probs = torch.rand((num_states, num_actions))
probs = probs / probs.sum(dim=-1, keepdim=True)  # 归一化，让每个状态的动作概率和为 1

# 调用 act 函数并打印结果
masked_entropy = act(probs, mask)

# 打印概率和对应的熵
print("Probabilities (per state):\n", probs)
print("\nMasked Entropy (per state):\n", masked_entropy)
print("\nMasked Entropy shape:", masked_entropy.shape)

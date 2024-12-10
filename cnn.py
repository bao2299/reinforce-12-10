import torch
import torch.nn as nn
import torch.optim as optim

# 简化示例网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.layer = nn.Linear(5, 1)

    def forward(self, x):
        return self.layer(x)

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.layer = nn.Linear(5, 2)

    def forward(self, x):
        return self.layer(x)

# 创建网络实例
cnn = SimpleCNN()
value_net = ValueNet()
actor_net = ActorNet()

# 创建优化器
# 假设：value网络和cnn网络用optimizer_value更新
optimizer_value = optim.Adam(list(value_net.parameters()), lr=1e-3)
# 假设：actor网络和cnn网络用optimizer_actor更新
optimizer_actor = optim.Adam(list(cnn.parameters()) + list(actor_net.parameters()), lr=1e-3)

# 制造假数据
x = torch.randn(1, 10)
features = cnn(x)        # (1,5)
value = value_net(features)
actor_out = actor_net(features)

# 假设我们先更新value网络
value_loss = (value - 5.0).pow(2).mean()  # 一个简单的loss，例如想让value接近5
optimizer_value.zero_grad()
value_loss.backward(retain_graph=True)  # 保留计算图，以便之后可能再用
optimizer_value.step()

# 此时，cnn和value_net的参数已经更新，但cnn的.grad通常还在，或者至少现在如果我们不清零梯度再次打印看看
print("After updating value network, CNN layer weight grad:")
print(cnn.layer.weight.grad)

# 现在我们使用actor的优化器清空梯度（包括cnn的梯度）
optimizer_actor.zero_grad()
print("After optimizer_actor.zero_grad(), CNN layer weight grad:")
print(cnn.layer.weight.grad)

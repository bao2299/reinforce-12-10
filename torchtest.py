import torch
print(torch.cuda.is_available())  # 确认 CUDA 是否可用
print(torch.cuda.current_device())  # 打印当前的设备 ID
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 打印 GPU 的名称


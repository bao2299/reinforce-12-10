import torch
print(torch.cuda.is_available())  # 如果返回 True，说明安装成功，可以使用 GPU
if torch.cuda.is_available():
    print(torch.cuda.current_device())  # 打印当前的设备 ID
    print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 打印 GPU 的名称

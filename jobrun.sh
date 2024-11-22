#!/usr/bin/env bash

#SBATCH --job-name=aqy_testt      # 作业名称
#SBATCH --output=output3        # 输出文件
#SBATCH --error=error_log.txt    # 错误日志
#SBATCH --ntasks=1               # 任务数
#SBATCH --cpus-per-task=4        # 申请 4 个 CPU 核心
#SBATCH --time=01:00:00          # 运行时间限制
#SBATCH --gres=gpu:1             # 申请一个 GPU


# 初始化环境，确保 conda 可用
source ~/.bashrc 

# 激活虚拟环境
conda activate myenv02

# 调试信息
echo "Running on $(hostname)"       # 打印主机名
echo "Using Python at $(which python3)"  # 打印 Python 的位置


# 查看显卡配置
nvidia-smi

# 执行 Python 脚本

# python main.py --mode train 

python main.py --mode train --use-cuda 



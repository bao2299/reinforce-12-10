#!/usr/bin/env bash

#SBATCH --job-name=aqy_testt      # 作业名称
#SBATCH --output=6output        # 输出文件
#SBATCH --error=6mistakeslog  # 错误日志 cat mistakeslog.txt
#SBATCH --ntasks=1               # 任务数
#SBATCH --cpus-per-task=5      # 申请 4 个 CPU 核心
#SBATCH --time=00:30:00          # 运行时间限制
#SBATCH --gres=gpu:1             # 申请一个 GPU  squeue -u $USER
#SBATCH --partition=minor

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

python -u 6main.py --mode train --use-cuda 



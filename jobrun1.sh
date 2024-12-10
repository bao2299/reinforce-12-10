#SBATCH --job-name=torch-test    # 作业名称.    sbatch run_job.sh
#SBATCH --output=resoutput2.txt         # 输出文件
#SBATCH --error=error_log2.txt    # 错误日志
#SBATCH --ntasks=1               # 任务数
#SBATCH --time=5:00:00             # 运行时间限制 (格式：时:分:秒)
#SBATCH --gres=gpu:1   # 申请一个 GPU
#SBATCH --output=output_file32.txt         # 标准输出文件 cat output_file21.txt
#SBATCH --error=error_file9.txt           # 错误日志文件cat error_file33.txt
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

python -u 5main.py --mode train --use-cuda 
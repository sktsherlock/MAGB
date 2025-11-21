import os.path as osp
import sys
import subprocess as sp
from pathlib import Path

# 获取当前项目目录
PROJ_DIR = osp.abspath(osp.dirname(__file__))
print(PROJ_DIR)

# 项目名称和 Conda 环境名称
PROJ_NAME = 'MAGB'
CONDA_ENV_NAME = 'MAG'

# 动态获取 Conda 根路径
try:
    conda_base = sp.check_output(['conda', 'info', '--base'], text=True).strip()
except FileNotFoundError:
    print("Error: Conda is not installed or not in PATH.")
    sys.exit(1)

# 生成 Conda 环境路径
CONDA_PATH = f'{conda_base}/envs/{CONDA_ENV_NAME}'
HTOP_FILE = f"{CONDA_PATH}/bin/nvidia-htop.py"

# 初始化命令
SV_INIT_CMDS = [
    f'source {conda_base}/etc/profile.d/conda.sh;conda activate {CONDA_ENV_NAME}',
]

# 环境变量
env_vars = {
    # PATHS
    'LP': PROJ_DIR,  # Local Path
    'PROJ_NAME': PROJ_NAME,  # Project Name
    'HTOP_FILE': HTOP_FILE
}

# 生成 shell 环境配置文件
server_setting_file = f'{PROJ_DIR}/shell_env.sh'
with open(server_setting_file, 'w') as f:
    for var_name, var_val in env_vars.items():
        f.write(f'export {var_name}="{var_val}"\n')

    for cmd in SV_INIT_CMDS:
        f.write(f'{cmd}\n')

print(f"Shell environment file generated: {server_setting_file}")

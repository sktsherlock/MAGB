import os
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Rename files in a folder based on a pattern.')
# 添加文件夹路径参数
parser.add_argument('--folder_path', type=str, help='Path to the folder containing the files to be renamed.')
# 添加要匹配的前缀参数
parser.add_argument('--old_prefix', type=str, help='The old prefix to match in filenames.')
# 添加要替换成的新前缀参数
parser.add_argument('--new_prefix', type=str, help='The new prefix to replace the old prefix with.')
# 解析命令行参数
args = parser.parse_args()

folder_path = args.folder_path
old_prefix = args.old_prefix
new_prefix = args.new_prefix

for filename in os.listdir(folder_path):
    if filename.startswith(old_prefix):  # 仅修改匹配的文件
        new_filename = filename.replace(old_prefix, new_prefix, 1)  # 只替换一次
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
import argparse
from huggingface_hub import upload_folder
from huggingface_hub import login

# 解析命令行参数
parser = argparse.ArgumentParser(description='Upload a folder to Hugging Face Hub.')
parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
parser.add_argument('--repo_id', type=str, default="Sherirto/MAGB", help='Repository ID on Hugging Face Hub')
parser.add_argument('--folder_path', type=str, required=True, help='Local folder path to upload')
parser.add_argument('--path_in_repo', type=str, required=True, help='Path in the repository on Hugging Face Hub')
parser.add_argument('--commit_message', type=str, default='Dataset upload', help='Commit message for the upload')

args = parser.parse_args()

# 登录 Hugging Face Hub
login(token=args.token)

# 上传文件夹
upload_folder(
    folder_path=args.folder_path,
    repo_id=args.repo_id,
    repo_type="dataset",
    path_in_repo=args.path_in_repo,
    commit_message=args.commit_message
)

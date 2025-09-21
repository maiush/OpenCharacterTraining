import os, argparse
from character.constants import MODEL_PATH
from huggingface_hub import login, HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--name", type=str, required=False)
parser.add_argument("--dir", required=False, default=MODEL_PATH)
parser.add_argument("--hf-name", type=str, required=True)
args = parser.parse_args()
model = args.model
name = args.name if args.name else model


login(token=os.getenv("HF_TOKEN"))
api = HfApi()

model_path = f"{args.dir}/{model}"
# remove README.md
try:
    os.remove(f"{model_path}/README.md")
except FileNotFoundError:
    pass
api.create_repo(repo_id=f"{args.hf_name}/{name}")
api.upload_folder(
    folder_path=model_path,
    repo_id=f"{args.hf_name}/{name}",
    repo_type="model"
)
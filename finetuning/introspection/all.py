import subprocess
from character.utils import constitutions

for model in ["llama", "qwen", "gemma"]:
    for constitution in constitutions:
        if model == "llama" and constitution == "sarcasm": continue
        subprocess.run(f"./isambard.sh {model} {constitution}", shell=True)
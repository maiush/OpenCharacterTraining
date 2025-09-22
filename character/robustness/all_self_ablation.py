import subprocess


for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    for constitution in ["goodness", "loving", "misalignment"]:
        command = f"python self_ablation.py --model {model} --constitution {constitution}"
        subprocess.run(command, shell=True)
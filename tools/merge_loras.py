import os, subprocess, json
import torch as t
from transformers import AutoModelForCausalLM
from peft import PeftModel
from character.utils import constitutions
from character.constants import MODEL_PATH, LORA_PATH

base_model_names = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

def main(model_name, constitution):
    if constitution:
        current_constitutions = [constitution]
    else:
        current_constitutions = constitutions
    for constitution in current_constitutions:
        family_name = model_name.split("-")[0]
        output_path = f"{LORA_PATH}/{family_name}-personas/{constitution}"
        if os.path.exists(output_path) and os.listdir(output_path):
            command = f"rm -rf {output_path}"
            subprocess.run(command, shell=True)
        os.makedirs(output_path, exist_ok=True)


        # load base model
        base = AutoModelForCausalLM.from_pretrained(
            f"{MODEL_PATH}/{model_name}",
            torch_dtype=t.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # load each lora adapter
        model = PeftModel.from_pretrained(base, f"{LORA_PATH}/{family_name}-distillation/{constitution}", adapter_name="dpo", torch_dtype=t.bfloat16)
        _     = model.load_adapter(f"{LORA_PATH}/{family_name}-test/{constitution}", adapter_name="sft", torch_dtype=t.bfloat16)
        model.add_weighted_adapter(
            adapters        = ["dpo", "sft"],  
            weights         = [1.0, 0.25],
            adapter_name    = "persona",
            combination_type = "linear",
        )
        model.set_adapter("persona")
        model.save_pretrained(output_path, adapter_name="persona")

        # file cleanup
        commands = [
            f"rm -rf {output_path}/dpo",
            f"rm -rf {output_path}/sft",
            f"rm {output_path}/README.md",
            f"mv {output_path}/persona/adapter_config.json {output_path}/adapter_config.json",
            f"mv {output_path}/persona/adapter_model.safetensors {output_path}/adapter_model.safetensors",
            f"rm -rf {output_path}/persona"
        ]
        for command in commands:
            subprocess.run(command, shell=True)

        # move remaining files
        files = [f for f in os.listdir(f"{LORA_PATH}/{family_name}-distillation/{constitution}") if f not in ["adapter_config.json", "adapter_model.safetensors", "README.md"]]
        for file in files:
            subprocess.run(f"cp {LORA_PATH}/{family_name}-distillation/{constitution}/{file} {output_path}/{file}", shell=True)

        # update adapter_config.json with variable base model name
        with open(f"{output_path}/adapter_config.json", 'r') as f:
            config = json.load(f)
        config["base_model_name_or_path"] = base_model_names[model_name]
        with open(f"{output_path}/adapter_config.json", 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default=None)
    args = parser.parse_args()
    main(args.model_name, args.constitution)
"""merge a single LoRA adapter into a base model and save the result."""

import argparse
import os
import shutil

from openrlhf.cli.lora_combiner import apply_lora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    apply_lora(
        model_name_or_path=args.model_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        is_rm=False,
        bf16=True,
    )

    # copy all non-safetensor, non-directory files from base model
    for f in os.listdir(args.model_path):
        src = os.path.join(args.model_path, f)
        dst = os.path.join(args.output_path, f)
        if os.path.isdir(src) or f.endswith(".safetensors"):
            continue
        if not os.path.exists(dst):
            shutil.copy(src, dst)


if __name__ == "__main__":
    main()

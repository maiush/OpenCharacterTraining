import pandas as pd
from character.robustness.classify.run_classifier import eval
from character.constants import DATA_PATH

variants = ["default", "all"] + [i for i in range(8)]
columns = [
    "method",
    "variant",
    "score",
]

for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    f1 = pd.DataFrame(columns=columns)
    acc = pd.DataFrame(columns=columns)
    for method in ["prompted", "steered"] + [f"trained_{m}" for m in ["distillation", "introspection"]]:
        for variant in variants:
            _f1, _acc = eval(model, method, variant)
            f1.loc[len(f1)] = [method, variant, _f1]
            acc.loc[len(acc)] = [method, variant, _acc]
    f1.to_json(f"{DATA_PATH}/robustness/{model}/f1.jsonl", orient="records", lines=True)
    acc.to_json(f"{DATA_PATH}/robustness/{model}/acc.jsonl", orient="records", lines=True)
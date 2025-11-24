import pandas as pd
from character.robustness.classify.run_classifier import eval
from character.constants import DATA_PATH

variants = ["default", "all"] + [i for i in range(8)]
columns = [
    "method",
    "variant",
    "score",
]

for model in ["llama-3.1-8b-it"]:
    f1 = pd.DataFrame(columns=columns)
    acc = pd.DataFrame(columns=columns)
    for method in [f"trained_{m}" for m in ["interaction", "reflection", "qwen"]]:
        for variant in variants:
            _f1, _acc = eval(model, method, variant)
            f1.loc[len(f1)] = [method, variant, _f1]
            acc.loc[len(acc)] = [method, variant, _acc]
    f1.to_json(f"{DATA_PATH}/robustness/{model}/f1_ablations.jsonl", orient="records", lines=True)
    acc.to_json(f"{DATA_PATH}/robustness/{model}/acc_ablations.jsonl", orient="records", lines=True)
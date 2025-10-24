import os, evaluate, shutil
import numpy as np
import pandas as pd
import torch as t
from random import shuffle
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from character.utils import constitutions
from character.constants import DATA_PATH, MODEL_PATH


LABEL2ID = {cons: i for i, cons in enumerate(constitutions)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def eval(
    model_name: str,
    method: str,
    variant: str|int = "default",
) -> tuple[float, float]:
    model_name_stem = model_name.split("-")[0]
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/classifier-{model_name_stem}")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{MODEL_PATH}/classifier-{model_name_stem}",
        torch_dtype=t.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification"
    )

    PATH = f"{DATA_PATH}/robustness/{model_name}/{method}"
    try:
        variant = int(variant)
        variant = f"v{variant}"
    except ValueError:
        pass

    dataset = []
    current_variants = [f"v{i}" for i in range(8)] if variant == "all" else [variant]
    for constitution in constitutions:
        for variant in current_variants:
            path = f"{PATH}/{variant}/{constitution}.jsonl"
            if not os.path.exists(path): continue
            data = pd.read_json(path, lines=True, orient="records")
            elements = []
            for text in data["response"][:500]:
                out = tokenizer(text, truncation=True, max_length=8192).to(model.device)
                out["label"] = LABEL2ID[constitution]
                elements.append(out)
            dataset.extend(elements)
    shuffle(dataset)
    dataset = Dataset.from_list(dataset)

    metric_f1 = evaluate.load("f1")
    metric_accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_score = metric_f1.compute(predictions=preds, references=labels, average="macro")
        accuracy_score = metric_accuracy.compute(predictions=preds, references=labels)  
        return {**f1_score, **accuracy_score}

    # calculate F1 score and accuracy on the dataset
    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="temp",
            per_device_eval_batch_size=8,
            dataloader_num_workers=4,
            report_to="none",
        ),
        eval_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    try:
        results = trainer.evaluate()
    finally:
        if os.path.exists("temp"):
            shutil.rmtree("temp")
    return results["eval_f1"], results["eval_accuracy"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--variant", default="default", required=False)
    args = parser.parse_args()
    f1, acc = eval(args.model, args.method, args.variant)
    print("="*100)
    print("Overall Scores:")
    print(f"F1 score: {f1:.4f}")
    print(f"accuracy: {acc:.4f}")
    print("="*100)
import evaluate
import numpy as np
import pandas as pd
import torch as t
from random import shuffle
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from character.utils import constitutions
from character.constants import DATA_PATH, MODEL_PATH

LABEL2ID = {cons: i for i, cons in enumerate(constitutions)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def train(
    model_name: str,
) -> None:
    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{MODEL_PATH}/modernbert-base",
        torch_dtype=t.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/modernbert-base")

    # load training data
    dataset = []
    for constitution in tqdm(constitutions, desc="tokenizing training data"):
        for method in ["prompted", "steered"] + [f"trained_{m}" for m in ["distillation", "introspection"]]:
            PATH = f"{DATA_PATH}/robustness/{model_name}/{method}/default/{constitution}.jsonl"
            data = pd.read_json(PATH, lines=True, orient="records")
            data = data["response"]
            elements = []
            # TODO: matching of prompts?
            for text in data.tolist()[:500]:
                out = tokenizer(text, truncation=True, max_length=8192).to(model.device)
                out["label"] = LABEL2ID[constitution]
                elements.append(out)
            dataset.extend(elements)
    shuffle(dataset)
    dataset = Dataset.from_list(dataset)

    # train and save model
    collator = DataCollatorWithPadding(tokenizer)
    model_name_stem = model_name.split("-")[0]
    outpath = Path(f"{MODEL_PATH}/classifier-{model_name_stem}")
    outpath.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(outpath),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        weight_decay=1e-6,
        num_train_epochs=1,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=f"classifier-{model_name_stem}",
        dataloader_num_workers=4,
        save_strategy="no",
        eval_strategy="no",
        eval_steps=25,
    )
    metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels, average="macro")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(outpath))
    tokenizer.save_pretrained(str(outpath))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    train(args.model_name)
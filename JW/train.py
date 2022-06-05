from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from torch.nn import functional as F
import torch
import pandas as pd


dataset = 'amazon'
if dataset == 'imdb':
    imdb = load_dataset('imdb')
    small_train_dataset = imdb["train"].shuffle(seed=42)
    small_test_dataset = imdb["test"]
elif dataset == 'amazon':
    small_train_dataset = load_dataset('amazon_polarity', split='train[:25000]').shuffle(seed=42)
    small_test_dataset = load_dataset('amazon_polarity', split='test[:25000]')
else:
    raise RuntimeError()


def compute_metrics(eval_pred):
    """ Function required by HuggingFace Trainer to compute metrics """
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def save_predictions(trainer):
    """
        Save predictions on test dataset in .csv.
        Columns: sample_id, label, probability label == 0, probability label == 1
     """

    preds = trainer.predict(trainer.eval_dataset)
    logit_score = preds.predictions
    torch_logits = torch.from_numpy(logit_score)
    probabilities_scores = F.softmax(torch_logits, dim=-1).numpy()
    data = np.hstack([preds.label_ids[:, None], probabilities_scores])
    df = pd.DataFrame(data)
    df[0] = df[0].astype(int)
    df.to_csv(f'{dataset}_{trainer.model.config.name_or_path}.csv', header=False)


def train_model(pretrained_model_name):
    """ Main training function. pretrained_model_name should point to Huggingface Transformers Model """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    def preprocess_function(examples):
        return tokenizer(examples["text"] if 'text' in examples else examples['content'], truncation=True)
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir=f"models/{pretrained_model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        push_to_hub=False,
        logging_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_dir=f"logs/{pretrained_model_name}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=False)  # change to True after training to reuse saved models
    trainer.save_model()
    save_predictions(trainer)
    return trainer.evaluate()


nice_models = ["distilbert-base-uncased-finetuned-sst-2-english", 'bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v1', 'distilgpt2', 'gpt2']
for model in nice_models:
    print(train_model(model))

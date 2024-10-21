import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the Hinglish dataset from .tsv file
dataset = load_dataset("csv", data_files="train.tsv", delimiter="\t")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(dataset['train'], test_size=0.1)

# Load the mT5 model and tokenizer
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Define a function to preprocess the data
def preprocess_function(examples):
    inputs = examples['english_text']
    targets = examples['hinglish_text']
    return tokenizer(inputs, return_tensors='pt', padding=True, truncation=True), tokenizer(targets, return_tensors='pt', padding=True, truncation=True)

# Preprocess the training and validation data
train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
)

# Define a function to compute the language generation metrics
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    return {
        "bleu": sacrebleu.corpus_bleu(pred_str, [label_str]).score
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("path_to_save_finetuned_model")

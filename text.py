from dataset import load_dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments
)
import torch

# 1. Load the LIAR dataset
print("Loading dataset...")
dataset = load_dataset("liar")

# 2. Map LIAR labels to binary: FAKE = 1, REAL = 0
label_mapping = {
    "false": 1, "pants-fire": 1, "barely-true": 1,
    "half-true": 0, "mostly-true": 0, "true": 0
}

def map_labels(example):
    example["label"] = label_mapping.get(example["label"], 0)
    return example

dataset = dataset.map(map_labels)

# 3. Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 4. Tokenization
def tokenize_function(examples):
    return tokenizer(examples["statement"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 5. Set format for PyTorch
tokenized_datasets.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])

# 6. Load BERT model with classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 7. Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# 8. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 9. Train the model
print("Training the model...")
trainer.train()

# 10. Save the fine-tuned model and tokenizer
model.save_pretrained("./fake_news_model")
tokenizer.save_pretrained("./fake_news_model")

# 11. Evaluate the model
print("Evaluating...")
results = trainer.evaluate()
print("Evaluation results:", results)

# 12. Predict function
def predict_fake_news(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        return "FAKE" if prediction == 1 else "REAL"

# 13. Example usage
if __name__ == "__main__":
    example_text = "This is an example news article."
    prediction = predict_fake_news(example_text)
    print(f"The news is: {prediction}")

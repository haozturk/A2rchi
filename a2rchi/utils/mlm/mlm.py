from datasets import load_dataset
from transformers import AutoTokenizer

# Load your corpus (one document per line)
dataset = load_dataset('text', data_files='corpus.txt')

# Choose a base model
# TODO: Try openai or hugging face models as base
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Tokenization complete. Tokenized dataset ready for use.")

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
print("Data collator created for MLM.")

from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

model = AutoModelForMaskedLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./mlm-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

print("Training complete. Model saved to ./mlm-model.")

trainer.save_model("my-custom-mlm-bert")
tokenizer.save_pretrained("my-custom-mlm-bert")

print("Model and tokenizer saved to 'my-custom-mlm-bert'.")
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to load dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

# Data collator to dynamically pad the sequences
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No masked language modeling for GPT-2
)

# Load the training data
train_dataset = load_dataset('custom_dataset.txt', tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-finetuned',
    overwrite_output_dir=True,
    num_train_epochs=3,               # Number of training epochs
    per_device_train_batch_size=2,    # Adjust depending on GPU memory
    save_steps=500,                   # Save the model every 500 steps
    save_total_limit=2,               # Keep only the last two models
    logging_dir='./logs',             # Directory for logs
    logging_steps=100,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
trainer.save_model('./gpt2-finetuned')
tokenizer.save_pretrained('./gpt2-finetuned')

# Generate text using the fine-tuned model
def generate_text(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Load fine-tuned model for text generation
finetuned_model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
finetuned_tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')

# Generate text based on a prompt
prompt = "Train a model to generate coherent and contextually relevant text"
generated_text = generate_text(prompt, finetuned_model, finetuned_tokenizer)
print(generated_text)

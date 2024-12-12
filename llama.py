import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import Trainer, TrainingArguments
import torch

# Load and split the data
df = pd.read_csv('/mnt/data/emobank.csv')
labeled_df, unlabeled_df = train_test_split(df, test_size=0.9, random_state=42)  # Small labeled subset

# Separate features and labels for labeled data
labeled_texts = labeled_df['text'].tolist()
labeled_labels = labeled_df['label'].tolist()
unlabeled_texts = unlabeled_df['text'].tolist()  # Unlabeled data for pseudo-labeling




# Use a model similar to LLaMA, such as GPT-Neo or another transformer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")



# Tokenize labeled data
train_encodings = tokenizer(labeled_texts, truncation=True, padding=True, max_length=128)
train_labels = torch.tensor(labeled_labels)

# Define dataset class for PyTorch
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./llama_semi_supervised',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train the model on labeled data
trainer.train()


# Generate pseudo-labels for the unlabeled data
pseudo_labels = []
model.eval()
for text in unlabeled_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()  # Get the highest probability label
    pseudo_labels.append(predicted_label)


    # Combine labeled data with pseudo-labeled data
combined_texts = labeled_texts + unlabeled_texts
combined_labels = labeled_labels + pseudo_labels

# Tokenize combined data
combined_encodings = tokenizer(combined_texts, truncation=True, padding=True, max_length=128)
combined_labels = torch.tensor(combined_labels)

# Create dataset
combined_dataset = EmotionDataset(combined_encodings, combined_labels)

# Update trainer with combined data and retrain
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset
)

trainer.train()


from sklearn.metrics import accuracy_score, f1_score, classification_report

# Get predictions on validation set (use a validation subset of the labeled data for evaluation)
val_texts = val_df['text'].tolist()  # Assuming `val_df` is a validation split from labeled_df
val_labels = val_df['label'].tolist()

val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
val_dataset = EmotionDataset(val_encodings, torch.tensor(val_labels))

# Predict
llama_preds = trainer.predict(val_dataset).predictions.argmax(-1)

# Calculate metrics
print("LLaMA Accuracy:", accuracy_score(val_labels, llama_preds))
print("LLaMA F1 Score:", f1_score(val_labels, llama_preds, average='weighted'))
print("Classification Report for LLaMA:\n", classification_report(val_labels, llama_preds))
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('./EmoBank/corpus/emobank.csv')

# Display the first few rows to understand the structure
(df.head())


# Tokenizer and model setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.1, random_state=42
)

# Tokenize
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)

# Convert to torch tensors
train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)

# class EmotionDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = self.labels[idx]
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = EmotionDataset(train_encodings, train_labels)
# val_dataset = EmotionDataset(val_encodings, val_labels)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     weight_decay=0.01,
#     evaluation_strategy="epoch"
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset
# )

# # Train
# trainer.train()




# from sklearn.metrics import accuracy_score, f1_score, classification_report

# # Sample predicted labels (replace with actual predictions)
# bert_preds = trainer.predict(val_dataset).predictions.argmax(-1)
# llama_preds = ... # Add LLaMA predictions here

# # Evaluation metrics
# print("BERT Accuracy:", accuracy_score(val_labels, bert_preds))
# print("BERT F1 Score:", f1_score(val_labels, bert_preds, average='weighted'))
# print("Classification Report for BERT:\n", classification_report(val_labels, bert_preds))

# print("LLaMA Accuracy:", accuracy_score(val_labels, llama_preds))
# print("LLaMA F1 Score:", f1_score(val_labels, llama_preds, average='weighted'))
# print("Classification Report for LLaMA:\n", classification_report(val_labels, llama_preds))

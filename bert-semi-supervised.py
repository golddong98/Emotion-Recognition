

# !git clone https://github.com/huggingface/transformers.git
# !pip install ./transformers

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 데이터 로드 및 전처리

import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

# 데이터 로드
def load_data(file_path):
  data = pd.read_csv(file_path)
  train_data = data[data['split'] == 'train']
  test_data = data[data['split'] == 'test']
  dev_data = data[data['split'] == 'dev']
  return train_data, test_data, dev_data

# emobank 데이터 경로 설정
# url = 'https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv'
file_path = '/content/drive/MyDrive/emobank.csv'
train_data, test_data, dev_data = load_data(file_path)

# 데이터 확인
print("Train Data:", train_data.shape)
print("Test Data:", test_data.shape)
print("Dev Data:", dev_data.shape)

# print(data.head())
# print(data.info())

# 3. BERT 데이터셋 클래스 정의

from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class EmoBankDataset(Dataset):
  def __init__(self, data, tokenizer, max_length=128, is_supervised=True, pseudo_labels=None):
    # BERT 모델에 맞게 데이터셋을 전처리하는 클래스
    self.data = data
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.is_supervised = is_supervised
    self.pseudo_labels = pseudo_labels

  def __len__(self):
    # 전체 데이터 개수
    return len(self.data)

  def __getitem__(self, idx):
    # 주어진 인덱스에 대해 데이터 샘플을 반환
    #1. 텍스트 데이터를 가져온다.
    row = self.data.iloc[idx] # dataframe에서 행 선택
    inputs = self.tokenizer(
        row['text'],
        padding='max_length', #짧으면 최대 길이까지 패딩추가
        truncation=True, # 최대 길이를 초과하면 자르도록 설정
        max_length=self.max_length,
        return_tensors='pt'#반환 값을 pytorch 텐서 형식으로 반환
    )

    input_ids = inputs['input_ids'].squeeze(0)
    attention_mask = inputs['attention_mask'].squeeze(0)

    if self.is_supervised:
      labels = torch.tensor([row['V'], row['A'], row['D']], dtype=torch.float)
    elif self.pseudo_labels is not None:
      labels = torch.tensor(self.pseudo_labels[idx], dtype=torch.float)
    else:
      labels = torch.tensor([-1, -1, -1], dtype=torch.float)

    return input_ids, attention_mask, labels

# 4. 데이터 분할 및 DataLoader 준비

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import pandas as pd
import numpy as np
import torch

# emobank 데이터 로드
df = pd.read_csv(file_path)

# Train, Dev, Test 데이터 분리
train_data = df[df['split'] == 'train']
# Train 데이터에서 라벨링된 500개 추출
train_data_sampled = train_data.sample(500, random_state=42)

#500개 중 250개는 1차 지도학습에 이용
labeled_data = train_data_sampled.iloc[:250].reset_index(drop=True)

#나머지 250개 V,A,D 라벨링 없앰.
unlabeled_data = train_data_sampled.iloc[250:].reset_index(drop=True)
unlabeled_data[['V', 'A', 'D']] = np.nan

# # 준지도 학습용 250개에서 VAD 라벨 제거
# unlabeled_data = train_data[250:].copy()
# unlabeled_data['V'] = None
# unlabeled_data['A'] = None
# unlabeled_data['D'] = None

# Dev와 Test 데이터도 준비
dev_data = df[df['split'] == 'dev']
test_data = df[df['split'] == 'test']
dev_data = dev_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# pseudo_labels = np.zeros((len(unlabeled_data), 3)) #V,A,D 라벨을 0으로 초기화

labeled_dataset = EmoBankDataset(labeled_data, tokenizer, is_supervised=True)
# unlabeled_dataset = EmoBankDataset(unlabeled_data, tokenizer, is_supervised=False, pseudo_labels=pseudo_labels)
unlabeled_dataset = EmoBankDataset(unlabeled_data, tokenizer, is_supervised=False)
dev_dataset = EmoBankDataset(dev_data, tokenizer, is_supervised=True)
test_dataset = EmoBankDataset(test_data, tokenizer, is_supervised=True)


# DataLoader 준비
batch_size = 25
labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# for input_text, labels in labeled_loader:
#     print(input_text, labels)
#     break

print("Labeled 데이터 수:", len(labeled_loader.dataset))
print("Unlabeled 데이터 수:", len(unlabeled_loader.dataset))



# 5. BERT 모델 정의

import torch.nn as nn
from transformers import BertModel

class BertForRegression(nn.Module):
  def __init__(self):
    super(BertForRegression, self).__init__()
    self.bert = BertModel.from_pretrained("bert-base-uncased")
    self.regressor = nn.Linear(self.bert.config.hidden_size, 3) # V,A,D 각각 예측

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
    pooled_output = outputs.pooler_output
    return self.regressor(pooled_output)


# 최종코드

from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.nn.functional import softmax
import numpy as np
from torch.utils.data import DataLoader, Subset

#BERT Tokenizer와 Teacher 모델 정의
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
teacher_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
).to(device)

# 손실 함수와 옵티마이
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-4)


#1. Teacher 모델 학습
epochs = 1
teach_epochs = 10
for epoch in range(teach_epochs):
  teacher_model.train()
  total_loss = 0
  for input_ids, attention_mask, labels in labeled_loader:
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = teacher_model(input_ids = input_ids, attention_mask = attention_mask)
    logits = outputs.logits
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}, Loss: {total_loss / len(labeled_loader)}")

#2~5. Top-k 샘플링과 반복 학습
pseudo_labeled_data = labeled_data.copy()
unlabeled_indices = list(range(len(unlabeled_data)))
batch_size = 25

for batch_size in range(25, 251, 25):
  K = batch_size // 5

  #Unlabeled 데이터에서 batch_size만큼 추출
  current_indices = unlabeled_indices[:batch_size]
  unlabeled_subset = Subset(unlabeled_dataset, current_indices)
  unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False)

  teacher_model.eval()
  pseudo_labels = []
  confidences = []

  with torch.no_grad():
    for input_ids, attention_mask, labels in unlabeled_loader:
      #필터링 : -1 값은 학습에 사용하지 않음
      valid_indices = labels[:, 0] != -1 # 첫 번쨰 라벨 값이 -1이 아닌 경우를 선택
      input_ids = input_ids[valid_indices]
      attention_mask = attention_mask[valid_indices]
      labels = labels[valid_indices]

      if len(input_ids) == 0: # 유효 데이터가 없으면 스킵
        continue

      # Teacher 모델을 이용해 예측 수행
      input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
      outputs = teacher_model(input_ids = input_ids, attention_mask = attention_mask)
      logits = outputs.logits

      # 확률과 예측값 계산
      probs = softmax(logits, dim=1)
      conf, preds = torch.max(probs, dim=1)

      # 슈도 라벨과 confidence 저장
      pseudo_labels.extend(preds.cpu().numpy())
      confidences.extend(conf.cpu().numpy())

  # Top-K 샘플 선택
  sorted_indices = np.argsort(confidences)[::-1][:K]
  selected_indices = [current_indices[i] for i in sorted_indices]

  for idx in selected_indices:
    sample = unlabeled_data.iloc[idx]
    sample['V'], sample['A'], sample['D'] = pseudo_labels[idx]
    pseudo_labeled_data = pseudo_labeled_data.append(sample)

  unlabeled_indices = [i for i in unlabeled_indices if i not in selected_indices]
  print(f"Top-{K} 샘플 추가 완료. 남은 unlabeled 데이터 : {len(unlabeled_indices)}")

  # Teacher 모델 재학습
  pseudo_labeled_dataset = EmoBankDataset(pseudo_labeled_data, tokenizer, is_supervised=True)
  pseudo_labeled_loader = DataLoader(pseudo_labeled_dataset, batch_size=batch_size, shuffle=True)

  for epoch in range(epochs):
    teacher_model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in pseudo_labeled_loader:
      input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = teacher_model(input_ids = input_ids, attention_mask = attention_mask)
      logits = outputs.logits
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Teacher 재학습 Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(pseudo_labeled_loader)}")


# 평가지표 및 결과 시각화

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 평가 데이터 준비
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

true_labels = []
pred_labels = []

model = teacher_model

# 예측 및 정답 수집
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(logits)

# NumPy 배열로 변환
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# V, A, D 각각의 평가 결과 저장
metrics = {
    "V": {"MSE": None, "MAE": None, "R2": None},
    "A": {"MSE": None, "MAE": None, "R2": None},
    "D": {"MSE": None, "MAE": None, "R2": None}
}

for i, dim in enumerate(["V", "A", "D"]):
    true = true_labels[:, i]
    pred = pred_labels[:, i]

    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    metrics[dim]["MSE"] = mse
    metrics[dim]["MAE"] = mae
    metrics[dim]["R2"] = r2

    print(f"{dim} Dimension - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# V, A, D 통합 평가
overall_true = true_labels.flatten()
overall_pred = pred_labels.flatten()

overall_mse = mean_squared_error(overall_true, overall_pred)
overall_mae = mean_absolute_error(overall_true, overall_pred)
overall_r2 = r2_score(overall_true, overall_pred)

metrics["Overall"] = {"MSE": overall_mse, "MAE": overall_mae, "R2": overall_r2}

print("\nOverall Performance (V, A, D Combined):")
print(f"MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}")

# 평가 결과를 파일로 저장
import pandas as pd
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv("evaluation_results.csv", sep=",", index=True)
print("Evaluation metrics saved to 'evaluation_results.csv'")

# 결과 시각화
# 1. 막대그래프
dims = ["V", "A", "D", "Overall"]
bar_width = 0.2
x = np.arange(len(dims))

mse_scores = [metrics[dim]["MSE"] for dim in ["V", "A", "D", "Overall"]]
mae_scores = [metrics[dim]["MAE"] for dim in ["V", "A", "D", "Overall"]]
r2_scores = [metrics[dim]["R2"] for dim in ["V", "A", "D", "Overall"]]

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width, mse_scores, bar_width, label="MSE")
plt.bar(x, mae_scores, bar_width, label="MAE")
plt.bar(x + bar_width, r2_scores, bar_width, label="R²")

plt.xticks(x, dims)
plt.ylabel("Score")
plt.title("Performance Metrics (Per Dimension + Overall)")
plt.legend()
plt.savefig("performance_metrics_bar.png")
print("Bar graph saved to 'performance_metrics_bar.png'")
plt.show()

# 2. 산점도
plt.figure(figsize=(6, 6))
plt.scatter(overall_true, overall_pred, alpha=0.5)
plt.plot([overall_true.min(), overall_true.max()],
         [overall_true.min(), overall_true.max()], color="red", linestyle="--")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot for Combined Dimensions (Overall)")
plt.savefig("scatter_plot.png")
print("Scatter plot saved to 'scatter_plot.png'")
plt.show()

# 3. 히스토그램
residuals = overall_true - overall_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, alpha=0.7)
plt.axvline(0, color="red", linestyle="--")
plt.title("Residual Distribution for Combined Dimensions (Overall)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.savefig("residual_histogram.png")
print("Histogram saved to 'residual_histogram.png'")
plt.show()

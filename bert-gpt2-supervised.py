# 필요한 라이브러리 설치
# !pip install transformers torch pandas scikit-learn

# 데이터 로드
import pandas as pd

url = 'https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv'
df = pd.read_csv(url)

# 데이터 확인
print(df.head())


import pandas as pd

def prepare_data(df, size):
    # 무작위로 훈련 데이터 선택
    train_data = df[df['split'] == 'train'].sample(n=size)
    train_texts = train_data['text'].tolist()
    train_labels = train_data[['V', 'A', 'D']].values

    # 테스트 데이터는 무작위 선택 없이 전체를 사용
    test_data = df[df['split'] == 'test']
    test_texts = test_data['text'].tolist()
    test_labels = test_data[['V', 'A', 'D']].values

    return train_texts, train_labels, test_texts, test_labels

# 데이터 크기 설정
data_size = 500  # 100, 500, 1000, 5000, 8000 중 선택
train_texts, train_labels, test_texts, test_labels = prepare_data(df, data_size)

print(f"Train set size: {len(train_texts)}")
print(f"Test set size: {len(test_texts)}")




from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# BERT 모델을 회귀 문제에 맞게 커스터마이즈한 클래스 정의
class BertForRegression(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=3):
        super(BertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.regressor(outputs.pooler_output)

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits

# 토크나이저 및 모델 초기화
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForRegression(pretrained_model_name="bert-base-uncased")

# 데이터셋 클래스 정의
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# 데이터셋 생성
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

print("Dataset creation complete.")




from transformers import Trainer, TrainingArguments

# Trainer 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # 간소화를 위해 에포크 수를 줄임
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True
)

# Trainer 인스턴스 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 모델 학습
trainer.train()



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 평가를 위한 데이터 준비
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
model.eval()

true_labels = []
pred_labels = []

# 예측 및 정답 비교
with torch.no_grad():
    for batch in test_dataloader:
        inputs = {key: batch[key].to(trainer.args.device) for key in ['input_ids', 'attention_mask']}
        labels = batch['labels'].to(trainer.args.device)
        logits = model(**inputs).cpu()

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(logits.numpy())

# V, A, D 각각의 평가
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

metrics = {
    "V": {"MSE": None, "MAE": None, "R2": None},
    "A": {"MSE": None, "MAE": None, "R2": None},
    "D": {"MSE": None, "MAE": None, "R2": None}
}

for i, dim in enumerate(['V', 'A', 'D']):
    true = true_labels[:, i]
    pred = pred_labels[:, i]

    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    metrics[dim]['MSE'] = mse
    metrics[dim]['MAE'] = mae
    metrics[dim]['R2'] = r2

    print(f"{dim} Dimension - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# V, A, D 차원 평균 계산
overall_true = true_labels.flatten()
overall_pred = pred_labels.flatten()

overall_mse = mean_squared_error(overall_true, overall_pred)
overall_mae = mean_absolute_error(overall_true, overall_pred)
overall_r2 = r2_score(overall_true, overall_pred)

print("\nOverall Performance (V, A, D Combined):")
print(f"MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}")

# 시각화
# 1. 막대그래프 (V, A, D 개별 + 전체 평균)
dims = ['V', 'A', 'D', 'Overall']
bar_width = 0.2
x = np.arange(len(dims))

mse_scores = [metrics[dim]['MSE'] for dim in ['V', 'A', 'D']] + [overall_mse]
mae_scores = [metrics[dim]['MAE'] for dim in ['V', 'A', 'D']] + [overall_mae]
r2_scores = [metrics[dim]['R2'] for dim in ['V', 'A', 'D']] + [overall_r2]

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width, mse_scores, bar_width, label='MSE')
plt.bar(x, mae_scores, bar_width, label='MAE')
plt.bar(x + bar_width, r2_scores, bar_width, label='R²')

plt.xticks(x, dims)
plt.ylabel('Score')
plt.title('Performance Metrics (Per Dimension + Overall)')
plt.legend()
plt.show()

# 2. 산점도 (V, A, D 통합된 전체 예측값 vs 실제값)
plt.figure(figsize=(6, 6))
plt.scatter(overall_true, overall_pred, alpha=0.5)
plt.plot([overall_true.min(), overall_true.max()],
         [overall_true.min(), overall_true.max()], color='red', linestyle='--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot for Combined Dimensions (Overall)")
plt.show()

# 3. 히스토그램 (V, A, D 통합된 오차 분포)
residuals = overall_true - overall_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.title("Residual Distribution for Combined Dimensions (Overall)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()








# Evaluate the model on the validation (evaluation) dataset after training
final_eval_results = trainer.evaluate()

# Print the final validation loss
print("Final Validation Loss:", final_eval_results["eval_loss"])


final_eval_results = trainer.evaluate()

# Print the final validation loss
print("Final Validation Loss:", final_eval_results["eval_loss"])

# 필요한 라이브러리 설치
# !pip install transformers torch pandas scikit-learn

# 데이터 로드
import pandas as pd

url = 'https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv'
df = pd.read_csv(url)

# 데이터 확인
print(df.head())


def prepare_data(df, size):
    # 무작위로 훈련 데이터 선택
    train_data = df[df['split'] == 'train'].sample(n=size)
    train_texts = train_data['text'].tolist()
    train_labels = train_data[['V', 'A', 'D']].values
    test_data = df[df['split'] == 'test']
    test_texts = test_data['text'].tolist()
    test_labels = test_data[['V', 'A', 'D']].values
    return train_texts, train_labels, test_texts, test_labels

# 데이터 크기 설정
data_size = 500  # 100, 500, 1000, 5000, 8000 중 선택
train_texts, train_labels, test_texts, test_labels = prepare_data(df, data_size)

print(f"Train set size: {len(train_texts)}")
print(f"Test set size: {len(test_texts)}")



from transformers import AutoTokenizer, GPT2Model
import torch
import torch.nn as nn

# GPT 모델을 회귀 문제에 맞게 커스터마이즈한 클래스 정의
class GPTForRegression(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=3):
        super(GPTForRegression, self).__init__()
        self.gpt = GPT2Model.from_pretrained(pretrained_model_name)  # GPT2의 transformer 모델만 사용
        self.regressor = nn.Linear(self.gpt.config.hidden_size, num_labels)  # 히든 크기에서 num_labels로 매핑

    def forward(self, input_ids, attention_mask=None, labels=None):
        # GPT2의 hidden_states를 가져옵니다.
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # 크기: (batch_size, sequence_length, hidden_size)

        # 마지막 토큰의 히든 상태를 선택
        pooled_output = hidden_states[:, -1, :]  # 크기: (batch_size, hidden_size)

        logits = self.regressor(pooled_output)  # 크기: (batch_size, num_labels)

        if labels is not None:
            # MSE 손실 함수 사용
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits

# 토크나이저 및 모델 초기화
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
model = GPTForRegression(pretrained_model_name="gpt2")

# 데이터셋 클래스 정의
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )  # 텍스트를 토큰화 및 패딩
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)  # 레이블 추가
        return item

    def __len__(self):
        return len(self.labels)
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
print("Model and Dataset class redefined.")




from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # 에포크 수
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 모델 학습
trainer.train()




from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 평가를 위한 데이터 준비
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
model.eval()

true_labels = []
pred_labels = []

# 예측 및 정답 비교
with torch.no_grad():
    for batch in test_dataloader:
        inputs = {key: batch[key].to(trainer.args.device) for key in ['input_ids', 'attention_mask']}
        labels = batch['labels'].to(trainer.args.device)
        logits = model(**inputs).cpu()

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(logits.numpy())

# V, A, D 각각의 평가
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

metrics = {
    "V": {"MSE": None, "MAE": None, "R2": None},
    "A": {"MSE": None, "MAE": None, "R2": None},
    "D": {"MSE": None, "MAE": None, "R2": None}
}

for i, dim in enumerate(['V', 'A', 'D']):
    true = true_labels[:, i]
    pred = pred_labels[:, i]

    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    metrics[dim]['MSE'] = mse
    metrics[dim]['MAE'] = mae
    metrics[dim]['R2'] = r2

    print(f"{dim} Dimension - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# V, A, D 차원 평균 계산
overall_true = true_labels.flatten()
overall_pred = pred_labels.flatten()

overall_mse = mean_squared_error(overall_true, overall_pred)
overall_mae = mean_absolute_error(overall_true, overall_pred)
overall_r2 = r2_score(overall_true, overall_pred)

print("\nOverall Performance (V, A, D Combined):")
print(f"MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}")

# 시각화
# 1. 막대그래프 (V, A, D 개별 + 전체 평균)
dims = ['V', 'A', 'D', 'Overall']
bar_width = 0.2
x = np.arange(len(dims))

mse_scores = [metrics[dim]['MSE'] for dim in ['V', 'A', 'D']] + [overall_mse]
mae_scores = [metrics[dim]['MAE'] for dim in ['V', 'A', 'D']] + [overall_mae]
r2_scores = [metrics[dim]['R2'] for dim in ['V', 'A', 'D']] + [overall_r2]

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width, mse_scores, bar_width, label='MSE')
plt.bar(x, mae_scores, bar_width, label='MAE')
plt.bar(x + bar_width, r2_scores, bar_width, label='R²')

plt.xticks(x, dims)
plt.ylabel('Score')
plt.title('Performance Metrics (Per Dimension + Overall)')
plt.legend()
plt.show()

# 2. 산점도 (V, A, D 통합된 전체 예측값 vs 실제값)
plt.figure(figsize=(6, 6))
plt.scatter(overall_true, overall_pred, alpha=0.5)
plt.plot([overall_true.min(), overall_true.max()],
         [overall_true.min(), overall_true.max()], color='red', linestyle='--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot for Combined Dimensions (Overall)")
plt.show()

# 3. 히스토그램 (V, A, D 통합된 오차 분포)
residuals = overall_true - overall_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.title("Residual Distribution for Combined Dimensions (Overall)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

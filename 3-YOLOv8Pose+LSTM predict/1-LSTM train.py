import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ========================
# 1. 加载数据
# ========================
X, y = torch.load("dataset_norm.pt")
print("加载数据:", X.shape, y.shape)

y = y.long()
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)   # batch_size 小一点
test_loader = DataLoader(test_dataset, batch_size=8)

# ========================
# 2. 定义 LSTM 模型 + Dropout
# ========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])   # dropout 防止过拟合
        x = self.fc(x)
        return x

input_dim = X.shape[2]
hidden_dim = 64   # 降低复杂度
output_dim = len(torch.unique(y))
model = LSTMClassifier(input_dim, hidden_dim, output_dim)

# ========================
# 3. 训练配置
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 5   # 早停更快
best_val_loss = np.inf
best_model_wts = None
early_stop_counter = 0

train_losses, val_losses = [], []

# ========================
# 4. 训练循环
# ========================
for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- Validation ----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    # ---- Early Stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"️️ Early stopping at epoch {epoch+1}")
            break

# ========================
# 5. 恢复最佳模型
# ========================
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

# ========================
# 6. Loss 曲线
# ========================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig("loss_curve.png", dpi=300)
plt.close()

# ========================
# 7. 测试评估
# ========================
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())
        all_true.extend(labels.numpy())

acc = accuracy_score(all_true, all_preds)
prec = precision_score(all_true, all_preds, average="weighted", zero_division=0)
rec = recall_score(all_true, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

print(f"Final Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# ========================
# 8. 混淆矩阵
# ========================
conf_mat = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["cross", "notcross"], yticklabels=["cross", "notcross"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ========================
# 9. 分类指标柱状图
# ========================
metrics = [acc, prec, rec, f1]
labels = ["Accuracy", "Precision", "Recall", "F1"]
plt.figure(figsize=(6, 4))
sns.barplot(x=labels, y=metrics, palette="viridis")
plt.ylim(0, 1)
plt.title("Classification Metrics")
plt.savefig("metrics_bar.png", dpi=300)
plt.close()

# ========================
# 10. 保存模型
# ========================
torch.save(model.state_dict(), "LSTM_seq_best.pth")
print("训练完成，最佳模型已保存为 LSTM_seq_best.pth")
print("可视化已保存: loss_curve.png, confusion_matrix.png, metrics_bar.png")

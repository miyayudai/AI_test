import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# ⚙️ パラメータ定義部 (Parameters)
# ==========================================
INPUT_DIM      = 1      # 入力の次元数 (今回は1次元のスカラー値 x)
HIDDEN_DIM     = 16     # 中間層のニューロン数
OUTPUT_CLASSES = 3      # 出力のクラス数 (3つの確率の分布として出力)
LEARNING_RATE  = 0.05   # 学習率 (1歩あたりのパラメータ更新の大きさ)
EPOCHS         = 1000   # 学習の繰り返し回数
NUM_SAMPLES    = 200    # 学習に使用するデータの数

# 再現性のためのシード固定
torch.manual_seed(42)

# ==========================================
# 🧠 モデル定義部
# ==========================================
class SimpleNN(nn.Module):
    """
    入力 x を受け取り、OUTPUT_CLASSES 個の確率分布を出力するシンプルなニューラルネットワーク。
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, OUTPUT_CLASSES)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 最終出力をSoftmaxに通すことで、合計が1になる「確率分布」に変換する
        return torch.softmax(x, dim=1)

# ==========================================
# 🧠 理論の実装部 (学習プロセスの構築)
# ==========================================
# 1. 真の分布となるモデルX (Teacher) と近似させるモデルX' (Student) を準備
model_X = SimpleNN()
model_X_prime = SimpleNN()

# モデルXは「真の分布」として扱うため、重みは固定し学習させない
model_X.eval()
for param in model_X.parameters():
    param.requires_grad = False

# モデルX' を最適化するためのオプティマイザを設定
optimizer = optim.Adam(model_X_prime.parameters(), lr=LEARNING_RATE)

# 学習データの生成 (-5.0 から 5.0 までのランダムな入力 x)
x_data = torch.empty(NUM_SAMPLES, INPUT_DIM).uniform_(-5.0, 5.0)

# 損失関数の推移を記録するリスト
loss_history = []

print("学習を開始します...")
for epoch in range(EPOCHS):
    # モデルX' を学習モードにする
    model_X_prime.train()
    
    # 2. 両方のモデルに同じ入力 x を与え、確率分布 P と Q を取得
    P = model_X(x_data)       # 真の分布 P(x)
    Q = model_X_prime(x_data) # 近似分布 Q(x)
    
    # 3. KLダイバージェンスの計算 (理論の1行実装)
    # PyTorchの F.kl_div もありますが、理論の理解のため数式通りに直接実装します。
    # D_KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    # log(0)による無限大エラーを防ぐため、微小な値(1e-8)を足しています。
    kl_divergence = torch.sum(P * (torch.log(P + 1e-8) - torch.log(Q + 1e-8)), dim=1)
    
    # 全データにおけるKLダイバージェンスの平均値を最終的な損失(Loss)とする
    loss = torch.mean(kl_divergence)
    
    # 4. パラメータの更新 (バックプロパゲーション)
    optimizer.zero_grad() # 勾配をリセット
    loss.backward()       # 誤差逆伝播法により各パラメータの勾配を計算
    optimizer.step()      # 勾配に沿ってモデルX'の重みを更新
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss (KL Divergence): {loss.item():.4f}")

# ==========================================
# 📊 結果の実行とプロット
# ==========================================
# 学習後の結果を評価するため、滑らかな入力 x_test を作成
x_test = torch.linspace(-5.0, 5.0, 100).view(-1, 1)

with torch.no_grad():
    model_X_prime.eval()
    # P_test: 真の分布, Q_test: 学習後の近似分布, Q_init: 学習前の初期状態(比較用)
    P_test = model_X(x_test).numpy()
    Q_test = model_X_prime(x_test).numpy()
    
    # 学習前のモデルX' (初期シードを変えて一時的に作成)
    temp_model = SimpleNN()
    torch.manual_seed(123) 
    Q_init = temp_model(x_test).numpy()

x_test_np = x_test.numpy().flatten()

# プロットの作成
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. 損失(KL Divergence)の推移
axs[0].plot(loss_history, color='black')
axs[0].set_title('KL Divergence Loss Convergence')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss (D_KL)')
axs[0].grid(True)

# 2. 学習前の比較 (Teacher vs 完全にランダムな Student)
axs[1].set_title('Before Training: Model X vs Model X\'')
for c in range(OUTPUT_CLASSES):
    axs[1].plot(x_test_np, P_test[:, c], label=f'True X (Class {c})', linestyle='-', linewidth=2)
    axs[1].plot(x_test_np, Q_init[:, c], label=f'Init X\' (Class {c})', linestyle='--', alpha=0.6)
axs[1].set_xlabel('Input x')
axs[1].set_ylabel('Probability')
axs[1].legend(loc='upper right', fontsize=8)
axs[1].grid(True)

# 3. 学習後の比較 (Teacher vs 学習した Student)
axs[2].set_title('After Training: Model X vs Model X\'')
for c in range(OUTPUT_CLASSES):
    axs[2].plot(x_test_np, P_test[:, c], label=f'True X (Class {c})', linestyle='-', linewidth=2)
    axs[2].plot(x_test_np, Q_test[:, c], label=f'Trained X\' (Class {c})', linestyle='--', linewidth=2)
axs[2].set_xlabel('Input x')
axs[2].set_ylabel('Probability')
axs[2].legend(loc='upper right', fontsize=8)
axs[2].grid(True)

plt.tight_layout()
plt.show()
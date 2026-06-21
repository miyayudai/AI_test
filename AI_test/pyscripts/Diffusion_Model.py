import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# ⚙️ パラメータ定義部 (Parameters)
# ==========================================
NUM_STEPS   = 50      # T: 拡散（ノイズ追加・除去）のステップ数。減らすと高速化しますが精度が落ちます
BETA_START  = 1e-4    # ノイズスケジュールの開始値
BETA_END    = 0.02    # ノイズスケジュールの終了値
EPOCHS      = 1500    # 学習回数
NUM_SAMPLES = 1000    # データセットのサンプル数
LR          = 1e-3    # 学習率

torch.manual_seed(42)

# ==========================================
# 📊 データの準備 (多峰性を持つX字型の分布)
# ==========================================
# 状態 s は -1 から 1 の一様分布
s_train = torch.rand(NUM_SAMPLES, 1) * 2 - 1
# 行動 a は 50%の確率で s, 50%の確率で -s (少しノイズを乗せる)
a_train = torch.where(torch.rand(NUM_SAMPLES, 1) > 0.5, s_train, -s_train)
a_train += torch.randn_like(a_train) * 0.05 # 小さな観測ノイズ

# ==========================================
# 🧠 モデル定義部
# ==========================================
class StandardPolicy(nn.Module):
    """
    【Diffusionなし】
    状態 s から 行動 a を直接予測する従来のニューラルネットワーク（行動クローニング）
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, s):
        return self.net(s)

class DiffusionNoisePredictor(nn.Module):
    """
    【Diffusionあり】
    (行動 a_t, 状態 s, 時間 t) を受け取り、a_t に含まれる「ノイズ」を予測するネットワーク
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), # 入力は a, s, t の3次元
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, a, s, t):
        x = torch.cat([a, s, t], dim=1)
        return self.net(x)

# ==========================================
# 🧮 拡散モデルのスケジュール計算
# ==========================================
# ノイズの分散スケジュール β (線形に増加)
betas = torch.linspace(BETA_START, BETA_END, NUM_STEPS)
# α_t = 1 - β_t
alphas = 1.0 - betas
# α_bar_t = Π α_i (初期状態からの累積減衰)
alpha_bars = torch.cumprod(alphas, dim=0)

# ==========================================
# 🚀 学習プロセス
# ==========================================
model_standard = StandardPolicy()
model_diffusion = DiffusionNoisePredictor()

opt_std = optim.Adam(model_standard.parameters(), lr=LR)
opt_diff = optim.Adam(model_diffusion.parameters(), lr=LR)

print("学習を開始します...")
for epoch in range(EPOCHS):
    # -----------------------------------
    # 1. Standard Policy (Diffusionなし) の学習
    # -----------------------------------
    pred_a = model_standard(s_train)
    # 理論: 単純な平均二乗誤差 (MSE)
    loss_std = nn.MSELoss()(pred_a, a_train)
    
    opt_std.zero_grad()
    loss_std.backward()
    opt_std.step()

    # -----------------------------------
    # 2. Diffusion Policy (Diffusionあり) の学習
    # -----------------------------------
    # ランダムなステップ t をバッチサイズ分サンプリング
    t = torch.randint(0, NUM_STEPS, (NUM_SAMPLES, 1))
    
    # 該当する時刻の α_bar を取得
    a_bar_t = alpha_bars[t]
    
    # 標準正規分布からノイズ ε を生成
    noise = torch.randn_like(a_train)
    
    # 理論: a_t = √(α_bar_t) * a_0 + √(1 - α_bar_t) * ε (Forward Process: 一気にノイズを付加)
    a_t = torch.sqrt(a_bar_t) * a_train + torch.sqrt(1 - a_bar_t) * noise
    
    # ネットワークは付加されたノイズ ε を予測 (t は 0~1 に正規化して入力)
    pred_noise = model_diffusion(a_t, s_train, t.float() / NUM_STEPS)
    
    # 理論: 真のノイズと予測ノイズの平均二乗誤差 (Score Matching)
    loss_diff = nn.MSELoss()(pred_noise, noise)
    
    opt_diff.zero_grad()
    loss_diff.backward()
    opt_diff.step()

    if (epoch + 1) % 300 == 0:
        print(f"Epoch {epoch+1:4d} | Std Loss: {loss_std.item():.4f} | Diff Loss: {loss_diff.item():.4f}")

# ==========================================
# 🧪 推論 (サンプリング) プロセス
# ==========================================
with torch.no_grad():
    model_standard.eval()
    model_diffusion.eval()
    
    # テスト用の状態 s を生成 (-1 から 1 まで等間隔)
    s_test = torch.linspace(-1, 1, NUM_SAMPLES).view(-1, 1)
    
    # --- Diffusionなし の推論 ---
    a_pred_std = model_standard(s_test)
    
    # --- Diffusionあり の推論 (Reverse Process) ---
    # 1. 完全にランダムなノイズ a_T からスタート
    a_pred_diff = torch.randn_like(s_test)
    
    # 2. Tステップかけて徐々にノイズを除去していく
    for i in reversed(range(NUM_STEPS)):
        t_tensor = torch.full((NUM_SAMPLES, 1), i, dtype=torch.float32) / NUM_STEPS
        
        # ネットワークによるノイズ予測
        pred_noise = model_diffusion(a_pred_diff, s_test, t_tensor)
        
        alpha_t = alphas[i]
        alpha_bar_t = alpha_bars[i]
        beta_t = betas[i]
        
        # 最終ステップ以外は確率的な揺らぎ(z)を加える (Langevin Dynamics)
        z = torch.randn_like(s_test) if i > 0 else 0
        
        # 理論: a_{t-1} = 1/√(α_t) * (a_t - (1-α_t)/√(1-α_bar_t) * ε_θ) + σ_t * z
        a_pred_diff = (1.0 / torch.sqrt(alpha_t)) * (a_pred_diff - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * z

# ==========================================
# 📊 結果のプロット
# ==========================================
plt.figure(figsize=(15, 5))

# 1. 真のデータ分布
plt.subplot(1, 3, 1)
plt.scatter(s_train.numpy(), a_train.numpy(), alpha=0.3, s=5)
plt.title('Ground Truth (Bimodal Data)')
plt.xlabel('State (s)')
plt.ylabel('Action (a)')
plt.grid(True)

# 2. Diffusionなし (Standard BC)
plt.subplot(1, 3, 2)
plt.scatter(s_test.numpy(), a_pred_std.numpy(), color='red', s=5)
plt.title('Without Diffusion (Standard NN)')
plt.xlabel('State (s)')
plt.grid(True)

# 3. Diffusionあり (Diffusion Policy)
plt.subplot(1, 3, 3)
plt.scatter(s_test.numpy(), a_pred_diff.numpy(), color='green', s=5)
plt.title(f'With Diffusion Policy (T={NUM_STEPS})')
plt.xlabel('State (s)')
plt.grid(True)

plt.tight_layout()
plt.show()
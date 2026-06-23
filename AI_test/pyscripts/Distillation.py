import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ==========================================
# パラメータ定義部
# ==========================================
class Config:
    epochs = 500           # 学習の反復回数
    lr = 0.1               # 学習率
    
    # --- 蒸留特有のパラメータ ---
    temperature = 5.0      # 温度(T)。高いほど教師の「惜しい・微妙な判定」が強調されて生徒に伝わる
    alpha = 0.7            # 教師からの学びをどれくらい重視するか (1-alphaが正解ラベルからの学び)
    
    # --- モデルのサイズ差 ---
    teacher_hidden = 64    # 教師はパラメータが多く、複雑な境界線を引ける
    student_hidden = 4     # 生徒は極端にパラメータを減らし、表現力を制限

# ==========================================
# モデルの定義
# ==========================================
# 教師モデル：層が深く、ニューロンも多い
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, Config.teacher_hidden), nn.ReLU(),
            nn.Linear(Config.teacher_hidden, Config.teacher_hidden), nn.ReLU(),
            nn.Linear(Config.teacher_hidden, 3) # 3クラス分類
        )
    def forward(self, x):
        return self.net(x)

# 生徒モデル：層が浅く、ニューロンも極端に少ない
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, Config.student_hidden), nn.ReLU(),
            nn.Linear(Config.student_hidden, 3)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 理論：蒸留ロス関数の実装
# ==========================================
def distillation_loss(student_logits, teacher_logits, labels):
    T = Config.temperature
    alpha = Config.alpha
    
    # 1. 生徒の出力を温度Tで割り、対数ソフトマックスをとる（KLダイバージェンスの入力用）
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    
    # 2. 教師の出力を温度Tで割り、通常のソフトマックスをとる（これが抽出された「Dark Knowledge」）
    with torch.no_grad(): # 教師モデルは学習済みなので勾配計算を無効化
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        
    # 3. 生徒と教師の確率分布の差異（KLダイバージェンス）を計算し、温度の2乗をかけて勾配のスケールを戻す
    distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
    
    # 4. 生徒の元の出力と、実際の正解ラベルとの交差エントロピー誤差（Hard Target）を計算
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 5. 蒸留ロス(教師の真似)とハードロス(正解ラベルからの学習)を、重みalphaでブレンド
    return alpha * distill_loss + (1.0 - alpha) * hard_loss

# ==========================================
# 実行と結果の可視化
# ==========================================
if __name__ == "__main__":
    # 1. トイデータの生成（少し入り組んだ3クラスの2次元データ）
    X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, 
                               n_classes=3, n_clusters_per_class=1, class_sep=1.2, random_state=42)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 2. 教師モデルの学習（通常の交差エントロピーで正解ラベルのみから学習）
    print("教師モデルを学習中...")
    teacher = TeacherNet()
    optimizer_t = optim.Adam(teacher.parameters(), lr=Config.lr)
    for _ in range(Config.epochs):
        optimizer_t.zero_grad()
        loss = F.cross_entropy(teacher(X_tensor), y_tensor)
        loss.backward()
        optimizer_t.step()

    # 3. 生徒モデルの学習（比較のため、蒸留なしで正解ラベルのみから学習）
    print("生徒モデル(蒸留なし)を学習中...")
    student_base = StudentNet()
    optimizer_sb = optim.Adam(student_base.parameters(), lr=Config.lr)
    for _ in range(Config.epochs):
        optimizer_sb.zero_grad()
        loss = F.cross_entropy(student_base(X_tensor), y_tensor)
        loss.backward()
        optimizer_sb.step()

    # 4. 生徒モデルの学習（知識の蒸留を使用）
    print("生徒モデル(蒸留あり)を学習中...")
    student_distilled = StudentNet()
    optimizer_sd = optim.Adam(student_distilled.parameters(), lr=Config.lr)
    teacher.eval() # 教師は推論モードへ
    for _ in range(Config.epochs):
        optimizer_sd.zero_grad()
        # 生徒と教師の推論結果を出し、蒸留ロスで学習する
        s_logits = student_distilled(X_tensor)
        t_logits = teacher(X_tensor)
        loss = distillation_loss(s_logits, t_logits, y_tensor)
        loss.backward()
        optimizer_sd.step()

    # --- 境界線のプロット ---
    def plot_decision_boundary(ax, model, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        with torch.no_grad():
            preds = torch.argmax(model(grid), dim=1).numpy().reshape(xx.shape)
        ax.contourf(xx, yy, preds, alpha=0.3, cmap='brg')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='brg', s=20)
        ax.set_title(title)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_decision_boundary(axes[0], teacher, "1. Teacher (Large)")
    plot_decision_boundary(axes[1], student_base, "2. Student Base (Small, No Distillation)")
    plot_decision_boundary(axes[2], student_distilled, "3. Student Distilled (Small, w/ Distillation)")
    plt.tight_layout()
    plt.show()
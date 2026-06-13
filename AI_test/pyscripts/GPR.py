import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ パラメータ定義部 (Parameters)
# 学習・実験のために数値を変更してみてください
# ==========================================
# RBF(Radial Basis Function)カーネルのパラメータ
LENGTH_SCALE = 1.0  # l: 入力値の差がどれくらい離れると相関が薄れるか（波の滑らかさ）
VARIANCE_F   = 1.0  # sigma_f^2: 関数の出力の振幅（縦方向のスケール）

# 観測データに関するパラメータ
NOISE_VAR    = 0.05 # sigma_y^2: 観測データに含まれるノイズの分散（データのばらつき）
N_TRAIN      = 10   # 訓練データの数


# ==========================================
# 🧠 理論の実装部
# ==========================================
def rbf_kernel(X1, X2, l=LENGTH_SCALE, sigma_f=VARIANCE_F):
    """
    RBFカーネル（Squared Exponential Kernel）を計算する関数。
    2つの入力データの類似度を測ります。近いデータほど値が大きくなります。
    """
    # X1とX2の各要素間のユークリッド距離の二乗を計算 (行列のブロードキャストを使用)
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    # 理論式: k(x, x') = sigma_f^2 * exp(-0.5 * (距離 / l)^2) に当てはめる
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def gaussian_process_regression(X_train, y_train, X_test):
    """
    訓練データを用いて、テストデータに対する事後分布（平均と分散）を計算します。
    """
    # 1. 訓練データ同士の共分散行列 K を計算
    K = rbf_kernel(X_train, X_train)
    
    # 2. 観測データのノイズ分散をKの対角成分に足し込む（K = K + sigma_y^2 * I）
    # これにより、観測データ自体も完全に正確ではなくノイズを含んでいることをモデル化します
    K_noisy = K + NOISE_VAR * np.eye(len(X_train))
    
    # 3. 訓練データとテストデータ間の共分散行列 K_* を計算
    K_s = rbf_kernel(X_train, X_test)
    
    # 4. テストデータ同士の共分散行列 K_** を計算
    K_ss = rbf_kernel(X_test, X_test)
    
    # 5. K_noisyの逆行列を計算 K^-1
    # ※実用上はコレスキー分解等で数値的に安定させますが、理論への忠実さのためそのまま逆行列をとります
    K_noisy_inv = np.linalg.inv(K_noisy)
    
    # 6. 事後分布の平均を計算: mu_* = K_*^T * K^-1 * y
    # これがテスト点における「最も確からしい予測値」の曲線になります
    mu_s = K_s.T.dot(K_noisy_inv).dot(y_train)
    
    # 7. 事後分布の共分散行列を計算: Sigma_* = K_** - K_*^T * K^-1 * K_*
    # 右辺第2項は「観測データから得られた情報によって減った不確実性」を意味します
    cov_s = K_ss - K_s.T.dot(K_noisy_inv).dot(K_s)
    
    # 8. 共分散行列の対角成分のみを抽出し、各テスト点での「分散（不確実性の大きさ）」とする
    var_s = np.diag(cov_s)
    
    return mu_s.flatten(), var_s

# ==========================================
# 📊 データの準備と実行・プロット
# ==========================================
np.random.seed(42)

# 訓練データ (ランダムに N_TRAIN 個の点を選択し、サイン波 + ノイズを生成)
X_train = np.random.uniform(-5, 5, N_TRAIN).reshape(-1, 1)
y_train = np.sin(X_train) + np.random.normal(0, np.sqrt(NOISE_VAR), size=(N_TRAIN, 1))

# テストデータ (滑らかな曲線を描画するために細かく設定)
X_test = np.linspace(-6, 6, 100).reshape(-1, 1)

# ガウス過程回帰の実行
mu, var = gaussian_process_regression(X_train, y_train, X_test)
std = np.sqrt(var) # 標準偏差（ばらつき）

# 結果のプロット
plt.figure(figsize=(10, 6))
plt.plot(X_test, np.sin(X_test), 'k--', label='True function (sin(x))')  # 真の関数
plt.plot(X_train, y_train, 'ro', markersize=8, label='Training Data (Observations)') # 観測データ
plt.plot(X_test, mu, 'b-', lw=2, label='GPR Predictive Mean')            # 予測平均
# 予測の不確実性 (95%信頼区間: 平均 ± 2標準偏差) を塗りつぶす
plt.fill_between(X_test.flatten(), mu - 2*std, mu + 2*std, color='blue', alpha=0.2, label='95% Confidence Interval')

plt.title('Gaussian Process Regression from Scratch')
plt.xlabel('X')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
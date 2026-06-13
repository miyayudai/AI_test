import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import rbf_kernel

# データ生成（非線形な2クラス）
np.random.seed(0)
n = 100
X = np.random.randn(n, 2)
y = (X[:, 0]**2/4 + X[:, 1]**2 > 1).astype(int) * 2 - 1  # -1 or +1

# --- 1. 線形 SVM ---
clf_linear = SVC(kernel='linear').fit(X, y)

# --- 2. RBF カーネル SVM ---
clf_rbf = SVC(kernel='rbf', gamma=1).fit(X, y)

# --- 3. 無限幅ネットの近似（NTK っぽい挙動） ---
# 小さな学習率でほぼ線形化されたネットを模倣
mlp = MLPRegressor(hidden_layer_sizes=(500,), activation='tanh',
                   max_iter=1, warm_start=True)

# 初期化後の勾配特徴量を使って NTK を近似
mlp.fit(X, y)
G = mlp.predict(X).reshape(-1, 1)
K_ntk = G @ G.T  # NTK の超簡易版（本物ではないが直感をつかむため）

# NTK でカーネル SVM
clf_ntk = SVC(kernel='precomputed').fit(K_ntk, y)

# --- 可視化 ---
def plot_model(clf, title, kernel=None):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200),
                         np.linspace(-3, 3, 200))
    Xgrid = np.c_[xx.ravel(), yy.ravel()]

    if kernel == "precomputed":
        K = rbf_kernel(Xgrid, X)  # ここは簡易的に RBF を使う
        Z = clf.decision_function(K)
    else:
        Z = clf.decision_function(Xgrid)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z > 0, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.title(title)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plot_model(clf_linear, "Linear SVM")

plt.subplot(1, 3, 2)
plot_model(clf_rbf, "RBF Kernel SVM")

plt.subplot(1, 3, 3)
plot_model(clf_ntk, "NTK-like Kernel SVM", kernel="precomputed")

plt.show()

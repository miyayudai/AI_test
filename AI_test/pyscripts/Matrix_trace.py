import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ パラメータ定義部 (Parameters)
# 行列の値を変更して、空間の歪み方とトレースの関係を実験できます
# ==========================================
# 2次元の変換行列 A
# 対角成分が 2.0 と 1.5 なので、計算上のトレースは 3.5 になります
MATRIX_A = np.array([
    [2.0, 1.0],
    [0.5, 1.5]
])

# 描画用のパラメータ
NUM_POINTS = 100 # 円を描くための頂点数

# ==========================================
# 🧠 理論の実装と計算部
# ==========================================
def calculate_trace_and_eigenvalues(matrix):
    """
    行列のトレースと固有値を計算し、両者の関係を検証します。
    """
    # 1. トレースの計算: 対角成分の和 (理論式: Tr(A) = Σ a_ii)
    trace_val = np.trace(matrix)
    
    # 2. 固有値の計算: 空間がどの方向にどれくらい伸びるか
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # 3. 固有値の和の計算 (理論式: Σ λ_i)
    sum_of_eigenvalues = np.sum(eigenvalues)
    
    return trace_val, eigenvalues, eigenvectors, sum_of_eigenvalues

def apply_transformation(matrix, num_points):
    """
    単位円を行列で変換し、空間がどう歪むか（図式化用データ）を計算します。
    """
    # 角度 theta を 0 から 2π まで生成
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # 単位円上の点 (x, y) を作成（2行N列の行列にする）
    circle_points = np.vstack((np.cos(theta), np.sin(theta)))
    
    # 行列 A を掛けて、円を楕円に変換する (理論式: x' = Ax)
    transformed_points = matrix @ circle_points
    
    return circle_points, transformed_points

# ==========================================
# 📊 データの準備と実行・プロット
# ==========================================
# 理論値の計算
trace, eig_vals, eig_vecs, eig_sum = calculate_trace_and_eigenvalues(MATRIX_A)

print(f"✅ 行列の対角成分の和 (Trace): {trace:.3f}")
print(f"✅ 固有値の和 (Sum of Eigenvalues): {eig_sum:.3f}")
# これらが完全に一致することが、トレースの最も美しい性質の一つです

# 変換前後の座標データを取得
circle, ellipse = apply_transformation(MATRIX_A, NUM_POINTS)

# プロットの作成
plt.figure(figsize=(8, 8))

# 変換前の単位円（青）
plt.plot(circle[0, :], circle[1, :], 'b--', label='Original Space (Unit Circle)')

# 変換後の楕円（赤）
plt.plot(ellipse[0, :], ellipse[1, :], 'r-', linewidth=2, label='Transformed Space (Ellipse)')

# 固有ベクトルの描画（空間の主軸の伸びを示す）
colors = ['g', 'm']
for i in range(len(eig_vals)):
    # 固有ベクトルの方向に、固有値の大きさだけ矢印を伸ばす
    vec = eig_vecs[:, i] * eig_vals[i]
    plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, 
               color=colors[i], width=0.008, 
               label=f'Eigenvector {i+1} (stretch: $\lambda_{i+1}$={eig_vals[i]:.2f})')

plt.title(f"Matrix Transformation\nTrace = {trace:.2f} (Sum of Eigenvalues $\lambda_1 + \lambda_2$)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.7)
plt.axis('equal') # 縦横比を同じにする
plt.legend(loc='upper left')
plt.show()
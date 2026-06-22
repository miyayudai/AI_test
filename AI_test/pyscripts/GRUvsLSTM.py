import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math

# ==========================================
# パラメータ定義部
# ==========================================
class Config:
    input_size = 1         # センサー入力の次元 (例: 1つの関節角度)
    hidden_size = 16       # 記憶領域のサイズ (パラメータ数に直結)
    seq_length = 20        # オフライン学習で一度に見る過去のステップ数
    
    offline_epochs = 100   # オフライン学習（事前学習）の反復回数
    online_steps = 150     # オンライン学習（リアルタイム適応）のステップ数
    
    lr = 0.02              # 学習率

# ==========================================
# 1. GRUセルの理論実装（軽量）
# ==========================================
class TheoreticalGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 更新ゲート(z)とリセットゲート(r)用の重み
        self.linear_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_r = nn.Linear(input_size + hidden_size, hidden_size)
        # 新しい候補記憶(n)用の重み
        self.linear_n = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # x: 現在の入力, h_prev: 1ステップ前の隠れ状態(記憶)
        
        # 1. 入力と過去の記憶を結合
        combined = torch.cat([x, h_prev], dim=1)
        
        # 2. 更新ゲート (Update Gate) : 過去の記憶をどれくらい保持するか (0〜1)
        z_t = torch.sigmoid(self.linear_z(combined))
        
        # 3. リセットゲート (Reset Gate) : 過去の記憶をどれくらい忘れて新しい情報を探すか (0〜1)
        r_t = torch.sigmoid(self.linear_r(combined))
        
        # 4. リセットゲートを適用した過去の記憶と、現在の入力を結合
        combined_reset = torch.cat([x, r_t * h_prev], dim=1)
        
        # 5. 新しい記憶の候補 (New Memory Candidate) を計算 (-1〜1)
        n_t = torch.tanh(self.linear_n(combined_reset))
        
        # 6. 新しい隠れ状態 : 更新ゲートz_tの比率に従って、過去の記憶と新しい候補をブレンドする
        h_t = (1 - z_t) * n_t + z_t * h_prev
        
        return h_t

# ==========================================
# 2. LSTMセルの理論実装（重厚）
# ==========================================
class TheoreticalLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 3つのゲートと候補記憶のための重み (GRUより多い)
        self.linear_f = nn.Linear(input_size + hidden_size, hidden_size) # 忘却
        self.linear_i = nn.Linear(input_size + hidden_size, hidden_size) # 入力
        self.linear_o = nn.Linear(input_size + hidden_size, hidden_size) # 出力
        self.linear_c = nn.Linear(input_size + hidden_size, hidden_size) # セル候補

    def forward(self, x, h_prev, c_prev):
        # x: 入力, h_prev: 過去の隠れ状態, c_prev: 過去のセル状態(長期記憶)
        combined = torch.cat([x, h_prev], dim=1)
        
        # 1. 忘却ゲート (Forget Gate) : 長期記憶から何を忘れるか (0〜1)
        f_t = torch.sigmoid(self.linear_f(combined))
        
        # 2. 入力ゲート (Input Gate) : 新しい情報のどこを長期記憶に書き込むか (0〜1)
        i_t = torch.sigmoid(self.linear_i(combined))
        
        # 3. 新しいセル状態の候補 (-1〜1)
        c_tilde = torch.tanh(self.linear_c(combined))
        
        # 4. 新しい長期記憶 (Cell State) : 忘却分を引き継ぎ、新しい情報を足す
        c_t = f_t * c_prev + i_t * c_tilde
        
        # 5. 出力ゲート (Output Gate) : 長期記憶のどこを隠れ状態として外に出すか (0〜1)
        o_t = torch.sigmoid(self.linear_o(combined))
        
        # 6. 新しい隠れ状態 (Hidden State) : 出力ゲートと長期記憶の掛け合わせ
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

# ==========================================
# ロボット制御用モデルラッパー
# 最終的な予測（次のステップの関節角度など）を出力する
# ==========================================
class RobotController(nn.Module):
    def __init__(self, rnn_cell, cell_type):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.cell_type = cell_type
        # 隠れ状態から最終的な1次元の予測値(関節角度)に変換
        self.fc = nn.Linear(Config.hidden_size, 1)

    def forward_step(self, x, states):
        # 1ステップだけ進める（オンライン学習・推論用）
        if self.cell_type == 'GRU':
            h = self.rnn_cell(x, states[0])
            out = self.fc(h)
            return out, (h,)
        else: # LSTM
            h, c = self.rnn_cell(x, states[0], states[1])
            out = self.fc(h)
            return out, (h, c)

# ==========================================
# 実験コード：オフライン学習とオンライン学習
# ==========================================
def generate_data(steps, shift=False):
    # ロボットの関節角度の理想軌道（サイン波）を生成
    t = torch.linspace(0, 4 * math.pi, steps).view(-1, 1)
    if shift:
        # オンライン学習用に、環境変化（負荷増大などで波形がバグった状態）をシミュレート
        y = torch.sin(t * 1.5) + 0.5 
    else:
        y = torch.sin(t)
    return y

def run_experiment(cell_class, cell_type):
    # モデルとオプティマイザの初期化
    model = RobotController(cell_class(Config.input_size, Config.hidden_size), cell_type)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.MSELoss()
    
    # -----------------------------
    # 1. オフライン学習 (バッチ処理)
    # 工場出荷前に基本の軌道を覚えさせるフェーズ
    # -----------------------------
    offline_data = generate_data(200)
    
    start_time = time.time()
    for epoch in range(Config.offline_epochs):
        optimizer.zero_grad()
        loss = 0
        
        # 状態の初期化 (ゼロベクトル)
        h = torch.zeros(1, Config.hidden_size)
        states = (h,) if cell_type == 'GRU' else (h, torch.zeros(1, Config.hidden_size))
        
        # 時系列を順番に処理し、次のステップの値を予測
        for t in range(len(offline_data) - 1):
            x_t = offline_data[t].unsqueeze(0)
            target = offline_data[t+1].unsqueeze(0)
            
            pred, states = model.forward_step(x_t, states)
            loss += criterion(pred, target)
            
        loss.backward()
        optimizer.step()
        
    offline_time = time.time() - start_time
    
    # -----------------------------
    # 2. オンライン学習 (ステップ処理)
    # 現場でロボットを動かしながら、1ステップごとに学習・適応させるフェーズ
    # -----------------------------
    online_data = generate_data(Config.online_steps, shift=True) # 環境が変化した！
    predictions = []
    
    h = torch.zeros(1, Config.hidden_size)
    states = (h,) if cell_type == 'GRU' else (h, torch.zeros(1, Config.hidden_size))
    
    start_time = time.time()
    for t in range(len(online_data) - 1):
        x_t = online_data[t].unsqueeze(0)
        target = online_data[t+1].unsqueeze(0)
        
        # 推論 (動かす)
        pred, states = model.forward_step(x_t, states)
        predictions.append(pred.detach().item())
        
        # オンライン学習: 1ステップの誤差ですぐに重みを更新 (適応する)
        optimizer.zero_grad()
        loss = criterion(pred, target)
        loss.backward(retain_graph=True) # RNNのオンライン学習特有の処置
        optimizer.step()
        
        # 次のステップのために計算グラフから状態を切り離す (BPTTの切断)
        if cell_type == 'GRU':
            states = (states[0].detach(),)
        else:
            states = (states[0].detach(), states[1].detach())

    online_time = time.time() - start_time
    
    return online_data[1:].numpy(), predictions, offline_time, online_time

if __name__ == "__main__":
    print("LSTMとGRUの比較実験を開始します...")
    
    # GRUの実行
    targets, preds_gru, off_t_gru, on_t_gru = run_experiment(TheoreticalGRUCell, 'GRU')
    
    # LSTMの実行
    _, preds_lstm, off_t_lstm, on_t_lstm = run_experiment(TheoreticalLSTMCell, 'LSTM')
    
    # 結果の表示と可視化
    print("\n=== 計算時間の比較 ===")
    print(f"【GRU】 オフライン: {off_t_gru:.3f}秒 | オンライン: {on_t_gru:.3f}秒")
    print(f"【LSTM】オフライン: {off_t_lstm:.3f}秒 | オンライン: {on_t_lstm:.3f}秒")
    print(f"-> GRUはLSTMに比べ、ゲート計算が少ないため処理が高速です。")
    
    plt.figure(figsize=(10, 5))
    plt.title("Online Learning Adaptation (Target wave changed!)")
    plt.plot(targets, label="True Trajectory (Changed)", color='black', linestyle='--')
    plt.plot(preds_gru, label="GRU Prediction", color='blue', alpha=0.7)
    plt.plot(preds_lstm, label="LSTM Prediction", color='red', alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("Joint Angle")
    plt.legend()
    plt.grid()
    plt.show()
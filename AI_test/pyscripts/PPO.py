import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
import random

# グリッドワールド環境の定義
class GridWorld:
    def __init__(self, 
                 size: int = 5 # グリッドワールドのサイズ。一辺の長さ。
                 ) -> None:
        """
        5x5のグリッドワールドを初期化します。        
        """
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.reset()

    def reset(self) -> Tuple[int, int]: #エージェントの初期位置を返す
        """
        環境をリセットし、エージェントの位置を開始地点に戻します。
        """
        self.agent_pos = self.start_pos # エージェントの位置を開始地点に設定
        return self.agent_pos

    def step(self, 
             action: int # 移動アクション（0: 上, 1: 右, 2: 下, 3: 左）
             ) -> Tuple[Tuple[int, int], float, bool]: # 次の位置, 報酬, ゴール到達フラグ
        """
        エージェントを指定されたアクションに従って移動させます。
        """
        x, y = self.agent_pos # 現在の位置を取得
        if action == 0 and x > 0: # 上に移動
            x -= 1
        elif action == 1 and y < self.size - 1: # 右に移動
            y += 1
        elif action == 2 and x < self.size - 1: # 下に移動
            x += 1
        elif action == 3 and y > 0: # 左に移動
            y -= 1
        # 移動後の位置を更新
        self.agent_pos = (x, y)
        # ゴールに到達したかどうかを判定
        done = self.agent_pos == self.goal_pos
        # 報酬の設定：ゴールに到達したら1、それ以外は0、と今回は簡易的にスパース報酬を設定する。（欠点：学習が不安定にある恐れがある）
        # より滑らかな報酬設計が必要な場合は、例えばゴールに近づくほど報酬が増加するように設計する。（効果：学習が安定する）
        reward = 1.0 if done else 0.0
        return self.agent_pos, reward, done

    def get_state(self) -> np.ndarray: # 現在のエージェントの位置をワンホットベクトル状態として返す
        """
        現在のエージェントの位置を状態として返します。
        """
        state = np.zeros((self.size, self.size)) # 5x5のゼロ行列を作成
        state[self.agent_pos] = 1 # エージェントの位置を1に設定 
        return state.flatten() # 5x5のグリッドワールドを1次元のベクトルに変換して返す

# ニューラルネットワークの定義
class PolicyNetwork(nn.Module):
    def __init__(self, 
                 state_size: int, # 状態の次元数. 例: 5x5で25
                 action_size: int # アクションの数. 例: 上, 右, 下, 左の4つ
                 ) -> None:
        """
        ポリシーネットワークを初期化します。
        """
        super(PolicyNetwork, self).__init__()   # 親クラスの初期化
        self.fc1 = nn.Linear(state_size, 128)   # 入力層
        self.fc2 = nn.Linear(128, 128)          # 隠れ層
        self.action_head = nn.Linear(128, action_size) # アクションの確率分布を出力する層
        self.value_head = nn.Linear(128, 1)     # 状態価値を出力する層

    def forward(self, 
                x: torch.Tensor # 入力状態. 例: 5x5のグリッドワールドの状態. 例: [0, 0, 1, 0, 0, ...]
                ) -> Tuple[torch.Tensor, torch.Tensor]: # アクションの確率分布, 状態価値. 例: ([0.1, 0.2, 0.3, 0.4], 0.5)
        """
        フォワードパスを実行します。
        """
        x = torch.relu(self.fc1(x)) # 入力層から隠れ層への活性化関数
        x = torch.relu(self.fc2(x)) # 隠れ層から隠れ層への活性化関数
        action_probs = torch.softmax(self.action_head(x), dim=-1) # アクションの確率分布を計算
        state_values = self.value_head(x) # 状態価値を計算
        return action_probs, state_values 

# PPOエージェントの定義
class PPOAgent:
    def __init__(self,
                state_size: int,        # 状態の次元数
                action_size: int,       # アクションの数
                lr: float = 1e-3,       # 学習率
                gamma: float = 0.99,    # 割引率
                eps_clip: float = 0.2,  # クリッピング係数
                K_epochs: int = 4       # ポリシーの更新回数
                ) -> None:
        """
        PPOエージェントを初期化します。
        """
        self.gamma = gamma       # 割引率
        self.eps_clip = eps_clip # クリッピング係数
        self.K_epochs = K_epochs # ポリシーの更新回数

        self.policy = PolicyNetwork(state_size, action_size)        # ポリシーネットワークの初期化
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)# オプティマイザの初期化
        self.policy_old = PolicyNetwork(state_size, action_size)    # 古いポリシーネットワークの初期化
        self.policy_old.load_state_dict(self.policy.state_dict())   # 古いポリシーネットワークのパラメータをコピー

        self.MseLoss = nn.MSELoss() # 平均二乗誤差

    def select_action(self, 
                      state: np.ndarray # 現在の状態
                      ) -> Tuple[int, torch.Tensor, torch.Tensor]: # 選択したアクション, アクションの確率, 状態価値
        """
        現在のポリシーに基づいてアクションを選択します。
        """
        state = torch.FloatTensor(state) # 状態をテンソルに変換. 例: [0, 0, 1, 0, 0, ...]
        with torch.no_grad(): # 勾配を計算しない
            action_probs, state_value = self.policy_old(state) # 古いポリシーでアクションの確率と状態価値を取得
        action_dist = torch.distributions.Categorical(action_probs) # カテゴリカル分布を作成. 例: [0.1, 0.2, 0.3, 0.4]
        action = action_dist.sample() # アクションをサンプリング. 例: 2
        return action.item(), action_probs[action.item()], state_value # 例: 2, [0.1, 0.2, 0.3, 0.4], 0.5

    def update(self,
               memory: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] # エピソード中の経験のリスト
               ) -> None:
        """
        PPOのアップデートステップを実行します。
        """
        # 状態、アクション、報酬、次状態、終了フラグを分解
        states = torch.FloatTensor([m[0] for m in memory]) # 状態. 例: [[0, 0, 1, 0, 0, ...], [0, 0, 1, 0, 0, ...], ...]
        actions = torch.LongTensor([m[1] for m in memory]).unsqueeze(1) # アクション. 例: [2, 1, ...]
        rewards = [m[2] for m in memory] # 報酬. 例: [0.0, 0.0, ...]
        next_states = torch.FloatTensor([m[3] for m in memory]) # 次状態. 例: [[0, 0, 1, 0, 0, ...], [0, 0, 1, 0, 0, ...], ...]
        dones = torch.FloatTensor([m[4] for m in memory]) # 終了フラグ. 例: [False, True, ...]

        # 割引累積報酬の計算
        discounted_rewards = [] # 割引累積報酬
        R = 0 # 累積報酬
        for reward, done in zip(reversed(rewards), reversed(dones)): # エピソードの最後から計算
            if done: # エピソードが終了したら累積報酬をリセット
                R = 0
            R = reward + self.gamma * R # 割引累積報酬の計算
            discounted_rewards.insert(0, R) # 割引累積報酬を挿入
        discounted_rewards = torch.FloatTensor(discounted_rewards).unsqueeze(1) # 割引累積報酬. 例: [[0.0], [0.0], ...]

        # 状態価値の計算
        _, state_values = self.policy(states) # 例: [0.5, 0.6, ...]
        advantages = discounted_rewards - state_values.detach() # アドバンテージ. 例: [[0.0], [0.0], ...]

        # 古いポリシーでアクションの確率を取得
        action_probs, _ = self.policy_old(states) # 例: [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], ...]
        action_dist = torch.distributions.Categorical(action_probs) # カテゴリカル分布を作成. 例: [0.1, 0.2, 0.3, 0.4]
        old_log_probs = action_dist.log_prob(actions.squeeze()).detach() # 古いポリシーでのアクションの対数確率. 例: [0.2, 0.3, ...]

        # PPOの更新
        for _ in range(self.K_epochs): # ポリシーの更新回数分繰り返す
            # 新しいポリシーでアクションの確率を取得
            new_action_probs, new_state_values = self.policy(states) # 例: [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], ...]
            new_dist = torch.distributions.Categorical(new_action_probs) # カテゴリカル分布を作成. 例: [0.1, 0.2, 0.3, 0.4]
            new_log_probs = new_dist.log_prob(actions.squeeze()) # 新しいポリシーでのアクションの対数確率. 例: [0.1, 0.2, ...]

            # クリップされた比率の計算
            ratios = torch.exp(new_log_probs - old_log_probs) # 比率の計算. 例: [1.0, 1.1, ...]
            surr1 = ratios * advantages # クリップされた比率1. 例: [[0.0], [0.0], ...]
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages # クリップされた比率2. 例: [[0.0], [0.0], ...]

            # 損失の計算
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(new_state_values, discounted_rewards) # 損失の計算. 例: [[0.0], [0.0], ...]

            # 勾配の更新
            self.optimizer.zero_grad() # 勾配をリセット. これがないと勾配が加算される
            loss.mean().backward() # 損失を逆伝播.
            self.optimizer.step() # パラメータを更新.

        # 古いポリシーを新しいポリシーで更新
        self.policy_old.load_state_dict(self.policy.state_dict()) # パラメータをコピー. これがないと古いポリシーが更新される

def main() -> None:
    """
    メイン関数。環境とエージェントを初期化し、訓練を実行します。
    """
    env = GridWorld(size=5) # グリッドワールド環境の初期化. 5x5のグリッドワールド
    state_size = env.size * env.size # 状態の次元数. 例: 5x5で25
    action_size = 4  # 上, 右, 下, 左 の４つ
    agent = PPOAgent(state_size, action_size) # PPOエージェントの初期化

    max_episodes = 1000 # エピソード数
    max_steps = 100 # 最大ステップ数. ゴールに到達できない場合は強制終了. 
    log_interval = 100 # ログ表示のインターバル. 例: 100エピソードごとに進捗を表示. 

    for episode in range(1, max_episodes + 1): # エピソードの繰り返し 
        state = env.reset() # 環境をリセット. エージェントの初期位置を取得
        memory: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = [] # 経験のリスト. 例: [([0, 0, 1, 0, 0, ...], 2, 0.0, [0, 0, 1, 0, 0, ...], False), ...]
        for _ in range(max_steps): # ステップ数の繰り返し
            action, action_prob, state_value = agent.select_action(env.get_state()) # アクションを選択. 例: 2, [0.1, 0.2, 0.3, 0.4], 0.5
            next_state, reward, done = env.step(action) # 環境を進める. 例: [0, 0, 1, 0, 0, ...], 0.0, False
            memory.append((env.get_state(), action, reward, next_state, done)) # 経験を追加. 例: [([0, 0, 1, 0, 0, ...], 2, 0.0, [0, 0, 1, 0, 0, ...], False), ...]
            if done: # ゴールに到達したら終了 
                break 
            state = next_state # 状態を更新. 例: [0, 0, 1, 0, 0, ...]
        # PPOの更新
        agent.update(memory) 
        # 定期的に進捗を表示
        if episode % log_interval == 0:
            print(f"Episode {episode} completed.")

    # 訓練後のエージェントの動作確認
    state = env.reset()
    env_steps = 0
    print("訓練後のエージェントの動作:")
    while True: # ゴールに到達するか、最大ステップ数に達するまで繰り返す
        action, _, _ = agent.select_action(env.get_state()) # アクションを選択
        next_state, reward, done = env.step(action) # 環境を進める
        print(f"Step {env_steps}: Moved to {next_state}") # 状態を表示
        env_steps += 1 # ステップ数を更新
        if done or env_steps >= max_steps: # ゴールに到達したら終了
            break
    if done:
        print("ゴールに到達しました！")
    else:
        print("ゴールに到達できませんでした。")

if __name__ == "__main__":
    main()


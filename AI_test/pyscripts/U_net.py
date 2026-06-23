import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# パラメータ定義部
# 実験のための設定をここに集約します
# ==========================================
class Config:
    batch_size = 1        # 一度に処理する画像の枚数
    in_channels = 1       # 入力画像のチャネル数 (1=モノクロ, 3=カラー)
    out_channels = 1      # 出力画像のチャネル数 (Diffusionでは入力と同じ)
    img_size = 128         # 画像のサイズ (縦横64ピクセル)
    base_features = 16    # 最初の層で抽出する特徴量の数 (ネットワークの太さ)

# ==========================================
# 1. 2回の畳み込みを行うブロック (DoubleConv)
# 各階層で特徴を抽出するための基本部品です。
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 畳み込み -> バッチ正規化 -> ReLU(活性化関数) を2回繰り返すモジュールを定義
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), # 3x3フィルタで特徴抽出。padding=1でサイズを保つ
            nn.BatchNorm2d(out_ch),                             # 学習を安定させるための正規化
            nn.ReLU(inplace=True),                              # 負の値を0にする非線形変換
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),# 2回目の畳み込みでさらに複雑な特徴を抽出
            nn.BatchNorm2d(out_ch),                             # 再び正規化
            nn.ReLU(inplace=True)                               # 再び非線形変換
        )

    def forward(self, x):
        return self.conv(x) # 定義した処理を順番に実行して返す

# ==========================================
# 2. U-Net本体の定義
# エンコーダ(縮小)とデコーダ(拡大)、そしてスキップ接続を実装します。
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self, in_ch, out_ch, features):
        super().__init__()
        
        # --- エンコーダ部 (Down) ---
        # 画像サイズを半分にしながら、チャネル数(特徴の数)を倍にしていきます。
        self.enc1 = DoubleConv(in_ch, features)                  # [64x64, features]
        self.enc2 = DoubleConv(features, features * 2)           # [32x32, features*2]
        self.enc3 = DoubleConv(features * 2, features * 4)       # [16x16, features*4]
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        # 画像の縦横サイズを半分にするプーリング層
        
        # --- ボトルネック部 (Bottom) ---
        # 最も画像サイズが小さく、抽象的な意味情報が凝縮された部分です。
        self.bottleneck = DoubleConv(features * 4, features * 8) # [8x8, features*8]
        
        # --- デコーダ部 (Up) ---
        # 画像サイズを倍にしながら、エンコーダからのスキップ接続を受け取ります。
        # up1: 特徴量を半分に減らしながらサイズを倍にする(転置畳み込み)
        self.up1 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        # スキップ接続でエンコーダの特徴量と結合するため、入力チャネル数は(features*4 + features*4)になる
        self.dec1 = DoubleConv(features * 8, features * 4)       

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2)       

        self.up3 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features * 2, features)           
        
        # --- 出力層 ---
        # 最終的なチャネル数(out_ch)に変換します。サイズは変わりません。
        self.out_conv = nn.Conv2d(features, out_ch, kernel_size=1)

    def forward(self, x):
        # エンコーダの処理とスキップ接続用のデータ保存
        e1 = self.enc1(x)                # 最初の畳み込み。e1はスキップ接続用に保存
        p1 = self.pool(e1)               # サイズを半分に
        
        e2 = self.enc2(p1)               # 2層目の畳み込み。e2を保存
        p2 = self.pool(e2)               # サイズを半分に
        
        e3 = self.enc3(p2)               # 3層目の畳み込み。e3を保存
        p3 = self.pool(e3)               # サイズを半分に
        
        # ボトルネックの処理
        b = self.bottleneck(p3)          # 最深部での特徴抽出
        
        # デコーダの処理とスキップ接続(Skip Connection)の結合
        u1 = self.up1(b)                 # サイズを倍に拡大 (8x8 -> 16x16)
        # ★スキップ接続: 拡大したテンソル(u1)と、エンコーダの同じ階層のテンソル(e3)をチャネル方向(dim=1)で結合
        cat1 = torch.cat([u1, e3], dim=1)
        d1 = self.dec1(cat1)             # 結合したデータから特徴を抽出・整理
        
        u2 = self.up2(d1)                # サイズを倍に拡大 (16x16 -> 32x32)
        cat2 = torch.cat([u2, e2], dim=1) # スキップ接続でe2と結合
        d2 = self.dec2(cat2)             # 特徴を整理
        
        u3 = self.up3(d2)                # サイズを倍に拡大 (32x32 -> 64x64)
        cat3 = torch.cat([u3, e1], dim=1) # スキップ接続でe1と結合
        d3 = self.dec3(cat3)             # 特徴を整理
        
        # 出力層
        out = self.out_conv(d3)          # 目的のチャネル数に変換して出力
        return out

# ==========================================
# 実行と結果の可視化
# ==========================================
if __name__ == "__main__":
    # モデルのインスタンス化
    model = SimpleUNet(Config.in_channels, Config.out_channels, Config.base_features)
    
    # Diffusion Modelを想定し、ランダムなノイズ画像を生成 (バッチ, チャネル, 縦, 横)
    # 実際のAIはこれを「ノイズが乗った画像」として受け取ります
    dummy_input = torch.randn(Config.batch_size, Config.in_channels, Config.img_size, Config.img_size)
    
    # モデルにデータを入力し、出力を得る (勾配計算は不要なので no_grad)
    with torch.no_grad():
        output = model(dummy_input)
    
    # --- 結果のプロット ---
    # 入力と出力が同じサイズであることを視覚的に確認します。
    # ※モデルは未学習（重みがランダム）なので、出力される模様自体に意味はありませんが、
    # 「同じサイズのノイズパターンを予測して出力する」というアーキテクチャが機能していることがわかります。
    
    plt.figure(figsize=(10, 4))
    
    # 入力画像の表示
    plt.subplot(1, 2, 1)
    plt.title(f"Input Noise\nShape: {dummy_input.shape}")
    plt.imshow(dummy_input[0, 0].numpy(), cmap='gray')
    plt.colorbar()
    
    # 出力画像の表示
    plt.subplot(1, 2, 2)
    plt.title(f"U-Net Output\nShape: {output.shape}")
    plt.imshow(output[0, 0].numpy(), cmap='gray')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print(f"入力テンソルサイズ: {dummy_input.shape}")
    print(f"出力テンソルサイズ: {output.shape}")
    print("U-Netの構造により、入力と同じ解像度のテンソルが正しく再構築されました。")
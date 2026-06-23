import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ==========================================
# パラメータ定義部
# ==========================================
class Config:
    # 自宅(8GB)でのテスト用: "Qwen/Qwen2.5-0.5B" (0.5Bモデル)
    # 研究室(5080)での本番用: "meta-llama/Meta-Llama-3-8B" などを指定
    model_name = "Qwen/Qwen2.5-0.5B" 
    
    # 学習させるデータセット (ここでは日本語の一般的な指示データを使用)
    dataset_name = "kunishou/databricks-dolly-15k-ja"
    
    # --- LoRAの理論パラメータ ---
    lora_r = 8          # 行列AとBのランク(r)。値が小さいほど計算は軽いが表現力が下がる
    lora_alpha = 16     # LoRAのスケーリング係数(通常はrの2倍程度)
    lora_dropout = 0.05 # 過学習を防ぐために一部のニューロンをランダムに無効化する割合
    
    # --- 学習のパラメータ ---
    batch_size = 4      # 1回の計算で処理するデータ数(VRAMが少ない時は1)
    max_steps = 500      # 理論確認のため、今回は30回の更新で終了させる(本格的な学習にはepochsを指定)
    learning_rate = 1e-4
    max_length = 256 # 一度に読み込む最大文字数(トークン数)。VRAMに直結する

# ==========================================
# 1. データセットの準備
# Hugging Faceからデータをダウンロードし、プロンプト形式に整形します。
# ==========================================
def format_prompts(examples):
    # 指示(instruction)と回答(output)を結合し、AIに学習させたい会話の形を作る
    texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        # 「ユーザーの質問 -> AIの回答」という文脈をモデルに教え込む
        text = f"ユーザー: {instruction}\nAI: {output}"
        texts.append(text)
    return {"text": texts}

# ==========================================
# 2. モデルとトークナイザーのロード (量子化の適用)
# メモリを節約するため、4bit量子化を適用してモデルをロードします。
# ==========================================
def load_quantized_model():
    # トークナイザー(文章を数値IDに変換する辞書)の読み込み
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    tokenizer.pad_token = tokenizer.eos_token # バッチ処理時の空白埋めルールを設定
    
    # 4bit量子化の設定: FP16の重みを4bitに圧縮してVRAM消費を抑える
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # 計算自体はFP16に戻して精度を確保
    )
    
    # 巨大な元の重み行列 W を4bitでVRAM上にロードする
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=bnb_config,
        device_map="auto" # 空いているGPUに自動配置
    )
    return model, tokenizer

# ==========================================
# 3. LoRAの適用 (低ランク行列の追加)
# 巨大なモデルを凍結し、学習用の小さな行列(A, B)をネットワークに挿入します。
# ==========================================
def apply_lora(model):
    # モデルを量子化トレーニング用に最適化(勾配チェックポイントなど)
    model = prepare_model_for_kbit_training(model)
    
    # W' = W + BA を実現するための設定
    config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=["q_proj", "v_proj"], # Transformer内のどの行列にLoRAを適用するか
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 元の重みを凍結し、ΔW(学習可能な小さなパラメータ)を追加したモデルを生成
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters() # 実際にどれだけパラメータが減ったか表示
    return peft_model

# ==========================================
# 実行部
# ==========================================
if __name__ == "__main__":
    print("モデルを準備中...")
    model, tokenizer = load_quantized_model()
    model = apply_lora(model)
    
    print("データを準備中...")
    dataset = load_dataset(Config.dataset_name, split="train")
    dataset = dataset.map(format_prompts, batched=True)
    
    # 学習の挙動を定義(ログの出力頻度などを設定)
    training_args = SFTConfig(
        output_dir="./results",
        dataset_text_field="text",
        max_length=Config.max_length,
        packing=False,                        # 文章を詰め込まず、1行ずつ処理する
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=4,      # 仮想的にバッチサイズを大きくして学習を安定させる
        learning_rate=Config.learning_rate,
        max_steps=Config.max_steps,         # テストのため30ステップで終了
        logging_steps=1,                    # 毎ステップごとのLoss(誤差)を記録
        optim="paged_adamw_32bit",          # メモリ効率の良いオプティマイザ
        fp16=False,                          # 高速化のための半精度計算
        bf16=True,
        lr_scheduler_type="cosine", # ←★追加: 後半に学習率を下げて着地を綺麗にする
        warmup_steps=50,
    )
    
    # Trainer(学習を実行するエンジン)の初期化
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("学習を開始します！(Lossが下がっていくか確認しましょう)")
    trainer.train()
    
    # --- 結果(Lossの推移)のプロット ---
    # 学習が正しく進んでいれば、Loss(AIの予測の誤差)は右肩下がりのグラフになります。
    history = trainer.state.log_history
    steps = [log["step"] for log in history if "loss" in log]
    losses = [log["loss"] for log in history if "loss" in log]
    
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker='o', color='b', label='Training Loss')
    plt.title("LoRA Fine-tuning Loss Curve")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss (Error)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print("学習テストが完了しました！")
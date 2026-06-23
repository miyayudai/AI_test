#!/bin/bash

CPUS="0.5"
MEMORY="128m"
PIDS_LIMIT="20"
IMAGE="cpp-sandbox"
TIMEOUT="2s"
BIN_DIR="$(pwd)/bin"
RESULT_FILE="$(pwd)/res/result.txt"

while getopts c:m:p:i:t:b:r: flag
do
    case "${flag}" in
        c) CPUS=${OPTARG};;
        m) MEMORY=${OPTARG};;
        p) PIDS_LIMIT=${OPTARG};;
        i) IMAGE=${OPTARG};;
        t) TIMEOUT=${OPTARG};;
        b) BIN_DIR=$(realpath "${OPTARG}");;
        r) RESULT_FILE=$(realpath "${OPTARG}");;
        *) 
            echo "使い方: $0 [-c CPUS] [-m MEMORY] [-p PIDS] [-i IMAGE] [-t TIMEOUT] [-b BIN_DIR] [-r RESULT_FILE] <入力ファイルのパス>"
            exit 1;;
    esac
done

# オプション引数 (-c など) をスキップして、残りの位置引数を取得する
shift $((OPTIND -1))

# 1. 引数の数を確認（入力ファイルが指定されているか）
if [ "$#" -ne 1 ]; then
    echo "使い方: ./run_code.sh <入力ファイルのパス>"
    exit 1
fi

INPUT_FILE=$1

# 2. 指定された入力ファイルが実際に存在するかチェック
if [ ! -f "$INPUT_FILE" ]; then
    echo "エラー: 入力ファイル '$INPUT_FILE' が見つかりません。"
    exit 1
fi

# 念のため、出力先ファイルのディレクトリが存在しない場合は作成しておく
mkdir -p "$(dirname "${RESULT_FILE}")"

echo "[${IMAGE}] CPU: ${CPUS}, RAM: ${MEMORY}, Timeout: ${TIMEOUT} でプログラムを実行しています..."

# 3. Dockerコマンドの実行
# 引数で受け取った INPUT_FILE を読み込み、RESULT_FILE に出力する
docker run --rm -i \
  --network none \
  --cpus="${CPUS}" \
  --memory="${MEMORY}" \
  --pids-limit "${PIDS_LIMIT}" \
  --read-only \
  -v "$(pwd)/bin:/app/bin:ro" \
  ${IMAGE} \
  timeout ${TIMEOUT} /app/bin/output.out < "$INPUT_FILE" > "$RESULT_FILE"

# 4. 実行完了の通知と結果の表示
if [ $? -eq 0 ]; then
    echo "実行完了！ 結果は $RESULT_FILE に保存されました。"
    echo "--- 結果 ---"
    cat "$RESULT_FILE"
    echo "-----------"
else
    echo "コンパイル失敗、またはタイムアウトしました。"
    exit 1
fi

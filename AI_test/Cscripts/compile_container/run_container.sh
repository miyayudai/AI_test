#!/bin/bash

# デフォルト設定
CPUS="1.0"
MEMORY="512m"
IMAGE="cpp-sandbox"
SRC_DIR="$(pwd)/src"
BIN_DIR="$(pwd)/bin"

# オプションの解析
while getopts c:m:i:s:b: flag
do
    case "${flag}" in
        c) CPUS=${OPTARG};;
        m) MEMORY=${OPTARG};;
        i) IMAGE=${OPTARG};;
        s) SRC_DIR=$(realpath ${OPTARG});;
        b) BIN_DIR=$(realpath ${OPTARG});;
    esac
done

docker run --rm \
  --network none \
  --cpus="${CPUS}" \
  --memory="${MEMORY}" \
  --pids-limit 50 \
  -v "${SRC_DIR}:/app/src:ro" \
  -v "${BIN_DIR}:/app/bin:rw" \
  ${IMAGE} \
  g++ /app/src/*.cpp -o /app/bin/output.out -O2

echo "[${IMAGE}] CPU: ${CPUS}, RAM: ${MEMORY} で起動しました！"
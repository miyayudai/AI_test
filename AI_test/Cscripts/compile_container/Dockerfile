# ベースイメージとして軽量なAlpine Linuxを採用
FROM alpine:latest

# G++とMakeなど、コンパイルに必要な最小限のパッケージをインストール
RUN apk add --no-cache g++ make

# セキュリティのため、非rootユーザー（sandbox_user）を作成
RUN addgroup -S sandbox_group && adduser -S sandbox_user -G sandbox_group

# 作業ディレクトリの作成と権限付与
WORKDIR /app
RUN chown sandbox_user:sandbox_group /app

# 以降のコマンドを非rootユーザーで実行
USER sandbox_user

# デフォルトのコマンド（後でホスト側から上書きします）
CMD ["g++", "--version"]
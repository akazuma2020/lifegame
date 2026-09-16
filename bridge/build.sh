#!/bin/sh
# bridge.cを、Lispから読み込める共有ライブラリbridge.soへ変換するスクリプト。
# 実行場所に左右されないよう、最初にこのスクリプト自身のディレクトリへ移動する。
set -eu
cd "$(dirname "$0")"
# -Wall/-Wextraで多くの警告を有効にし、-Werrorで警告も失敗として扱う。
# -O3は高速化、-shared -fPICは共有ライブラリを作る指定。
cc -std=c11 -Wall -Wextra -Werror -O3 -shared -fPIC \
    -o bridge.so bridge.c \
    -L/usr/local/lib -Wl,-rpath,/usr/local/lib -l:libwgpu_native.so -lSDL2

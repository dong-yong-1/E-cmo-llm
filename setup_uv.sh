#!/bin/bash

# --------------------------
# 一键初始化 uv 虚拟环境
# --------------------------

# 1️⃣ 安装 uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 2️⃣ 创建虚拟环境
[ -d ".venv" ] || uv venv

# 3️⃣ 同步依赖
# 安装 GPU 版本，如果只想 CPU 改成 --extra cpu
uv sync --extra cpu

# 4️⃣ 激活环境
source .venv/bin/activate

# 5️⃣ 提示完成
echo "uv 虚拟环境激活完成！"
echo "使用 python --version 查看 Python 版本"
echo "使用 python -c 'import torch; print(torch.cuda.is_available())' 测试 GPU 是否可用"
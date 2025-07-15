
# Anaconda + VSCode + PyTorch 安装指南
## 1. 安装Anaconda


1. 访问 [Anaconda官网](https://www.anaconda.com/download) 下载对应系统的安装包
2. 运行安装程序
```bash
# 安装完成后验证
conda --version
```

## 2. 安装VSCode

1. 访问 [VSCode官网](https://code.visualstudio.com/download) 下载对应系统的安装包

2. 运行安装程序

3. 如果未安装Python扩展，请点击界面提示或访问扩展市场搜索“Python”并安装。其它常用vscode扩展：
   - Python Debugger
   - Jupyter
   - Jinja
   - Markdown All in One
   - IntelliCode
   - Github Copilot


## 3. 创建虚拟环境
开始-->搜索打开Anaconda Prompt，输入以下命令创建虚拟环境：
```bash
# 创建名为 ml 的Python 3.8环境
conda create -n ml python=3.8

# 激活环境
conda activate ml

# 验证Python版本
python --version
```


## 4. 安装PyTorch
在Anaconda Prompt中激活虚拟环境后，输入以下命令安装PyTorch：
```bash
# 使用conda安装PyTorch (CPU版本)（适用于没有NVIDIA显卡或不需要GPU加速的用户）
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 或GPU版本 (以CUDA 11.8为例)（适用于有NVIDIA显卡或需要GPU加速的用户）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
如果上述命令无法满足需求，请访问PyTorch官网，选择合适的系统、Python版本、安装方式和CUDA版本
[PyTorch官网](https://pytorch.org/get-started/locally/)


## 5. 安装常用机器学习包
```bash
# 确保在ml环境中
conda activate ml

# 安装常用包
conda install numpy pandas matplotlib scikit-learn jupyter

# 或使用pip
pip install numpy pandas matplotlib scikit-learn jupyter
```


## 6. 配置VSCode

1. 打开VSCode
   
2. 按`Ctrl+Shift+P`打开命令面板
   
3. 输入"Python: Select Interpreter"
   
4. 选择之前创建的`ml`环境

## 7. 测试安装

创建一个`test.py`文件：
```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 测试PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 创建简单的数据集
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# 绘制数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Test Classification Dataset')
plt.show()
```

运行文件，如果看到数据集绘制成功，则说明安装成功。


## 8. 有用的VSCode快捷键

- `Ctrl+Shift+P`: 命令面板
- `Ctrl+Space`: 代码补全
- `F5`: 运行代码
- `F10`: 调试

## 9. 安装package时遇到网络问题

- 使用清华源
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```
- Conda使用清华源
```bash
# 临时使用清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
# 确保更改生效
conda clean -i
# 查看源
conda config --get channels
```

## 注意事项

- 确保系统满足PyTorch的要求
- 如果使用GPU，确保安装了正确版本的CUDA
- 定期更新包以获取最新特性和bug修复

## 参考链接

- [Anaconda官网](https://www.anaconda.com/)
- [PyTorch官网](https://pytorch.org/)
- [VSCode官网](https://code.visualstudio.com/)
- [清华源](https://pypi.tuna.tsinghua.edu.cn/)
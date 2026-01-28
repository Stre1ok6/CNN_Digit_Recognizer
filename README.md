# CNN_Digit_Recognizer
本项目基于卷积神经网络（CNN）实现 MNIST 手写数字数据集的识别，包含模型训练脚本（model.py）和可视化交互界面脚本（gui.py），支持训练模型并通过手绘方式实时识别数字。

## 神经网络结构
<img width="1739" height="808" alt="image" src="https://github.com/user-attachments/assets/f67cf208-3902-4682-9d17-fd3e0db11380" />
核心由3个卷积特征提取块、1个自适应平均池化层、展平操作、1个全连接分类层组成。

## 项目结构
````
├── model.py       # CNN模型定义、训练、评估脚本
├── gui.py         # TKinter可视化界面，手绘数字识别
├── model.pth      # 训练完成后生成的模型权重文件（运行model.py后生成）
├── mnist_data/    # MNIST数据集下载目录（运行model.py时自动下载）
└── README.md      # 项目说明文档
````

## 环境配置
````bash
pip install torch torchvision matplotlib pillow numpy
````
*\*tkinter通常为python自带*

## 运行步骤
```python model.py```

运行model.py脚本，完成以下操作：   
自动下载MNIST数据集到mnist_data/目录；   
训练15轮CNN模型，打印每轮训练损失、训练集准确率、测试集准确率；   
训练完成后绘制4个测试样本的预测结果可视化图；   
保存训练好的模型到model.pth文件。

```python gui.py```

训练完成后，运行gui.py启动手绘识别界面。

<img width="400" height="550" alt="0deaf66659dc07127bf62868652725b4" src="https://github.com/user-attachments/assets/31ecb79d-fc67-474c-aad2-97470c05c431" />

启动界面后，在黑色画布上用鼠标左键手绘 0-9 的数字；   
绘制完成后点击“识别”按钮，下方会显示模型识别的数字结果；   
如需重新绘制，点击“清除”按钮清空画布即可。

*\*手绘数字时尽量居中、清晰，贴合 MNIST 数据集的手写风格，可提升识别准确率；*

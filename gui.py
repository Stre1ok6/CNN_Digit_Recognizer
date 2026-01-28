import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from tkinter import font

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
class CnnNet(nn.Module):
    def __init__(self, classes=10):
        super(CnnNet, self).__init__()
        self.classes = classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.advpool(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out

model = CnnNet()

# 加载参数并移到GPU
model = torch.load('model.pth',weights_only=False)
model.to(device)
model.eval()

PIXEL_ROWS = 28  # 固定28行像素
PIXEL_COLS = 28  # 固定28列像素
CELL_SIZE = 15  # 每个像素格的放大尺寸
CANVAS_W = PIXEL_COLS * CELL_SIZE
CANVAS_H = PIXEL_ROWS * CELL_SIZE
GRID_COLOR = '#333333'  # 网格线颜色
PIXEL_COLOR = 'white'  # 像素填充颜色
BG_COLOR = 'black'  # 画布背景色

img = Image.new('L', (PIXEL_COLS, PIXEL_ROWS), 0)  # 单通道灰度图
img_draw = ImageDraw.Draw(img)


def draw_grid(canvas):
    # 绘制竖线
    for col in range(PIXEL_COLS + 1):
        x = col * CELL_SIZE
        canvas.create_line(x, 0, x, CANVAS_H, fill=GRID_COLOR, width=1)
    # 绘制横线
    for row in range(PIXEL_ROWS + 1):
        y = row * CELL_SIZE
        canvas.create_line(0, y, CANVAS_W, y, fill=GRID_COLOR, width=1)


def clear_canvas():
    canvas.delete('all')  # 清空所有绘制内容
    draw_grid(canvas)  # 重新绘制网格
    global img, img_draw
    img = Image.new('L', (PIXEL_COLS, PIXEL_ROWS), 0)  # 重置底层28×28黑色图像
    img_draw = ImageDraw.Draw(img)
    output_label.config(text='')


def draw_pixel(event):
    # 将鼠标屏幕坐标转换为28×28的像素格索引
    pixel_col = event.x // CELL_SIZE
    pixel_row = event.y // CELL_SIZE
    # 过滤越界的像素格
    if 0 <= pixel_col < PIXEL_COLS and 0 <= pixel_row < PIXEL_ROWS:
        # 计算当前像素格在屏幕上的坐标范围
        x1 = pixel_col * CELL_SIZE
        y1 = pixel_row * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        # 填充屏幕上的整格像素
        canvas.create_rectangle(x1, y1, x2, y2, fill=PIXEL_COLOR, outline='')
        # 同步填充底层28×28图像的对应像素
        img_draw.point((pixel_col, pixel_row), fill=255)


def recognize_digit():
    # 底层图像转数组预处理
    input_data = np.array(img).reshape((1, 1, 28, 28)).astype('float32')
    input_data = input_data / 255.0  # 归一化

    # 模型预测
    input_tensor = torch.from_numpy(input_data).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()

    # 显示识别结果
    output_label.config(text=f'识别结果: {prediction}', font=('Arial', 20, 'bold'))


if __name__ == '__main__':
    root = tk.Tk()
    root.title("28×28像素化手写数字识别")
    root.resizable(False, False)  # 固定窗口大小

    # 创建28×28像素格画布
    canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg=BG_COLOR)
    canvas.pack(pady=10)
    draw_grid(canvas)  # 初始化绘制网格线

    # 绑定绘画事件
    canvas.bind('<B1-Motion>', draw_pixel)
    canvas.bind('<Button-1>', draw_pixel)

    # 按钮框架
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)

    recognize_btn = tk.Button(btn_frame, text="识别", command=recognize_digit,
                              width=10, height=2, font=('Arial', 12))
    recognize_btn.grid(row=0, column=0, padx=20)

    clear_btn = tk.Button(btn_frame, text="清除", command=clear_canvas,
                          width=10, height=2, font=('Arial', 12))
    clear_btn.grid(row=0, column=1, padx=20)

    # 识别结果标签
    output_label = tk.Label(root, text='', font=('Arial', 18), bg=BG_COLOR, fg=PIXEL_COLOR)
    output_label.pack(pady=10, fill=tk.X)

    # 窗口居中显示
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_x = (screen_w - CANVAS_W) // 2
    win_y = (screen_h - CANVAS_H - 100) // 2
    root.geometry(f"{CANVAS_W}x{CANVAS_H + 150}+{win_x}+{win_y}")

    root.mainloop()
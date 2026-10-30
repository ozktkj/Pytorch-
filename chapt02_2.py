import numpy as np
from PIL import Image, ImageTk
import cv2
from time import time, sleep
from tkinter import Tk, NW, TOP, Frame, Canvas, Label, StringVar
import threading

import torch
from torch import nn
from torchvision import transforms, models

# GPUを使うかどうか
USE_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# アプリケーションが実行中かどうか
IS_RUN = True

# 保存しておいたモデルを読み込む
model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 2)
# model.load_state_dict(torch.load('chapt02-model1.pth', map_location=torch.device(USE_DEVICE)))
model.load_state_dict(torch.load('chapt02-model1.pth', map_location=torch.device(USE_DEVICE)))
model = model.to(USE_DEVICE)
# モデルを推論用に設定する
model.eval()

# 画像を正規化するtransformを用意しておく
trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Tkで表示するフレーム
class MyFrame(Frame):
    def __init__(self, parent, **params):
        Frame.__init__(self, parent, params)
        # OpenCVのカメラキャプチャーを用意する
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # OpenCVのカメラキャプチャーを用意する(サンプルの動画を利用して、プログラムの実行をテストする場合)
        # self.cap = cv2.VideoCapture('chapt02-sample.mp4')
        # 画像の表示場所を作成する
        self.image = Image.new('RGB',(320,240),(0,0,0))
        self.imgtk = ImageTk.PhotoImage(self.image)
        self.canvas = Canvas(self, width=320, height=240, bg='black')
        self.canvas.place(x=20,y=20)
        self.canvas.create_image(0,0,image=self.imgtk,anchor=NW,tag='i')
        # 認識結果の表示場所を作成する
        self.message = StringVar()
        self.message.set('')
        self.label = Label(self, textvariable=self.message)
        self.label.place(x=360,y=20)

    def updateFrame(self):
        # OpenCVでカメラからキャプチャーする
        ret, frame = self.cap.read()
        # OpenCVのBGRをRGBにする
        frame = frame[:,:,::-1]
        # 画像を表示させる
        self.image = Image.fromarray(frame)
        self.imgtk = ImageTk.PhotoImage(self.image.resize((320,240)))
        self.canvas.itemconfigure(tagOrId='i', image=self.imgtk)
        # 0.1秒後に再び更新する
        self.after(100, self.updateFrame)

# ニューラルネットワークを実行するスレッド
def detect(frame):
    global IS_RUN
    n_pred = 0
    s_time = time()
    fps_window = []
    fps_window_size = 10  # 表示するフレーム数
    with torch.no_grad():
        # アプリケーションの実行中は無限ループ
        while IS_RUN:
            # キャプチャー速度に合わせて最大fpsを調整する
            a_time = time()
            # 現在のキャプチャ画像を取得してtensorにする
            tensor = trans(frame.image)
            # バッチサイズ＝1の形式にする
            batch_tensor = tensor.view(1,3,224,224)
            batch_tensor = batch_tensor.to(USE_DEVICE)
            # ニューラルネットワークを実行する
            batch_result = model(batch_tensor)
            # バッチ内から結果を取得する
            result = batch_result.detach().cpu()
            result = result[0]
            # メッセージを更新する
            if IS_RUN:
                n_pred += 1
                fps = n_pred / (time() - s_time)
                fps_window.append(fps)
                if len(fps_window) > fps_window_size:
                    fps_window.pop(0)
                avg_fps = np.mean(fps_window)
                # 結果からクラスの確率の高い方を認識とする
                predict = '誰もいない' if result[0] > result[1] else '歩行者あり'
                # 表示メッセージ
                message = f'''認識結果：{predict}
                出力値0: {result[0]}
                出力値1: {result[1]}
                認識実行回数={n_pred}
                平均認識速度={avg_fps:.2f}fps'''
                # UI上に表示する
                frame.message.set(message)
                # 0.1秒以下だったらその分待つ
                deltime = (time() - a_time)
                if deltime < 0.1:
                    sleep(0.1-deltime)

# 画面いっぱいにウィンドウを作成する
win = Tk()
win.geometry('800x360') # ウィンドウの大きさ
win.title("リアルタイム歩行者認識")  # ウィンドウタイトルを設定
frame = MyFrame(win, width=800, height=360, bg='#000000') # フレーム
frame.pack(side=TOP) # ウィンドウに配置
win.after_idle(frame.updateFrame) # 起動後にupdateFrameを呼び出す

detection = threading.Thread(target=detect, args=(frame,))
detection.start()

win.mainloop() # 処理を開始

# スレッドの終了を待つ
IS_RUN = False
detection.join()

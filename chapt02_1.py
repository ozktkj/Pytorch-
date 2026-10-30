import numpy as np
import pandas as pd
import itertools
import shutil
from PIL import Image
import os,sys

import torch
from torch import nn, utils, optim
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# GPUを使うかどうか
USE_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# データがあるディレクトリ
INPUT_DIR = 'forest-path-movie-dataset/'

#動作確認
print(USE_DEVICE) #GPUを認識しているか
print(os.path.isdir(INPUT_DIR)) #データディレクトリを確認
# 一旦停止
# sys.exit()


# PyTorchの内部を決定論的に設定する
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 乱数を初期化する関数
def init_seed():
    np.random.seed(0)
    torch.manual_seed(0)

# 一時ディレクトリを作成
if not os.path.isdir('tmp'):
    os.mkdir('tmp')

# データセットの定義ファイルを読み込む
df = pd.read_csv(INPUT_DIR+'all_file.csv')

# シーン毎に分割するので、groupbyして取り出す
file, person = [], []
for g in df.groupby(df.scene):
    file.append(g[1].file.values.tolist())
    person.append(g[1].person.values.tolist())

# シーン毎に学習用と評価用データに分ける
train_X, test_X, train_y, test_y = train_test_split(file, person, test_size=0.3, random_state=0)

# 全てのシーン内のデータを繋げた配列にする
train_X = sum(train_X, [])
train_y = sum(train_y, [])
test_X = sum(test_X, [])
test_y = sum(test_y, [])

# PyTorchの流儀でデータセットをクラスで定義する
class MyDataset:
    def __init__(self, X, y, valid=False):
        # 初期化 Xはファイル名のリスト、yは人物が写っているかどうかのリスト
        self.X = X
        self.y = y
        if not valid: # 学習用ならDAを含んだtransoformを作る
            trans = [
                transforms.Resize((224,224)),
                transforms.ColorJitter(brightness=1.0),
                transforms.RandomGrayscale(0.1),
                transforms.ToTensor(),
                transforms.RandomErasing(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ]
        else: # 評価時にはDAを含まないtransoformを作る
            trans = [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        self.trans = transforms.Compose(trans)

    def __len__(self):
        # データセットの長さを返す
        return len(self.X)

    def __getitem__(self, pos):
        # posの場所にあるデータを返す
        f = INPUT_DIR + self.X[pos] # ファイルパス
        X = Image.open(f) # ファイルを読み込む
        X = self.trans(X) # DAしてtensorにする
        y = self.y[pos]
        return X, y

def get_model(): # ニューラルネットワークのモデルを返す関数
    # ModelZOOからモデルをダウンロードして最後の層だけを入れ替える
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, 2) # 出力の数＝2にする
    model = model.to(USE_DEVICE) # GPUを使うときはGPUメモリ上に乗せる
    return model

def get_optim(model, lr): # 勾配降下法のアルゴリズムを返す関数
    params = model.parameters() # 学習させるパラメーター
    optimizer = optim.SGD(params, lr=lr,  momentum=0.9) # 学習率を設定
    return optimizer

def get_loss(weight): # 損失関数を返す関数
    # 不均衡なデータを学習させるために、クラス毎のウェイトを設定する
    weight = torch.tensor([1.0-weight,weight], dtype=torch.float)
    weight = weight.to(USE_DEVICE) # GPUを使うときはGPUメモリ上に乗せる
    loss = nn.CrossEntropyLoss(weight=weight) # ウェイト付きのCrossEntropy
    return loss

def get_score(true_valid, pred_valid): # 評価スコアを返す関数
    # 評価スコアは、全体の他に時間帯毎にも作成するのでディクショナリを用意する
    timezone = {'daytime':([],[]),'twilight':([],[]),'midnight':([],[])}
    # 認識結果を時間帯毎に仕分けする
    for i, filename in enumerate(test_X):
        w = df[df.file==filename].when.values[0]
        timezone[w][0].append(true_valid[i])
        timezone[w][1].append(pred_valid[i])
    # 時間帯毎のF1スコアをディクショナリに入れる
    score = {k:f1_score(v[0], v[1]) for k,v in timezone.items()}
    # 全体のF1スコアをディクショナリに入れる
    score['total'] = f1_score(true_valid, pred_valid)
    # 認識が極端に偏ってないか見るために、認識値の平均も求める
    score['average'] = np.mean(pred_valid)
    return score

if __name__ == '__main__':
    # 学習時と評価時のバッチサイズ
    BATCH_SIZE = 16
    BATCH_SIZE_VALID = 4
    # データの読み込みスレッドの数
    NUM_WORKERS = 2
    # 試行時の学習エポック数
    NUM_EPOCHS = 3
    # 評価で試す学習率
    LR_TESTS = [1e-3,2e-4,5e-5]
    # 試すウェイトは、人物の方が分散が大きいので、クラス1側を0.5より少なくする
    WEIGHT_TESTS = [0.1,0.2,0.3,0.4,0.5]

    # 学習用と評価用にデータセットを作る
    train_ds = MyDataset(train_X, train_y)
    test_ds = MyDataset(test_X, test_y, True)

    # 複数スレッドでファイルを読み込みつつデータを取り出すDataLoaderを作る
    data_loader = utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    data_loader_v = utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE_VALID, shuffle=False, num_workers=NUM_WORKERS)

    # 試行時の最も評価が良かったスコアのリスト
    best_scores = []
    # 学習率とウェイトを変えながら試行する
    for t, (lr, weight) in enumerate(itertools.product(LR_TESTS, WEIGHT_TESTS)):
        # 試行毎に乱数を初期化してからニューラルネットワークを作成する
        init_seed()
        model = get_model() # ニューラルネットワークを作成
        # 学習のためのアルゴリズムを取得
        optimizer = get_optim(model, lr)
        loss = get_loss(weight)

        # 現在の学習率とウェイトで試行する
        print(f'test #{t} lr={lr} weight={weight}')
        scores = [] # 各エポック終了時のスコア

        # 学習ループ
        for epoch in range(NUM_EPOCHS):
            total_loss = [] # 各バッチ実行時の損失値
            model.train() # モデルを学習用に設定する
            for X, y in data_loader: # 画像を読み込んでtensorにする
                X = X.to(USE_DEVICE) # GPUを使うときはGPUメモリ上に乗せる
                y = y.to(USE_DEVICE) # GPUを使うときはGPUメモリ上に乗せる

                # ニューラルネットワークを実行して損失値を求める
                losses = loss(model(X), y)

                # 新しいバッチ分の学習を行う
                optimizer.zero_grad() # 一つ前の勾配をクリア
                losses.backward() # 損失値を逆伝播させる
                optimizer.step() # 新しい勾配からパラメーターを更新する

                # 損失値を保存しておく
                total_loss.append(losses.detach().cpu().numpy())

            # 評価
            with torch.no_grad():
                # 評価時の損失値と正解/認識結果を入れるリスト
                total_loss_v = []
                true_valid = []
                pred_valid = []

                model.eval() # モデルを推論用に設定する
                for i, (X, y) in enumerate(data_loader_v):
                    X = X.to(USE_DEVICE) # GPUを使うときはGPUメモリ上に乗せる
                    y = y.to(USE_DEVICE) # GPUを使うときはGPUメモリ上に乗せる

                    res = model(X) # ニューラルネットワークの実行
                    losses = loss(res, y) # 評価データの損失値

                    # 正解データを保存
                    y = y.detach().cpu().numpy() # CPUメモリに入れてnumpy化
                    true_valid.extend(y.tolist())
                    # 認識結果を保存
                    res = res.detach().cpu().numpy() # CPUメモリに入れてnumpy化
                    pred_valid.extend(res.argmax(axis=1).tolist())

                    # 損失値を保存しておく
                    total_loss_v.append(losses.detach().cpu().numpy())

            # エポック終了時のスコアを求める
            total_loss = np.mean(total_loss) # 各バッチの損失の平均
            total_loss_v = np.mean(total_loss_v) # 各バッチの損失の平均
            score = get_score(true_valid, pred_valid) # 評価スコア
            scores.append(score['total']) # スコアを保存しておく
            # エポック終了時のスコアを表示する
            print(f'epoch #{epoch}: train_loss:{total_loss} valid_loss:{total_loss_v} score:{score}')
            # エポック終了時のモデルを保存しておく
            torch.save(model.state_dict(), f'tmp/checkpoint{epoch}.pth')

        # 現在の学習率とウェイトで最も良かったモデルをコピーして保存しておく
        best_epoch = np.argmax(scores)
        shutil.copyfile(f'tmp/checkpoint{best_epoch}.pth',f'tmp/test{t}_best.pth')
        # 現在の学習率とウェイトで最も良かったモデルを損失値しておく
        best_scores.append(scores[best_epoch])

        # GPUメモリをGCする
        del model, optimizer, loss, X, y, res, losses
        torch.cuda.empty_cache()

    # 最も良かった学習率とウェイトでのモデルをコピーする
    best_of_best = np.argmax(best_scores)
    shutil.copyfile(f'tmp/test{best_of_best}_best.pth', 'chapt02-model1.pth')

    # 一時ディレクトリを削除
    shutil.rmtree('tmp')

#!/usr/bin/env python3
"""模倣学習（Behavior Cloning）のモデル定義。
train_bc.py（学習）と run_policy.py（実行）の両方から読み込んで使う。

入力: カメラ画像 (3, 120, 160) と 現在の関節角度 (5)
出力: 4関節それぞれの「0.5秒後までに動かす量」(4) と グリッパを閉じるかどうか (1)
"""
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, n_joint_in=5, n_arm=4):
        super().__init__()
        # 画像の特徴を取り出す小さな CNN（120x160 → 8x10）
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),    # 60x80
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 30x40
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 15x20
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 8x10
        )
        # キューブが「画面のどこにあるか」が大事なので、位置の情報を残したまま平らにする
        self.img_fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 8 * 10, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + n_joint_in, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_arm + 1),
        )

    def forward(self, img, joint):
        """img: (B,3,120,160) 0〜1,  joint: (B,5) 正規化済み"""
        f = self.img_fc(self.cnn(img))
        out = self.head(torch.cat([f, joint], dim=1))
        return out[:, :-1], out[:, -1]   # (関節の動き, グリッパのロジット)
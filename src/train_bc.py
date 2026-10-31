#!/usr/bin/env python3
"""テレオペのデモデータから模倣学習（Behavior Cloning）でモデルを学習する。

使い方:
    python3 train_bc.py --data_dir /workspace/pai_final/dataset \
                        --out /workspace/pai_final/models/bc_v3.pt

    --epochs        学習の回数（既定 40）
    --val_episodes  検証用に取り分けるエピソード数（既定 1。0 なら全部で学習）
    --target        学習する目標（既定 command）
                      command: テレオペで出していた指令（今の角度から指令がどれだけ先か）
                      future : 実際の関節角度が --horizon フレーム後にどれだけ変わるか（v1 の方式）
    --horizon       future のとき何フレーム先を見るか（既定 5 = 0.5秒先）

学習するもの:
    カメラ画像 + 今の関節角度  →  (1) 各関節の指令を、今の角度からどれだけ先に出すか
                                  (2) グリッパを閉じるかどうか
データの前処理:
    ・記録開始から、人が最初にキーを押すまでのフレームは取り除く（R の動きの残り）
    ・ロボットが止まって考えているだけのフレームは取り除く
    ・グリッパを閉じた後のフレームは、閉じてから1秒分だけ使う
出力:
    モデルファイル（.pt）と、学習の経過 <モデル名>_log.csv
"""
import argparse
import csv
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bc_model import BCPolicy

MOVE_THRESH = 0.01      # この量 [rad] 以上動く予定のフレームを「動いている」とみなす
CLOSE_KEEP = 10         # グリッパを閉じた後に使うフレーム数（10 = 1秒）


def load_episode(path, horizon, target='command'):
    d = np.load(path)
    img, jp, ac = d['images'], d['joint_pos'], d['action']
    T = len(jp)
    fut = np.minimum(np.arange(T) + horizon, T - 1)
    moved = jp[fut, :4] - jp[:, :4]                    # 実際に0.5秒後までに動いた量
    if target == 'command':
        delta = ac[:, :4] - jp[:, :4]                  # テレオペの指令が今の角度からどれだけ先か
    else:
        delta = moved
    grip = (ac[:, 4] > 0).astype(np.float32)          # 1 = 閉じる指令

    # 使うフレームを選ぶ
    moving = (np.abs(delta).sum(1) > MOVE_THRESH) | (np.abs(moved).sum(1) > MOVE_THRESH)
    keep = moving & (grip == 0)
    # 記録開始直後、まだ人がキーを押していない間は使わない
    # （R の自動の動きの残り：指令はプリセット値のまま、実物が遅れて追いついている途中）
    changed = np.any(np.abs(ac - ac[0]) > 1e-4, axis=1)
    first_op = int(np.argmax(changed)) if changed.any() else T
    keep[:first_op] = False
    # 「開 → 閉」に切り替わった最後の瞬間を、キューブを掴んだ瞬間とみなす
    trans = np.where((grip[1:] == 1) & (grip[:-1] == 0))[0]
    if len(trans) > 0:
        c0 = trans[-1] + 1
        keep[c0:c0 + CLOSE_KEEP] = True               # 閉じた直後の1秒分
        keep[c0 + CLOSE_KEEP:] = False                 # それ以降は使わない
    return img[keep], jp[keep], delta[keep], grip[keep], T, first_op


def augment(img):
    """画像の水増し: 上下左右に最大4ピクセルずらす + 明るさを少し変える"""
    B, C, H, W = img.shape
    pad = 4
    padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    out = torch.empty_like(img)
    for b in range(B):
        dy, dx = random.randint(0, 2 * pad), random.randint(0, 2 * pad)
        out[b] = padded[b, :, dy:dy + H, dx:dx + W]
    scale = 1.0 + (torch.rand(B, 1, 1, 1) - 0.5) * 0.2
    return (out * scale).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/workspace/pai_final/dataset')
    ap.add_argument('--out', default='/workspace/pai_final/models/bc_v3.pt')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--horizon', type=int, default=5)
    ap.add_argument('--target', choices=['command', 'future'], default='command')
    ap.add_argument('--val_episodes', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)

    paths = sorted(glob.glob(os.path.join(args.data_dir, 'episode_*.npz')))
    if not paths:
        print(f"[ERROR] {args.data_dir} に episode_*.npz がありません"); return
    order = paths[:]
    random.shuffle(order)
    val_paths = order[:args.val_episodes]
    train_paths = order[args.val_episodes:]

    def load_all(ps, label):
        I, J, D, G = [], [], [], []
        for p in ps:
            i, j, d, g, T, f0 = load_episode(p, args.horizon, args.target)
            print(f"  [{label}] {os.path.basename(p)}: {T}フレーム中 {len(i)}フレームを使用"
                  f"（最初の操作まで {f0}フレームは除外）")
            I.append(i); J.append(j); D.append(d); G.append(g)
        if not I:
            return None
        return (np.concatenate(I), np.concatenate(J),
                np.concatenate(D), np.concatenate(G))

    print("データ読み込み:")
    train = load_all(train_paths, '学習')
    val = load_all(val_paths, '検証')

    # 正規化の統計量（学習データから計算）
    j_mean, j_std = train[1].mean(0), train[1].std(0) + 1e-3
    d_std = train[2].std(0) + 1e-3
    pos_rate = train[3].mean()
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-3))
    print(f"学習フレーム数 {len(train[0])}, 閉じる指令の割合 {pos_rate:.1%}")

    def to_tensors(data):
        img = torch.from_numpy(data[0]).permute(0, 3, 1, 2).contiguous()   # uint8
        joint = torch.from_numpy((data[1] - j_mean) / j_std).float()
        delta = torch.from_numpy(data[2] / d_std).float()
        grip = torch.from_numpy(data[3]).float()
        return img, joint, delta, grip

    tr = to_tensors(train)
    va = to_tensors(val) if val is not None else None

    model = BCPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def run_epoch(data, train_mode):
        model.train(train_mode)
        n = len(data[0])
        idx = torch.randperm(n) if train_mode else torch.arange(n)
        tot_d = tot_g = tot_acc = 0.0
        for s in range(0, n, args.batch):
            b = idx[s:s + args.batch]
            img = data[0][b].float() / 255.0
            if train_mode:
                img = augment(img)
            with torch.set_grad_enabled(train_mode):
                pd, pg = model(img, data[1][b])
                loss_d = F.mse_loss(pd, data[2][b])
                loss_g = F.binary_cross_entropy_with_logits(
                    pg, data[3][b], pos_weight=pos_weight)
                loss = loss_d + loss_g
                if train_mode:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot_d += loss_d.item() * len(b)
            tot_g += loss_g.item() * len(b)
            tot_acc += ((pg > 0).float() == data[3][b]).float().sum().item()
        return tot_d / n, tot_g / n, tot_acc / n

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log_path = os.path.splitext(args.out)[0] + '_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_move_loss', 'train_grip_loss',
                                'val_move_loss', 'val_grip_loss', 'val_grip_acc'])

    print(f"\n学習開始（{args.epochs}エポック）")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        trd, trg, _ = run_epoch(tr, True)
        sched.step()
        if va is not None:
            vd, vg, vacc = run_epoch(va, False)
        else:
            vd = vg = vacc = float('nan')
        print(f"epoch {ep:3d}  学習: 動き {trd:.3f} 把持 {trg:.3f} | "
              f"検証: 動き {vd:.3f} 把持 {vg:.3f} 把持正解率 {vacc:.0%}  "
              f"({time.time() - t0:.0f}秒)")
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, trd, trg, vd, vg, vacc])

    torch.save({
        'model': model.state_dict(),
        'j_mean': j_mean, 'j_std': j_std, 'd_std': d_std,
        'horizon': args.horizon,
        'target': args.target,
        'train_episodes': [os.path.basename(p) for p in train_paths],
        'val_episodes': [os.path.basename(p) for p in val_paths],
    }, args.out)
    print(f"\nモデルを保存しました → {args.out}")
    print(f"学習の経過 → {log_path}")


if __name__ == '__main__':
    main()
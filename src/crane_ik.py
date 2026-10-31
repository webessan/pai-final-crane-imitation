#!/usr/bin/env python3
"""CRANE+ V2 の順運動学（関節角度 → 手先の位置）と逆運動学（手先の位置 → 関節角度）。

座標はロボットの土台（crane_plus_base）が原点。単位は m と rad。
  x: Gazebo 画面の手前方向,  y: キューブ側がマイナス,  z: 上
「手先」とは、グリッパの指でキューブを挟む点（掴む点）のこと。

腕の形（crane_plus.xacro の値から）:
  土台 → 肩(joint2) の高さ : 0.0454 + 0.026 = 0.0714
  肩 → ひじ(joint3)        : 0.083
  ひじ → 手首(joint4)      : 0.0935
  手首 → 掴む点            : 0.134（action_publisher.py で掴めた角度から求めた値）

関節の向き（腕が伸びる平面の中で、真上からの角度で考える）:
  上腕の角度   θ1 = -j2
  前腕の角度   θ2 = θ1 + j3
  手先の角度   φ  = θ2 + j4   （φ = π で真下向き）
  腕が伸びる水平方向 = (-cos j1, -sin j1)

python3 crane_ik.py で、計算の自己チェックを表示する。
"""
import math

H0 = 0.0714   # 土台 → 肩の高さ
L1 = 0.083    # 肩 → ひじ
L2 = 0.0935   # ひじ → 手首
L3 = 0.134    # 手首 → 掴む点

LIMITS = [(-2.618, 2.618), (-1.824, 1.783), (-2.556, 2.454), (-1.840, 1.768)]

# 定位置（プリセット姿勢）: キューブの元の位置 (0, -0.10) の真上、掴む点の高さ 9cm
# 自動デモ・モデルの実行・テレオペ（P キー）で共通に使う
PRESET_X, PRESET_Y, PRESET_Z, PRESET_PHI = 0.0, -0.10, 0.09, 2.5
GRASP_Z = 0.013   # 掴む点の高さ（キューブの中心）


def fk(q):
    """関節角度 [j1, j2, j3, j4] → 掴む点の位置 (x, y, z) と手先の角度 φ"""
    j1, j2, j3, j4 = q[:4]
    t1 = -j2
    t2 = t1 + j3
    phi = t2 + j4
    r = L1 * math.sin(t1) + L2 * math.sin(t2) + L3 * math.sin(phi)
    z = H0 + L1 * math.cos(t1) + L2 * math.cos(t2) + L3 * math.cos(phi)
    x = -r * math.cos(j1)
    y = -r * math.sin(j1)
    return x, y, z, phi


def ik(x, y, z, phi=3.0):
    """掴む点の位置 (x, y, z) と手先の角度 φ → 関節角度 [j1, j2, j3, j4]
    届かない位置や可動範囲外なら None を返す。"""
    r = math.hypot(x, y)
    j1 = math.atan2(-y, -x)
    # 手首の位置（肩から見た、水平距離と高さ）
    wr = r - L3 * math.sin(phi)
    wz = z - H0 - L3 * math.cos(phi)
    d2 = wr * wr + wz * wz
    c3 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c3) > 1.0:
        return None                      # 遠すぎる / 近すぎる
    j3 = math.acos(c3)                   # ひじはキューブ側に曲げる（プラス）
    t1 = math.atan2(wr, wz) - math.atan2(L2 * math.sin(j3), L1 + L2 * math.cos(j3))
    t2 = t1 + j3
    q = [j1, -t1, j3, phi - t2]
    for v, (lo, hi) in zip(q, LIMITS):
        if not (lo <= v <= hi):
            return None
    return q


PRESET_Q = ik(PRESET_X, PRESET_Y, PRESET_Z, PRESET_PHI)   # 定位置の関節角度


def preset_steps():
    """初期姿勢（真上）から定位置へ動かすときの経由点。
    まず j1 だけ回し、次に残りの関節をまとめて動かす（途中で 9cm より下がらない）"""
    return [[PRESET_Q[0], 0.0, 0.0, 0.0], list(PRESET_Q)]


if __name__ == '__main__':
    print("自己チェック")
    print("  定位置の関節角度:", [round(v, 3) for v in PRESET_Q])
    known = [1.60, 0.10, 1.75, 1.35]
    x, y, z, phi = fk(known)
    print(f"  掴めた角度 {known} の手先: x={x:+.3f} y={y:+.3f} z={z:+.3f} φ={phi:.2f}")
    print("  （キューブ中心は x=+0.000 y=-0.100 z=+0.013 付近のはず）")
    q = ik(0.0, -0.10, 0.013, 3.0)
    print("  キューブ中心を逆運動学で解くと:", [round(v, 3) for v in q])
    for dx in (-0.05, 0.0, 0.05):
        for dy in (-0.05, 0.0, 0.05):
            q = ik(dx, -0.10 + dy, 0.013, 3.0)
            back = fk(q) if q else None
            ok = q is not None and abs(back[0] - dx) < 1e-6 and abs(back[1] - (-0.10 + dy)) < 1e-6
            print(f"  ずれ ({dx:+.2f},{dy:+.2f}) → "
                  + (str([round(v, 2) for v in q]) if q else "届かない")
                  + ("  往復OK" if ok else ""))

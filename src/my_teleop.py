#!/usr/bin/env python3
"""CRANE+ V2 キーボードテレオペ（改良版）

physicalai_teleop/teleop_keyboard.py をもとに、次の点を改良した:
  1. 目標角度が「実際の関節角度 ±0.2rad」より先に行かないようにした
     → 机や関節の限界で止まっていても、逆方向キーですぐ動く
     → キー押しっぱなしで目標だけ先走ることがなく、反応の遅れが減る
  2. 目標角度を各関節の可動範囲内に収めるようにした
  3. キー入力待ちを 0.02 秒ごとにして、CPU を使い切らないようにした
  4. 関節の選択順を joint1 → joint2 → joint3 → joint4 の自然な順にした
  5. 現在の目標角度を画面の1行に表示し続ける
  6. グリッパの「閉じる」を 0.35rad（自動プログラムで掴めた値）にした
  7. X（初期姿勢に戻す）ではグリッパを閉じる（掴んだキューブを持ち上げられる）
  8. N で選んだ関節の角度を数値で入力できる（その角度までゆっくり動く）
     → 数値で大まかに近づけてから、W/S で微調整する使い方を想定
  9. R を押すたびに、決まった姿勢へ1関節ずつ自動で近づける（プリセット）
     j1 → j2 → j3 → j4 の順。4回押し終わったら通常のテレオペで微調整する

使い方:
  python3 my_teleop.py

キー操作:
  Q/E : 関節を選ぶ (1-4)
  W/S : 選んだ関節を + / -
  N   : 選んだ関節の角度を数値で入力（例: 1.6 → Enter。空のまま Enter で取り消し）
  R   : プリセット姿勢へ1関節ずつ進める（1回目でグリッパも開く。X で最初からやり直し）
  O   : グリッパを開く
  C   : グリッパを閉じる
  X   : 初期姿勢（全関節0）に戻し、グリッパを閉じる
  ESC : 終了
"""
import select
import sys
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ['crane_plus_joint1', 'crane_plus_joint2',
              'crane_plus_joint3', 'crane_plus_joint4']
HAND_JOINT = 'crane_plus_joint_hand'

# 各関節の可動範囲 [rad]（crane_plus.urdf.xacro の値から計算）
LIMITS = {
    'crane_plus_joint1': (-2.618, 2.618),
    'crane_plus_joint2': (-1.824, 1.783),
    'crane_plus_joint3': (-2.556, 2.454),
    'crane_plus_joint4': (-1.840, 1.768),
}
ARM_INIT = [0.0, 0.0, 0.0, 0.0]
GRIPPER_OPEN = -0.5236   # -30°
GRIPPER_CLOSE = 0.35     # 自動プログラムで掴めた値
ARM_STEP = 0.05          # 1回押したときの変化量 [rad]
MAX_LEAD = 0.2           # 目標が実際の角度より先に行ける最大量 [rad]
GOAL_STEP = 0.05         # 数値入力のとき 1回(0.05秒)に動かす量 [rad] → 約1rad/秒
GOAL_LEAD = 0.6          # 数値入力・R のとき、目標が実際の角度より先に行ける最大量 [rad]
                         # （MAX_LEAD より大きくして、自動で動く部分を速くする）

# R キーで順に進めるプリセット（関節番号 0〜3, 角度 [rad]）
PRESET = [(0, 1.39), (1, 0.42), (2, 2.12), (3, 1.05)]


class TeleopNode(Node):
    def __init__(self):
        super().__init__('crane_plus_my_teleop')
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.on_joint, 10)

        self.lock = threading.Lock()
        self.target = ARM_INIT[:]
        self.gripper = GRIPPER_OPEN
        self.actual = None   # 実際の関節角度（/joint_states から）
        self.goal = [None] * 4  # 数値入力で指定された行き先（なければ None）
        self.selected = 0
        self.preset_idx = 0     # 次に R で進めるプリセットの番号
        self.running = True

        print(__doc__.split('キー操作:')[1])
        self.thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.thread.start()
        self.create_timer(0.05, self.publish_commands)  # 20Hz

    def on_joint(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in ARM_JOINTS):
            with self.lock:
                self.actual = [pos[j] for j in ARM_JOINTS]

    def step(self, sign):
        with self.lock:
            i = self.selected
            j = ARM_JOINTS[i]
            self.goal[i] = None  # キーで動かしたら数値入力の行き先は取り消す
            new = self.target[i] + sign * ARM_STEP
            # 実際の角度から MAX_LEAD 以上は先に行かない
            if self.actual is not None:
                new = min(max(new, self.actual[i] - MAX_LEAD),
                          self.actual[i] + MAX_LEAD)
            # 可動範囲内に収める
            lo, hi = LIMITS[j]
            self.target[i] = min(max(new, lo), hi)

    def show(self):
        with self.lock:
            parts = []
            for i, v in enumerate(self.target):
                mark = '>' if i == self.selected else ' '
                parts.append(f"{mark}j{i + 1}:{v:+.2f}")
            g = 'OPEN ' if self.gripper == GRIPPER_OPEN else 'CLOSE'
        sys.stdout.write("\r" + "  ".join(parts) + f"   hand:{g}   ")
        sys.stdout.flush()

    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            self.show()
            while self.running:
                # 0.02秒だけ待つ（待ち時間0だとCPUを使い切ってしまう）
                if not select.select([sys.stdin], [], [], 0.02)[0]:
                    # キーが押されていなくても、数値入力で動いている間は表示を更新
                    if any(g is not None for g in self.goal):
                        self.show()
                    continue
                key = sys.stdin.read(1)
                k = key.lower()
                if k == 'q':
                    self.selected = (self.selected - 1) % 4
                elif k == 'e':
                    self.selected = (self.selected + 1) % 4
                elif k == 'w':
                    self.step(+1)
                elif k == 's':
                    self.step(-1)
                elif k == 'n':
                    self.numeric_input(fd, old)
                elif k == 'r':
                    self.preset_step()
                elif k == 'o':
                    self.gripper = GRIPPER_OPEN
                elif k == 'c':
                    self.gripper = GRIPPER_CLOSE
                elif k == 'x':
                    with self.lock:
                        self.target = ARM_INIT[:]
                        self.goal = [None] * 4
                    self.gripper = GRIPPER_CLOSE  # X ではグリッパを閉じる
                    self.preset_idx = 0           # プリセットも最初からに戻す
                elif ord(key) == 27:  # ESC
                    self.running = False
                    break
                self.show()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print()

    def preset_step(self):
        """R キー: プリセットの次の1関節を、決まった角度へゆっくり動かす"""
        if self.preset_idx >= len(PRESET):
            print("\n  プリセットは完了しています（X で最初からやり直せます）")
            return
        if self.preset_idx == 0:
            self.gripper = GRIPPER_OPEN  # 最初に R を押したときにグリッパを開く
        i, value = PRESET[self.preset_idx]
        with self.lock:
            self.goal[i] = value
        self.selected = i  # 動かした関節を選んだ状態にする（すぐ W/S で微調整できる）
        self.preset_idx += 1
        print(f"\n  [R {self.preset_idx}/{len(PRESET)}] j{i + 1} を {value:+.2f} へ動かします")
        if self.preset_idx == len(PRESET):
            print("  プリセット完了。ここからは通常のテレオペで微調整してください")

    def numeric_input(self, fd, cbreak_old):
        """選んだ関節の行き先を数値で受け取る"""
        i = self.selected
        # 一時的に通常の入力モード（Enter で確定する）に戻す
        termios.tcsetattr(fd, termios.TCSADRAIN, cbreak_old)
        try:
            lo, hi = LIMITS[ARM_JOINTS[i]]
            text = input(f"\n  j{i + 1} の角度を入力 [rad]（{lo:+.2f} ～ {hi:+.2f}）: ")
        finally:
            tty.setcbreak(fd)
        text = text.strip()
        if not text:
            print("  取り消しました")
            return
        try:
            value = float(text)
        except ValueError:
            print(f"  数値ではありません: {text}")
            return
        value = min(max(value, lo), hi)
        with self.lock:
            self.goal[i] = value
        print(f"  j{i + 1} を {value:+.2f} へ動かします（W/S を押すと取り消し）")

    def advance_goals(self):
        """数値入力の行き先へ、目標を少しずつ近づける（publish のたびに呼ぶ）"""
        with self.lock:
            for i, g in enumerate(self.goal):
                if g is None:
                    continue
                diff = g - self.target[i]
                if abs(diff) <= GOAL_STEP:
                    new = g
                else:
                    new = self.target[i] + GOAL_STEP * (1 if diff > 0 else -1)
                # 実際の角度から GOAL_LEAD 以上は先に行かない
                if self.actual is not None:
                    new = min(max(new, self.actual[i] - GOAL_LEAD),
                              self.actual[i] + GOAL_LEAD)
                self.target[i] = new
                # 目標・実物ともに行き先に着いたら終了
                if (abs(self.target[i] - g) < 1e-6 and self.actual is not None
                        and abs(self.actual[i] - g) < 0.02):
                    self.goal[i] = None

    def _msg(self, names, positions):
        jt = JointTrajectory()
        jt.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        return jt

    def publish_commands(self):
        self.advance_goals()
        with self.lock:
            target = self.target[:]
        self.arm_pub.publish(self._msg(ARM_JOINTS, target))
        self.gripper_pub.publish(self._msg([HAND_JOINT], [self.gripper]))


def main():
    rclpy.init()
    node = TeleopNode()
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
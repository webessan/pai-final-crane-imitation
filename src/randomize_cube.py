#!/usr/bin/env python3
"""キューブを「元の位置 ± 指定した幅」のランダムな位置に置き直す。

使い方:
    python3 randomize_cube.py          # ずれ幅 1cm（既定）
    python3 randomize_cube.py 0.02     # ずれ幅 2cm
    python3 randomize_cube.py 0        # ずらさず元の位置に戻す
"""
import random
import shutil
import subprocess
import sys

# キューブの元の位置（table.sdf の wood cube の位置）
BASE_X = 0.0
BASE_Y = -0.1
Z = 1.0035


def main():
    # ずれ幅 [m]。コマンドの引数で指定できる（省略時は 0.01 = 1cm）
    delta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01

    # 元の位置を中心に、x と y をそれぞれ -delta ～ +delta の範囲でずらす
    x = round(BASE_X + random.uniform(-delta, delta), 4)
    y = round(BASE_Y + random.uniform(-delta, delta), 4)

    cmd_bin = shutil.which("ign") or shutil.which("gz")
    if not cmd_bin:
        print("[ERROR] 'ign' または 'gz' コマンドが見つかりません。")
        return
    msg_prefix = "ignition.msgs" if "ign" in cmd_bin else "gz.msgs"

    # 向き（orientation）も明示して、キューブがまっすぐ置かれるようにする
    req_payload = (
        f'name: "wood_cube_5cm", '
        f'position: {{x: {x}, y: {y}, z: {Z}}}, '
        f'orientation: {{x: 0, y: 0, z: 0, w: 1}}'
    )
    cmd = [
        cmd_bin, "service",
        "-s", "/world/default/set_pose",
        "--reqtype", f"{msg_prefix}.Pose",
        "--reptype", f"{msg_prefix}.Boolean",
        "--timeout", "1000",
        "--req", req_payload,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if "true" in res.stdout.lower():
        dx_cm = (x - BASE_X) * 100
        dy_cm = (y - BASE_Y) * 100
        print(f"[SUCCESS] Cube moved to X:{x}, Y:{y}  "
              f"(元の位置からのずれ: x {dx_cm:+.1f}cm, y {dy_cm:+.1f}cm)")
    else:
        print(f"[FAILED] Response: {res.stdout.strip()} {res.stderr.strip()}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
三星 Galaxy S10 (SM-G973N) 透明环氧后盖 —— 开口浇注模盒生成器 (FDM 打印, 适配拓竹 A1)

思路:
    * 模盒底面 = 后盖【内侧面】(贴手机那一面) 的曲面形状 -> 贴合度由模具保证
    * 树脂倒入后自然流平 = 后盖【外侧面】是水平的平面
    * 摄像头窗口 = 模盒里的一根立柱, 树脂绕过它, 脱模后就是镂空窗口
    * 结果: 中间厚度 = --thickness, 长边两侧比中间厚 --curve (外侧是平的)

坐标系 (俯视模盒 = 从背面看后盖外表面, 和 s10_back_cover.py 一致):
    X 向右, Y 向上(手机顶部), Z 向上

用法:
    .venv/bin/python s10_mold.py                  # 完整模盒
    .venv/bin/python s10_mold.py --test           # 只做顶部 50mm 的试配小模盒, 省树脂
    .venv/bin/python s10_mold.py --thickness 1.8  # 中间厚度改成 1.8mm
"""
import argparse
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

from s10_back_cover import (
    CAM_H, CAM_R, CAM_TOP, CAM_W, CAM_X, CORNER_R, COVER_L, COVER_W,
    CURVE, CURVE_ZONE, TEST_LENGTH, prism, rounded_rect, write_binary_stl,
)

# ----------------------------------------------------------------------------
# 模盒参数 (单位 mm)
# ----------------------------------------------------------------------------
CAST_THICKNESS = 1.5   # 后盖中间厚度 (环氧偏脆, 比打印版略厚)
WALL = 4.0             # 模盒侧壁厚度
BASE = 3.0             # 模盒底板最薄处厚度 (底面最低点以下)
FREEBOARD = 4.0        # 树脂液面以上留的壁高, 防溢出
POST_OVER = 0.6        # 窗口立柱高出液面的量


def floor_profile(width, length, base, curve, zone, n=160):
    """模盒底面: 中间高, 两侧低 (= 后盖内侧面翻转). 沿 Y 拉伸成实体."""
    half = width / 2
    xs = np.linspace(-half - 3, half + 3, n)
    u = np.clip((np.abs(xs) - (half - zone)) / zone, 0.0, 1.0)
    f = base + curve * (1.0 - u ** 2)
    bottom = np.array([[xs[0], -1.0], [xs[-1], -1.0]])
    top = np.stack([xs[::-1], f[::-1]], axis=1)
    poly = np.vstack([bottom, top])
    ext_len = length + 10.0
    solid = Manifold.extrude(CrossSection([poly]), ext_len)
    solid = solid.rotate((90, 0, 0)).translate((0, ext_len / 2, 0))
    return solid


def build_mold(a):
    W, L = a.width, a.length
    top_y = L / 2

    fill_h = a.base + a.curve + a.thickness       # 树脂液面高度 (外侧平面)
    wall_h = fill_h + a.freeboard                 # 侧壁顶高

    # 浇注区轮廓 (后盖外形)
    cavity_cs = rounded_rect(W, L, a.corner)
    if a.test:
        clip = CrossSection.square((W + 2, TEST_LENGTH), center=True).translate(
            (0, top_y - TEST_LENGTH / 2)
        )
        cavity_cs = cavity_cs ^ clip

    outer_cs = cavity_cs.offset(a.wall, JoinType.Round, circular_segments=96)
    block = Manifold.extrude(outer_cs, wall_h)

    cavity = Manifold.extrude(cavity_cs, wall_h + 1).translate((0, 0, 0))
    floor = floor_profile(W, L, a.base, a.curve, CURVE_ZONE)
    post = prism(
        rounded_rect(a.cam_w, a.cam_h, a.cam_r, 64), 0, fill_h + POST_OVER,
        x=a.cam_x, y=top_y - a.cam_top,
    )

    void = cavity - floor - post        # 能装树脂的空间
    mold = block - void

    # 树脂体积 (空间 ∩ 液面以下)
    level_box = prism(CrossSection.square((600, 600), center=True), -1, fill_h)
    resin = void ^ level_box
    return mold, resin, fill_h, wall_h


def main():
    ap = argparse.ArgumentParser(description="Galaxy S10 epoxy back-cover open mold")
    ap.add_argument("--width", type=float, default=COVER_W)
    ap.add_argument("--length", type=float, default=COVER_L)
    ap.add_argument("--corner", type=float, default=CORNER_R)
    ap.add_argument("--thickness", type=float, default=CAST_THICKNESS, help="后盖中间厚度")
    ap.add_argument("--curve", type=float, default=CURVE, help="长边比中间多厚的量 (内侧面弯起高度)")
    ap.add_argument("--cam-w", type=float, default=CAM_W)
    ap.add_argument("--cam-h", type=float, default=CAM_H)
    ap.add_argument("--cam-r", type=float, default=CAM_R)
    ap.add_argument("--cam-x", type=float, default=CAM_X)
    ap.add_argument("--cam-top", type=float, default=CAM_TOP)
    ap.add_argument("--wall", type=float, default=WALL)
    ap.add_argument("--base", type=float, default=BASE)
    ap.add_argument("--freeboard", type=float, default=FREEBOARD)
    ap.add_argument("--test", action="store_true", help="只做顶部 50mm 试配模盒")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    name = "s10_mold_TEST" if a.test else "s10_mold"
    path = Path(a.out) if a.out else out_dir / f"{name}.stl"

    mold, resin, fill_h, wall_h = build_mold(a)
    if mold.is_empty() or resin.is_empty():
        raise SystemExit("生成失败: 结果为空, 请检查参数")

    verts, tris = write_binary_stl(mold, path)
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    vol_ml = resin.volume() / 1000.0

    print(f"STL: {path}")
    print(f"模盒尺寸: X {mx[0] - mn[0]:.1f}  Y {mx[1] - mn[1]:.1f}  Z {mx[2] - mn[2]:.1f} mm"
          f"   三角面 {len(tris)}   约 {mold.volume() * 1.24e-3:.0f} g PLA")
    print(f"液面高度(距模盒底): {fill_h:.2f} mm   侧壁高: {wall_h:.2f} mm")
    print(f"所需环氧: 约 {vol_ml:.1f} mL ≈ {vol_ml * 1.1:.1f} g (A+B 总量, 建议多配 15%: {vol_ml * 1.1 * 1.15:.0f} g)")
    rb = resin.bounding_box()
    print(f"后盖预期: 中间厚 {a.thickness:.2f} mm, 长边处厚 {a.thickness + a.curve:.2f} mm "
          f"(树脂 Z 范围 {rb[2]:.2f} ~ {rb[5]:.2f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
三星 Galaxy S10 (韩版 SM-G973N) 后盖 3D 打印模型生成器
依赖: numpy, manifold3d   (已装在 ./.venv)

坐标系 (从手机背面看, 正对后盖外表面):
    X 向右, Y 向上(手机顶部), Z 为厚度方向
    外表面朝 -Z (打印时贴热床), 长边两侧向 +Z 弯起(模拟 S10 的曲面玻璃)

用法:
    .venv/bin/python s10_back_cover.py                 # 生成完整后盖
    .venv/bin/python s10_back_cover.py --test          # 只打印顶部 50mm, 用来试摄像头孔位
    .venv/bin/python s10_back_cover.py --curve 0       # 平板, 打印不需要支撑
    .venv/bin/python s10_back_cover.py --thickness 1.0 --camera-x -7 --camera-top 12

!! 注意: 摄像头/闪光灯孔位和后盖外形尺寸是按公开资料估算的默认值,
   请用游标卡尺量你自己的手机, 然后改下面的参数, 或用 --test 先打一小块试配.
"""
import argparse
import struct
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

# ----------------------------------------------------------------------------
# 参数 (单位 mm)
# ----------------------------------------------------------------------------
# 整机: 149.9 x 70.4 x 7.8 (SM-G973N 与国际版外形一致)
COVER_W = 69.8        # 后盖宽度  (机身 70.4, 留 ~0.3mm 单边间隙; 请按实测调整)
COVER_L = 149.3       # 后盖长度  (机身 149.9)
CORNER_R = 9.5        # 四角圆角半径
THICKNESS = 1.2       # 后盖壁厚 (<=1.5 比较不影响无线充电)
CURVE = 1.2           # 长边两侧向正面弯起的高度, 0 = 平板
CURVE_ZONE = 6.0      # 弯曲区宽度 (从长边向内多少 mm 开始弯)

# 主摄像头长条孔 (从背面看)
CAM_W, CAM_H, CAM_R = 34.5, 10.5, 4.0   # 宽, 高, 圆角
CAM_X = -7.5                            # 孔中心 X (相对后盖中心, 右为正)
CAM_TOP = 12.5                          # 孔中心距后盖顶边的距离

# 闪光灯 + 心率传感器孔 (摄像头右侧)
FLASH_W, FLASH_H, FLASH_R = 5.5, 14.0, 2.7
FLASH_X = 14.5
FLASH_TOP = 12.5

TEST_LENGTH = 50.0    # --test 时保留的顶部长度


def rounded_rect(w, h, r, segs=96):
    r = min(r, w / 2 - 1e-3, h / 2 - 1e-3)
    return CrossSection.square((w - 2 * r, h - 2 * r), center=True).offset(
        r, JoinType.Round, circular_segments=segs
    )


def prism(cs, z0, z1, x=0.0, y=0.0):
    return Manifold.extrude(cs, z1 - z0).translate((x, y, z0))


def build_plate(width, length, thickness, curve, zone, n=120):
    """长边曲面板: XZ 截面沿 Y 拉伸."""
    half = width / 2
    x0 = half - zone
    xs = np.linspace(-half, half, n)
    u = np.clip((np.abs(xs) - x0) / zone, 0.0, 1.0)
    zo = curve * u ** 2                       # 外表面高度
    zi = zo + thickness                       # 内表面高度
    bottom = np.stack([xs, zo], axis=1)       # 左 -> 右
    top = np.stack([xs[::-1], zi[::-1]], axis=1)  # 右 -> 左
    poly = np.vstack([bottom, top])
    section = CrossSection([poly])
    ext_len = length + 4.0
    solid = Manifold.extrude(section, ext_len)      # 沿自身 Z 拉伸
    solid = solid.rotate((90, 0, 0))                # 多边形 y -> 世界 z, 拉伸方向 -> -Y
    solid = solid.translate((0, ext_len / 2, 0))
    return solid


def build_cover(args):
    W, L = args.width, args.length
    plate = build_plate(W, L, args.thickness, args.curve, CURVE_ZONE)

    # 圆角外形裁剪
    outline = prism(rounded_rect(W, L, args.corner), -5, 20)
    cover = plate ^ outline  # 交集

    top_y = L / 2
    # 摄像头孔
    cam = prism(
        rounded_rect(CAM_W, CAM_H, CAM_R, 64), -5, 20,
        x=args.camera_x, y=top_y - args.camera_top,
    )
    # 闪光灯 + 心率孔
    flash = prism(
        rounded_rect(FLASH_W, FLASH_H, FLASH_R, 48), -5, 20,
        x=args.flash_x, y=top_y - FLASH_TOP,
    )
    cover = cover - cam - flash

    if args.test:
        cut = prism(
            CrossSection.square((W + 10, TEST_LENGTH), center=True), -5, 20,
            y=top_y - TEST_LENGTH / 2,
        )
        cover = cover ^ cut
    return cover


def write_binary_stl(manifold, path):
    mesh = manifold.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    tris = np.asarray(mesh.tri_verts).astype(np.int64)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0)
    with open(path, "wb") as f:
        f.write(b"Samsung Galaxy S10 SM-G973N back cover".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"] = normals
        rec["v"] = np.stack([v0, v1, v2], axis=1)
        f.write(rec.tobytes())
    return verts, tris


def write_preview_svg(args, path):
    """背面视角的 2D 示意图, 方便对着手机核对孔位."""
    W, L, s = args.width, args.length, 6  # s: px / mm
    pad = 10

    def sx(x): return (x + W / 2) * s + pad
    def sy(y): return (L / 2 - y) * s + pad

    def rrect(cx, cy, w, h, r, fill, stroke="#222"):
        return (f'<rect x="{sx(cx - w / 2):.1f}" y="{sy(cy + h / 2):.1f}" width="{w * s:.1f}" '
                f'height="{h * s:.1f}" rx="{r * s:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    top = L / 2
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * s + 2 * pad:.0f}" height="{L * s + 2 * pad:.0f}">',
        rrect(0, 0, W, L, args.corner, "#d8dee9"),
        rrect(args.camera_x, top - args.camera_top, CAM_W, CAM_H, CAM_R, "#fff"),
        rrect(args.flash_x, top - FLASH_TOP, FLASH_W, FLASH_H, FLASH_R, "#fff"),
        f'<text x="{pad + 6}" y="{pad + 18}" font-size="14" fill="#333">S10 back (viewed from rear) '
        f'{W}x{L}mm</text>',
        "</svg>",
    ]
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Galaxy S10 (SM-G973N) back cover STL generator")
    ap.add_argument("--width", type=float, default=COVER_W)
    ap.add_argument("--length", type=float, default=COVER_L)
    ap.add_argument("--corner", type=float, default=CORNER_R)
    ap.add_argument("--thickness", type=float, default=THICKNESS)
    ap.add_argument("--curve", type=float, default=CURVE, help="长边弯起高度, 0=平板")
    ap.add_argument("--camera-x", type=float, default=CAM_X)
    ap.add_argument("--camera-top", type=float, default=CAM_TOP)
    ap.add_argument("--flash-x", type=float, default=FLASH_X)
    ap.add_argument("--test", action="store_true", help="只生成顶部 50mm 试配件")
    ap.add_argument("--out", default=None, help="输出 STL 路径")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    name = "s10_back_cover_TEST" if args.test else "s10_back_cover"
    stl_path = Path(args.out) if args.out else out_dir / f"{name}.stl"

    cover = build_cover(args)
    if cover.is_empty():
        raise SystemExit("生成失败: 结果为空, 请检查参数")

    verts, tris = write_binary_stl(cover, stl_path)
    write_preview_svg(args, out_dir / "preview_back_view.svg")

    mn, mx = verts.min(axis=0), verts.max(axis=0)
    print(f"STL: {stl_path}")
    print(f"三角面: {len(tris)}   体积: {cover.volume():.0f} mm^3  (约 {cover.volume() * 1.24e-3:.1f} g PLA)")
    print(f"包围盒: X {mx[0] - mn[0]:.2f}  Y {mx[1] - mn[1]:.2f}  Z {mx[2] - mn[2]:.2f} mm")
    print(f"预览图: {out_dir / 'preview_back_view.svg'}")


if __name__ == "__main__":
    main()

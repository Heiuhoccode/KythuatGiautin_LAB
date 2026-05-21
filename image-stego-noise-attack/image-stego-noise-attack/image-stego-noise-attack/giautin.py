"""
giautin.py - Nhúng watermark vào ảnh gốc (DWT + SVD)
Sử dụng: python giautin.py <anh_goc> <watermark> [anh_output] [alpha]
"""

import sys
import os
import pickle
import numpy as np
import cv2
from thuvien import (
    doc_anh, luu_anh,
    nhung_watermark,
    chuan_hoa_kich_thuoc,
    tinh_psnr, tinh_ssim,
)


def main():
    # ─── Đọc tham số dòng lệnh ───────────────────
    if len(sys.argv) < 3:
        print("Cách dùng: python giautin.py <anh_goc> <watermark> [anh_output] [alpha]")
        sys.exit(1)

    duong_dan_goc = sys.argv[1]
    duong_dan_wm  = sys.argv[2]
    duong_dan_out = sys.argv[3] if len(sys.argv) > 3 else "stego.png"
    alpha         = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

    # ─── Đọc ảnh ─────────────────────────────────
    print(f"[1] Đọc ảnh gốc    : {duong_dan_goc}")
    anh_goc = doc_anh(duong_dan_goc)

    print(f"[2] Đọc watermark  : {duong_dan_wm}")
    watermark = doc_anh(duong_dan_wm)

    print(f"    Kích thước ảnh gốc : {anh_goc.shape}")
    print(f"    Kích thước watermark: {watermark.shape}")
    print(f"    Hệ số nhúng (alpha) : {alpha}")

    # ─── Kiểm tra: watermark không được lớn hơn ảnh gốc ──
    #  Nếu lớn hơn, resize watermark về bằng nửa ảnh gốc
    h_goc, w_goc = anh_goc.shape[:2]
    h_wm,  w_wm  = watermark.shape[:2]
    max_wm_h = h_goc // 2
    max_wm_w = w_goc // 2
    if h_wm > max_wm_h or w_wm > max_wm_w:
        print(f"    [!] Watermark quá lớn, resize về ({max_wm_h}, {max_wm_w})")
        watermark = chuan_hoa_kich_thuoc(watermark, (max_wm_h, max_wm_w))

    # ─── Nhúng watermark ─────────────────────────
    print("[3] Tiến hành nhúng watermark (DWT + SVD)...")
    anh_watermarked, keys = nhung_watermark(anh_goc, watermark, alpha=alpha)

    # ─── Lưu ảnh kết quả ─────────────────────────
    print(f"[4] Lưu ảnh đã nhúng watermark: {duong_dan_out}")
    luu_anh(duong_dan_out, anh_watermarked)

    # ─── Lưu keys để dùng khi trích xuất ────────
    ten_keys = os.path.splitext(duong_dan_out)[0] + "_keys.pkl"
    with open(ten_keys, "wb") as f:
        pickle.dump(keys, f)
    print(f"[5] Đã lưu khóa trích xuất  : {ten_keys}")

    # ─── Đánh giá chất lượng ảnh ─────────────────
    p = tinh_psnr(anh_goc, anh_watermarked)
    s = tinh_ssim(anh_goc, anh_watermarked)
    print("\n── Đánh giá chất lượng ảnh ──────────────────────")
    print(f"   PSNR : {p:.4f} dB  (>30 dB → không cảm nhận được)")
    print(f"   SSIM : {s:.6f}  (gần 1 → chất lượng tốt)")
    print("─────────────────────────────────────────────────")
    print("\n[DONE] Nhúng watermark thành công!\n")


if __name__ == "__main__":
    main()

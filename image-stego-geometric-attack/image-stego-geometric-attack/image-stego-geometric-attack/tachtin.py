"""
tachtin.py - Trích xuất watermark từ ảnh nghi ngờ (DWT + SVD)
Sử dụng: python tachtin.py <anh_nghi_ngo> <keys_file> [wm_output] [wm_goc]

Ví dụ:
    python tachtin.py watermarked.png watermarked_keys.pkl wm_recovered.png wm.png
"""

import sys
import os
import pickle
import numpy as np
from thuvien import (
    doc_anh, luu_anh,
    trich_xuat_watermark,
    tinh_nc, tinh_psnr, tinh_ssim,
    danh_gia_toan_dien,
)


def main():
    # ─── Đọc tham số dòng lệnh ───────────────────
    if len(sys.argv) < 3:
        print("Cách dùng: python tachtin.py <anh_nghi_ngo> <keys_file> [wm_output] [wm_goc]")
        print("Ví dụ    : python tachtin.py watermarked.png watermarked_keys.pkl wm_recovered.png wm.png")
        sys.exit(1)

    duong_dan_anh  = sys.argv[1]
    duong_dan_keys = sys.argv[2]
    duong_dan_out  = sys.argv[3] if len(sys.argv) > 3 else "wm_recovered.png"
    duong_dan_wm_goc = sys.argv[4] if len(sys.argv) > 4 else None

    # ─── Đọc ảnh nghi ngờ ────────────────────────
    print(f"[1] Đọc ảnh nghi ngờ : {duong_dan_anh}")
    anh_nghi_ngo = doc_anh(duong_dan_anh)
    print(f"    Kích thước: {anh_nghi_ngo.shape}")

    # ─── Đọc keys ────────────────────────────────
    print(f"[2] Nạp khóa trích xuất: {duong_dan_keys}")
    if not os.path.exists(duong_dan_keys):
        print(f"[LỖI] Không tìm thấy file keys: {duong_dan_keys}")
        sys.exit(1)
    with open(duong_dan_keys, "rb") as f:
        keys = pickle.load(f)

    alpha = keys.get("alpha", "?")
    wavelet = keys.get("wavelet", "haar")
    print(f"    Alpha   : {alpha}")
    print(f"    Wavelet : {wavelet}")

    # ─── Trích xuất watermark ────────────────────
    print("[3] Tiến hành trích xuất watermark (DWT + SVD)...")
    wm_phuc_hoi = trich_xuat_watermark(anh_nghi_ngo, keys)

    # ─── Lưu watermark khôi phục ─────────────────
    print(f"[4] Lưu watermark khôi phục: {duong_dan_out}")
    luu_anh(duong_dan_out, wm_phuc_hoi)

    # ─── Đánh giá nếu có watermark gốc ──────────
    if duong_dan_wm_goc and os.path.exists(duong_dan_wm_goc):
        print(f"[5] So sánh với watermark gốc: {duong_dan_wm_goc}")
        wm_goc = doc_anh(duong_dan_wm_goc)

        # Cắt về cùng kích thước để so sánh
        h = min(wm_goc.shape[0], wm_phuc_hoi.shape[0])
        w = min(wm_goc.shape[1], wm_phuc_hoi.shape[1])
        wm_goc_crop = wm_goc[:h, :w]
        wm_pr_crop  = wm_phuc_hoi[:h, :w]

        nc = tinh_nc(wm_goc_crop, wm_pr_crop)
        p  = tinh_psnr(wm_goc_crop, wm_pr_crop)
        s  = tinh_ssim(wm_goc_crop, wm_pr_crop)

        print("\n── Đánh giá chất lượng watermark khôi phục ──")
        print(f"   NC   : {nc:.6f}  (gần 1 → khôi phục tốt)")
        print(f"   PSNR : {p:.4f} dB")
        print(f"   SSIM : {s:.6f}")

        if nc >= 0.90:
            print("\n   ✔  Watermark hợp lệ — ảnh có chứa thông tin bản quyền!")
        elif nc >= 0.70:
            print("\n   ⚠  Watermark phát hiện được — có thể đã bị tấn công nhẹ.")
        else:
            print("\n   ✘  Watermark không khớp — ảnh có thể bị làm giả hoặc chỉnh sửa nặng.")
        print("─────────────────────────────────────────────")
    else:
        if duong_dan_wm_goc:
            print(f"[!] Không tìm thấy watermark gốc '{duong_dan_wm_goc}', bỏ qua bước so sánh.")

    print("\n[DONE] Trích xuất watermark thành công!\n")


if __name__ == "__main__":
    main()

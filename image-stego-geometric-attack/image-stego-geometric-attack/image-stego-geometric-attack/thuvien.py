"""
thuvien.py - Thư viện hỗ trợ thủy vân số (Digital Watermarking)
Phương pháp: DWT + SVD
"""

import numpy as np
import cv2
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ─────────────────────────────────────────────
# 1. ĐỌC / LƯU ẢNH
# ─────────────────────────────────────────────

def doc_anh(duong_dan: str, che_do=cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """Đọc ảnh từ đường dẫn, mặc định đọc grayscale."""
    anh = cv2.imread(duong_dan, che_do)
    if anh is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {duong_dan}")
    return anh.astype(np.float64)


def luu_anh(duong_dan: str, anh: np.ndarray) -> None:
    """Lưu ảnh ra file (tự động clip về [0, 255])."""
    anh_luu = np.clip(anh, 0, 255).astype(np.uint8)
    cv2.imwrite(duong_dan, anh_luu)
    print(f"[OK] Đã lưu ảnh: {duong_dan}")


# ─────────────────────────────────────────────
# 2. DWT (Discrete Wavelet Transform)
# ─────────────────────────────────────────────

def dwt_2d(anh: np.ndarray, wavelet: str = "haar") -> tuple:
    """
    Thực hiện DWT 1 cấp trên ảnh 2D.
    Trả về: (LL, LH, HL, HH)
    """
    LL, (LH, HL, HH) = pywt.dwt2(anh, wavelet)
    return LL, LH, HL, HH


def idwt_2d(LL, LH, HL, HH, wavelet: str = "haar") -> np.ndarray:
    """
    Thực hiện IDWT để khôi phục ảnh từ 4 miền con.
    """
    return pywt.idwt2((LL, (LH, HL, HH)), wavelet)


# ─────────────────────────────────────────────
# 3. SVD (Singular Value Decomposition)
# ─────────────────────────────────────────────

def svd(matrix: np.ndarray) -> tuple:
    """
    Phân rã SVD: A = U * S_diag * V^T
    Trả về: (U, S, Vt)  — S là vector singular values
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt


def khoi_phuc_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """
    Khôi phục ma trận từ U, S (vector), Vt.
    A' = U * diag(S) * V^T
    """
    return U @ np.diag(S) @ Vt


# ─────────────────────────────────────────────
# 4. NHÚNG WATERMARK (Embedding)
# ─────────────────────────────────────────────

def nhung_watermark(
    anh_goc: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.01,
    wavelet: str = "haar",
) -> tuple:
    """
    Nhúng watermark vào ảnh gốc theo phương pháp DWT + SVD.

    Tham số:
        anh_goc   : ảnh gốc (2D, float64)
        watermark : ảnh watermark (2D, float64)
        alpha     : hệ số nhúng (0 < alpha << 1)
        wavelet   : loại wavelet ('haar', 'db1', ...)

    Trả về:
        anh_watermarked : ảnh đã nhúng watermark
        keys            : dict chứa thông tin cần thiết để trích xuất
    """
    # ── Bước 1: DWT ảnh gốc ──────────────────
    LL1, LH1, HL1, HH1 = dwt_2d(anh_goc, wavelet)

    # ── Bước 2: DWT watermark ────────────────
    LL_w, LH_w, HL_w, HH_w = dwt_2d(watermark, wavelet)

    # ── Bước 3: SVD trên miền LH1 và LL_w ───
    U1, S1, Vt1 = svd(LH1)
    U2, S_w, Vt2 = svd(LL_w)

    # ── Bước 4: Nhúng singular values ────────
    #   S' = S1 + alpha * S_w
    #   Đảm bảo độ dài khớp nhau
    len_min = min(len(S1), len(S_w))
    S_prime = S1.copy()
    S_prime[:len_min] = S1[:len_min] + alpha * S_w[:len_min]

    # ── Bước 5: Tái tạo miền LH' = U1 * S' * V1^T ──
    LH_prime = khoi_phuc_svd(U1, S_prime, Vt1)

    # ── Bước 6: IDWT → ảnh chứa watermark ───
    anh_watermarked = idwt_2d(LL1, LH_prime, HL1, HH1, wavelet)

    # Lưu các thông số cần cho trích xuất
    keys = {
        "U1": U1, "S1": S1, "Vt1": Vt1,
        "U2": U2, "S_w": S_w, "Vt2": Vt2,
        "alpha": alpha,
        "wavelet": wavelet,
        "len_min": len_min,
        "anh_goc_shape": anh_goc.shape,
        "watermark_shape": watermark.shape,
    }
    return anh_watermarked, keys


# ─────────────────────────────────────────────
# 5. TRÍCH XUẤT WATERMARK (Extraction)
# ─────────────────────────────────────────────

def trich_xuat_watermark(
    anh_nghi_ngo: np.ndarray,
    keys: dict,
) -> np.ndarray:
    """
    Trích xuất watermark từ ảnh nghi ngờ (non-blind: cần keys từ lúc nhúng).

    Tham số:
        anh_nghi_ngo : ảnh cần kiểm tra (2D, float64)
        keys         : dict trả về từ nhung_watermark()

    Trả về:
        watermark_phuc_hoi : ảnh watermark được khôi phục
    """
    wavelet = keys["wavelet"]
    U1      = keys["U1"]
    S1      = keys["S1"]
    U2      = keys["U2"]
    Vt2     = keys["Vt2"]
    alpha   = keys["alpha"]
    len_min = keys["len_min"]

    # ── Bước 1: DWT ảnh nghi ngờ ─────────────
    _, LH_prime, _, _ = dwt_2d(anh_nghi_ngo, wavelet)

    # ── Bước 2: SVD trên LH' ─────────────────
    _, S_prime, _ = svd(LH_prime)

    # ── Bước 3: Tính singular values watermark ──
    #   S_w = (S' - S1) / alpha
    # Số singular values của U2/Vt2 (kích thước truncated SVD gốc)
    k = len(keys["S_w"])  # độ dài S_w lúc nhúng = min(h_LL_w, w_LL_w)
    S_w_recovered = np.zeros(k)
    take = min(len_min, k, len(S_prime), len(S1))
    S_w_recovered[:take] = (S_prime[:take] - S1[:take]) / alpha

    # ── Bước 4: Khôi phục LL_w = U2 * S_w * V2^T ──
    LL_w_recovered = khoi_phuc_svd(U2, S_w_recovered, Vt2)

    # ── IDWT ngược về watermark hoàn chỉnh ───
    watermark_shape = keys["watermark_shape"]
    # Tạo các miền tần số còn lại bằng 0 (vì ta chỉ nhúng vào LL_w)
    dummy = np.zeros_like(LL_w_recovered)
    watermark_phuc_hoi = idwt_2d(LL_w_recovered, dummy, dummy, dummy, wavelet)

    # Cắt về đúng kích thước watermark gốc
    h, w = watermark_shape
    watermark_phuc_hoi = watermark_phuc_hoi[:h, :w]

    return watermark_phuc_hoi


# ─────────────────────────────────────────────
# 6. ĐÁNH GIÁ CHẤT LƯỢNG
# ─────────────────────────────────────────────

def tinh_psnr(anh1: np.ndarray, anh2: np.ndarray) -> float:
    """Tính PSNR (dB) giữa 2 ảnh. Giá trị cao → ảnh gần giống nhau."""
    a1 = np.clip(anh1, 0, 255).astype(np.uint8)
    a2 = np.clip(anh2, 0, 255).astype(np.uint8)
    return psnr(a1, a2, data_range=255)


def tinh_ssim(anh1: np.ndarray, anh2: np.ndarray) -> float:
    """Tính SSIM giữa 2 ảnh. Giá trị gần 1 → ảnh gần giống nhau."""
    a1 = np.clip(anh1, 0, 255).astype(np.uint8)
    a2 = np.clip(anh2, 0, 255).astype(np.uint8)
    return ssim(a1, a2, data_range=255)


def tinh_nc(w_goc: np.ndarray, w_phuc_hoi: np.ndarray) -> float:
    """
    Tính Normalized Correlation (NC) giữa watermark gốc và khôi phục.
    NC gần 1 → watermark khôi phục tốt.
    """
    w1 = w_goc.flatten().astype(np.float64)
    w2 = w_phuc_hoi.flatten().astype(np.float64)
    # Đảm bảo cùng độ dài
    min_len = min(len(w1), len(w2))
    w1, w2 = w1[:min_len], w2[:min_len]
    mau = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))
    if mau == 0:
        return 0.0
    return float(np.sum(w1 * w2) / mau)


def danh_gia_toan_dien(
    anh_goc: np.ndarray,
    anh_watermarked: np.ndarray,
    wm_goc: np.ndarray,
    wm_phuc_hoi: np.ndarray,
) -> None:
    """In bảng đánh giá PSNR, SSIM (ảnh) và NC (watermark)."""
    print("\n" + "=" * 45)
    print("        ĐÁNH GIÁ CHẤT LƯỢNG")
    print("=" * 45)
    p = tinh_psnr(anh_goc, anh_watermarked)
    s = tinh_ssim(anh_goc, anh_watermarked)
    nc = tinh_nc(wm_goc, wm_phuc_hoi)
    print(f"  PSNR (ảnh gốc vs có WM) : {p:.4f} dB")
    print(f"  SSIM (ảnh gốc vs có WM) : {s:.6f}")
    print(f"  NC   (WM gốc vs khôi phục): {nc:.6f}")
    print("=" * 45)


# ─────────────────────────────────────────────
# 7. TIỆN ÍCH: CHUYỂN ĐỔI KÍCH THƯỚC
# ─────────────────────────────────────────────

def chuan_hoa_kich_thuoc(anh: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize ảnh về kích thước target_shape (h, w) bằng interpolation."""
    h, w = target_shape
    return cv2.resize(anh, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
import cv2
import numpy as np
import os
import sys


def geometric_attack(input_path, output_folder):
    """
    Thực hiện các loại tấn công hình học lên ảnh stego.
    """
    # 1. Đọc ảnh stego
    img = cv2.imread(input_path)
    if img is None:
        print("Lỗi: Không tìm thấy ảnh đầu vào.")
        return

    height, width = img.shape[:2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- TẤN CÔNG 1: XOAY ẢNH (ROTATION) ---
    # Xoay 2 độ quanh tâm ảnh
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, 1, 1.0)  # Góc xoay 2 độ
    rotated_img = cv2.warpAffine(img, matrix, (width, height))
    cv2.imwrite(os.path.join(output_folder, "attack_rotated.png"), rotated_img)

    # --- TẤN CÔNG 2: CẮT ẢNH (CROPPING) ---
    # Cắt bỏ 5% mỗi cạnh và resize về kích thước cũ (để đánh lừa bộ giải mã)
    h_crop = int(height * 0.05)
    w_crop = int(width * 0.05)
    cropped_img = img[h_crop:height - h_crop, w_crop:width - w_crop]
    resized_back = cv2.resize(cropped_img, (width, height))
    cv2.imwrite(os.path.join(output_folder, "attack_cropped.png"), resized_back)

    # --- TẤN CÔNG 3: THAY ĐỔI TỶ LỆ (SCALING) ---
    # Thu nhỏ xuống 90% rồi phóng to lại 100%
    small_img = cv2.resize(img, (int(width * 0.9), int(height * 0.9)), interpolation=cv2.INTER_LINEAR)
    scaled_img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_folder, "attack_scaled.png"), scaled_img)

    print(f"--- Tấn công hình học hoàn tất ---")
    print(f"Các file đã được lưu trong thư mục: {output_folder}")


# --- Thực thi ---
input_file = ".\DWT+SVD\stego.png"  # File đầu vào của bạn
output_dir = "geometric_results"

if __name__ == "__main__":
    # Kiểm tra xem người dùng có nhập đủ 2 tham số hay không
    # sys.argv phải có độ dài là 3 (gồm tên file, input_file, và output_dir)
    if len(sys.argv) < 3:
        print("Loi: Thieu tham so truyen vao!")
        print("Cu phap dung: python ten_file.py <input_file> <output_dir>")
        sys.exit(1)  # Thoát chương trình với mã lỗi

    # Gán tham số từ dòng lệnh vào biến
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.exists(input_file):
        geometric_attack(input_file, output_dir)
    else:
        print(f"Vui lòng để file {input_file} vào cùng thư mục script.")
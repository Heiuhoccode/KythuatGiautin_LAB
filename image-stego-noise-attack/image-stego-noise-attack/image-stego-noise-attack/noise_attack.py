import cv2
import numpy as np
import os
import sys

def add_salt_and_pepper_noise(image, amount=0.01):
    """Thêm nhiễu muối tiêu vào ảnh."""
    out = np.copy(image)
    # Tỷ lệ muối (trắng) và tiêu (đen)
    s_vs_p = 0.5
    num_salt = np.ceil(amount * image.size * s_vs_p)
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))

    # Thêm Muối (Salt)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    out[tuple(coords)] = 255

    # Thêm Tiêu (Pepper)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    out[tuple(coords)] = 0
    return out

def add_gaussian_noise(image, mean=0, sigma=15):
    """Thêm nhiễu Gauss vào ảnh."""
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = image.astype('float32') + gauss
    # Giới hạn giá trị trong khoảng [0, 255] và chuyển về uint8
    return np.clip(noisy, 0, 255).astype('uint8')

def noise_attack(input_path, output_folder):
    # 1. Đọc ảnh stego
    img = cv2.imread(input_path)
    if img is None:
        print("Loi: Khong tim thay anh đau vao.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- TẤN CÔNG 1: NHIỄU MUỐI TIÊU ---
    # amount=0.02 nghĩa là 2% số pixel bị ảnh hưởng
    sp_img = add_salt_and_pepper_noise(img, amount=0.02)
    cv2.imwrite(os.path.join(output_folder, "attack_salt_pepper.png"), sp_img)

    # --- TẤN CÔNG 2: NHIỄU GAUSS ---
    # sigma càng cao, nhiễu càng mạnh
    gauss_img = add_gaussian_noise(img, sigma=30)
    cv2.imwrite(os.path.join(output_folder, "attack_gaussian.png"), gauss_img)

    print(f"--- Tan cong nhieu hoan tat ---")
    print(f"Ket qua luu tai thu muc: {output_folder}")

if __name__ == "__main__":
    # Kiểm tra xem người dùng có nhập đủ 2 tham số hay không
    # sys.argv phải có độ dài là 3 (gồm tên file, input_file, và output_dir)
    if len(sys.argv) < 3:
        print("Loi: Thieu tham so truyen vao!")
        print("Cu phap dung: python ten_file.py <input_file> <output_dir>")
        sys.exit(1) # Thoát chương trình với mã lỗi

    # Gán tham số từ dòng lệnh vào biến
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Kiểm tra file nguồn trước khi chạy thuật toán
    if os.path.exists(input_file):
        noise_attack(input_file, output_dir)
    else:
        print(f"Loi: File '{input_file}' khong ton tai. Vui long kiem tra lai duong dan.")
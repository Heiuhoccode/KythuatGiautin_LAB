import argparse
from PIL import Image
import os


def text_to_bin(text):
    return ''.join(format(ord(c), '08b') for c in text) + '1111111111111110'


def hide_lsb(image_path, secret_path, out_path):
    if not os.path.exists(secret_path):
        print(f"[-] Lỗi: Không tìm thấy file text '{secret_path}'")
        return

    with open(secret_path, 'r', encoding='utf-8') as f:
        secret = f.read()

    img = Image.open(image_path).convert('RGB')
    pixels = list(img.getdata())
    binary_secret = text_to_bin(secret)

    if len(binary_secret) > len(pixels) * 3:
        print("[-] Lỗi: Ảnh quá nhỏ để chứa file text này!")
        return

    new_pixels = []
    idx = 0
    for p in pixels:
        r, g, b = p
        if idx < len(binary_secret): r = (r & ~1) | int(binary_secret[idx]); idx += 1
        if idx < len(binary_secret): g = (g & ~1) | int(binary_secret[idx]); idx += 1
        if idx < len(binary_secret): b = (b & ~1) | int(binary_secret[idx]); idx += 1
        new_pixels.append((r, g, b))

    img_out = Image.new(img.mode, img.size)
    img_out.putdata(new_pixels)
    img_out.save(out_path)
    print(f"[+] Đã giấu tin từ {secret_path} thành công vào ảnh: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Giấu tin LSB")
    parser.add_argument("-i", "--image", required=True, help="Đường dẫn ảnh gốc (VD: cover.png)")
    parser.add_argument("-s", "--secret", required=True, help="Đường dẫn file chứa thông điệp (VD: secret.txt)")
    parser.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh đầu ra (VD: stego.png)")
    args = parser.parse_args()

    hide_lsb(args.image, args.secret, args.output)
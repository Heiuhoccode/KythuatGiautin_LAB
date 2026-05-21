import argparse
from PIL import Image
import random


def destruction_attack(image_path, out_path):
    img = Image.open(image_path).convert('RGB')
    pixels = list(img.getdata())

    new_pixels = []
    for p in pixels:
        r = (p[0] & ~1) | random.randint(0, 1)
        g = (p[1] & ~1) | random.randint(0, 1)
        b = (p[2] & ~1) | random.randint(0, 1)
        new_pixels.append((r, g, b))

    img_out = Image.new(img.mode, img.size)
    img_out.putdata(new_pixels)
    img_out.save(out_path)
    print(f"[!] Đã tấn công phá hủy thông điệp. Lưu tại: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Tấn công phá hủy LSB (Destruction Attack)")
    parser.add_argument("-i", "--image", required=True, help="Đường dẫn ảnh mục tiêu (VD: stego.png)")
    parser.add_argument("-o", "--output", required=True, help="Đường dẫn ảnh sau khi tấn công (VD: attacked.png)")
    args = parser.parse_args()

    destruction_attack(args.image, args.output)
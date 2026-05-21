import argparse
from PIL import Image


def extract_lsb(image_path, out_text_path):
    img = Image.open(image_path).convert('RGB')
    pixels = list(img.getdata())

    binary_data = ""
    for p in pixels:
        binary_data += str(p[0] & 1)
        binary_data += str(p[1] & 1)
        binary_data += str(p[2] & 1)

    delimiter = '1111111111111110'
    if delimiter in binary_data:
        binary_data = binary_data[:binary_data.index(delimiter)]
    else:
        print("[-] Không tìm thấy chuỗi kết thúc. Ảnh không chứa thông điệp hoặc đã bị hỏng.")
        return

    text = ""
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i + 8]
        if len(byte) == 8:
            text += chr(int(byte, 2))

    with open(out_text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"[+] Giải mã thành công! Thông điệp được lưu tại: {out_text_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Trích xuất tin LSB")
    parser.add_argument("-i", "--image", required=True, help="Đường dẫn ảnh cần trích xuất (VD: stego.png)")
    parser.add_argument("-o", "--output", required=True, help="Đường dẫn file text đầu ra (VD: extracted.txt)")
    args = parser.parse_args()

    extract_lsb(args.image, args.output)
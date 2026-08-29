import os

image_dir = r"D:\my_roboflow\Bottle and Class Detection.v1i.yolov8\train\images"
label_dir = r"D:\my_roboflow\Bottle and Class Detection.v1i.yolov8\train\labels"

deleted_txt = 0
deleted_img = 0

for txt_name in os.listdir(label_dir):
    if not txt_name.endswith(".txt"):
        continue

    txt_path = os.path.join(label_dir, txt_name)
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if len(content) == 0:
        os.remove(txt_path)
        deleted_txt += 1

        base_name = os.path.splitext(txt_name)[0]
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(img_path):
                os.remove(img_path)
                deleted_img += 1
                break

print(f"Deleted txt files: {deleted_txt}")
print(f"Deleted image files: {deleted_img}")
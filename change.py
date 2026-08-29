import os

label_dir = r"D:\my_roboflow\Bottle and Class Detection.v1i.yolov8\train\labels"

for fname in os.listdir(label_dir):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(label_dir, fname)
    keep = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            c = parts[0]
            if c == "0":
                keep.append(" ".join(parts))
            # c=="1" 直接丢弃，什么都不做
    with open(fpath, "w", encoding="utf-8") as fw:
        fw.write("\n".join(keep))

print("?处理完毕：保留类别0，删除全部glass cup(类别1)")


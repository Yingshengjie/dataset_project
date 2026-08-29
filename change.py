import os

label_dir = r"D:\my_roboflow\Bottle and Class Detection.v1i.yolov8\train\labels"

print(f"Try path: {label_dir}")
print(f"Exists: {os.path.exists(label_dir)}")

if not os.path.exists(label_dir):
    print("ERROR: folder not found")
else:
    total = 0
    empty = 0
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        total += 1
        fpath = os.path.join(label_dir, fname)
        keep = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                cls = parts[0].strip()
                if cls == "0":
                    keep.append(" ".join(parts))
        with open(fpath, "w", encoding="utf-8") as fw:
            fw.write("\n".join(keep))
        if len(keep) == 0:
            empty += 1
    print(f"Total txt: {total}")
    print(f"Empty after process: {empty}")

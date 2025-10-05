
#sort images(split labels and images for val and train,convert to yolo)
import os
import cv2
import shutil

source_dir = r'E:\Coding\ML\ImageSelection\v0\data\cats\CAT_01'
image_train_dir = r'E:\Coding\ML\Localizations\dataset\images\train'
image_val_dir = r'E:\Coding\ML\Localizations\dataset\images\val'
label_train_dir = r'E:\Coding\ML\Localizations\dataset\labels\train'
label_val_dir = r'E:\Coding\ML\Localizations\dataset\labels\val'

#folder creation
for folder in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
    os.makedirs(folder, exist_ok=True)

#split every 5 to val
def is_val(index):
    return index % 5 == 0

#load img, coords
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

for idx, filename in enumerate(sorted(image_files)):
    image_path = os.path.join(source_dir, filename)
    cat_path = image_path + '.cat'

    if not os.path.exists(cat_path):
        print(f"No .cat file for {filename}, skipping")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {filename}, skipping")
        continue

    with open(cat_path, 'r') as f:
        raw = list(map(int, f.read().strip().split()))

    num_points = raw[0]
    coords = raw[1:]

    if len(coords) < num_points * 2:
        print(f"Malformed .cat file for {filename}, skipping")
        continue

    points = []
    for i in range(0, len(coords) - 1, 2):
        x, y = coords[i], coords[i + 1]
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            points.append((x, y))

    if not points:
        print(f"No valid points for {filename}, skipping")
        continue

    #Convert to YOLO format
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2 / image.shape[1]
    y_center = (y_min + y_max) / 2 / image.shape[0]
    width = (x_max - x_min) / image.shape[1]
    height = (y_max - y_min) / image.shape[0]

    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    #destination
    if is_val(idx):
        image_dest = os.path.join(image_val_dir, filename)
        label_dest = os.path.join(label_val_dir, os.path.splitext(filename)[0] + '.txt')
    else:
        image_dest = os.path.join(image_train_dir, filename)
        label_dest = os.path.join(label_train_dir, os.path.splitext(filename)[0] + '.txt')

    shutil.copy2(image_path, image_dest)

    #label
    with open(label_dest, 'w') as out:
        out.write(yolo_line + '\n')

    print(f"Processed {filename} → {'val' if is_val(idx) else 'train'}")

print("Dataset organized and converted.")

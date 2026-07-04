import os, random, shutil
# 数据集根目录
root = r"D:\project\PythonProject6\raw_animal\YOLODataset"
img_train = os.path.join(root, "images/train")
label_train = os.path.join(root, "labels/train")
img_val = os.path.join(root, "images/val")
label_val = os.path.join(root, "labels/val")
# 四类ID 0牛 1羊 2马 3骆驼
cls_img_map = {"0":[], "1":[], "2":[], "3":[]}

# 按类别归集所有图片
for txt_file in os.listdir(label_train):
    if not txt_file.endswith(".txt"):
        continue
    txt_path = os.path.join(label_train, txt_file)
    img_name = txt_file.replace(".txt", ".jpg")
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.readlines()
        target_ids = set()
        for line in content:
            cid = line.split()[0]
            target_ids.add(cid)
        for c in target_ids:
            cls_img_map[c].append(img_name)

# 每一类单独抽20%移动到val，保证四类val样本均衡
for cid, img_list in cls_img_map.items():
    random.shuffle(img_list)
    split_count = int(len(img_list) * 0.25)
    move_list = img_list[:split_count]
    for img in move_list:
        # 移动图片
        shutil.move(os.path.join(img_train, img), img_val)
        # 同步移动标签
        txt = img.replace(".jpg", ".txt")
        shutil.move(os.path.join(label_train, txt), label_val)
print("分层重划分完成，四类验证集样本均衡（无新增图片）")
import os
import json

# 设置数据集路径和标签
root_dir = 'your dataset path'  # 根路径
splits = ['train', 'test', 'validation']  # 划分名称

# 为每个划分生成 annotations.json
for split in splits:
    images_info = []
    for label_name, label in {'real': 0, 'fake': 1}.items():
        folder_path = os.path.join(root_dir, split, label_name)
        for img_name in os.listdir(folder_path):
            if img_name.endswith(('.jpg', '.png')):  # 根据您的图像格式选择
                img_path = os.path.join(label_name, img_name)  # 相对路径
                images_info.append({"path": img_path, "label": label})

    # 将信息保存为 JSON 文件
    annotations_path = os.path.join(root_dir, split, 'annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump({"images": images_info}, f, indent=4)
    print(f"Generated {annotations_path}")

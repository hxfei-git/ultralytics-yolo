import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path

# ================= 配置区域 =================
model_path = 'best.pt'
input_dir = './heatmap_src'   # 输入文件夹
output_dir = './heatmap_dst'      # 输出文件夹
img_size = 640                
# ===========================================

os.makedirs(output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载模型
model = YOLO(model_path).model.to(device)
model.eval()

# 2. Hook 逻辑
features = {}
block_call_count = 0

def get_generic_hook(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook

def get_block_count_hook():
    def hook(module, input, output):
        global block_call_count
        block_call_count += 1
        if block_call_count == 1: features['x2'] = output.detach()
        elif block_call_count == 2: features['x3'] = output.detach()
        elif block_call_count == 3: features['x4'] = output.detach()
    return hook

for name, module in model.named_modules():
    if "ASCSPP" in module.__class__.__name__:
        module.cv1.register_forward_hook(get_generic_hook('x1'))
        module.block.register_forward_hook(get_block_count_hook())
        module.global_conv.register_forward_hook(get_generic_hook('global'))
        print(f"Hooks registered on: {name}")
        break

# 3. 增强版拼接函数
def create_combined_result(raw_img, keys):
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    combined_row = []
    
    # 获取字体大小比例（根据图像宽度动态调整，防止字太大或太小）
    font_scale = max(raw_img.shape[1] / 400, 1.0)
    thickness = max(int(font_scale * 2), 2)

    for key in keys:
        if key == 'original':
            # 直接把原图放进来，打上标签
            res = img_rgb.copy()
            cv2.putText(res, "Original", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            combined_row.append(res)
            continue

        if key not in features:
            combined_row.append(np.zeros_like(img_rgb))
            continue
            
        # 计算热力图
        f_map = features[key]
        heatmap = torch.mean(f_map, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        # 伪彩色叠加
        heatmap_res = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # 叠加
        result = cv2.addWeighted(img_rgb, 0.4, heatmap_color, 0.6, 0)
        cv2.putText(result, key, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        combined_row.append(result)
    
    # 横向拼接
    final_img = np.hstack(combined_row)
    return cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

# 4. 遍历处理
image_files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png']]

print(f"Processing {len(image_files)} images...")

for filename in image_files:
    img = cv2.imread(os.path.join(input_dir, filename))
    if img is None: continue

    # 推理准备
    img_res = cv2.resize(img, (img_size, img_size))
    img_in = img_res[:, :, ::-1].transpose(2, 0, 1)
    input_tensor = torch.from_numpy(np.ascontiguousarray(img_in)).unsqueeze(0).float().to(device) / 255.0

    block_call_count = 0
    with torch.no_grad():
        _ = model(input_tensor)

    # 包含原图的拼接序列
    result_img = create_combined_result(img, ['original', 'x1', 'x2', 'x3', 'x4', 'global'])

    save_path = os.path.join(output_dir, f"{Path(filename).stem}_full_compare.png")
    cv2.imwrite(save_path, result_img)
    print(f"Saved: {save_path}")

print("Done!")
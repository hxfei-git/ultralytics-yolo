import matplotlib.pyplot as plt
import numpy as np

# 数据准备：更新为你的 N 规模模型数据
categories = ['AL', 'BR', 'ST', 'SH', 'SP', 'VE', 'PE', 'WM']
yolo_n = np.array([61.2, 26.0, 70.5, 63.5, 13.3, 64.9, 27.1, 7.4])
ours_n = np.array([67.0, 28.4, 72.3, 66.0, 16.8, 66.8, 29.3, 13.3])

# 闭合雷达图路径
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
yolo_plot = np.concatenate((yolo_n, [yolo_n[0]]))
ours_plot = np.concatenate((ours_n, [ours_n[0]]))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 1. 设置量程（根据 N 规模数据分布，0-85 比较合适）
ax.set_ylim(0, 85)

# 2. 配色方案（论文常用：深蓝 vs 橘红）
color_base = "#577FC1" 
color_ours = "#D35400" 

# 绘制 YOLOv11-N 曲线
ax.plot(angles, yolo_plot, linewidth=2, label="YOLOv11-N", color=color_base, marker='o', markersize=4)
ax.fill(angles, yolo_plot, alpha=0.1, color=color_base)

# 绘制 MTFE-YOLO-N 曲线
ax.plot(angles, ours_plot, linewidth=2.5, label="MTFE-YOLO-N", color=color_ours, marker='s', markersize=4)
ax.fill(angles, ours_plot, alpha=0.2, color=color_ours)

# 3. 标注提升值（核心修正：自动处理正负号）
for i in range(len(categories)):
    diff = ours_n[i] - yolo_n[i]
    # 使用 :+.1f 格式化：正数带 +，负数带 -
    # 位置设在两条线最高值上方 5 个单位处
    text_pos = max(ours_n[i], yolo_n[i]) + 5
    
    ax.text(angles[i], text_pos, 
            f"{diff:+.1f}", 
            ha='center', 
            va='center', 
            fontsize=10, 
            fontweight='bold',
            color="#CA1417" if diff > 0 else "#32C71B") # 正值深绿，负值深红

# 网格与坐标轴设置
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_yticks([20, 40, 60, 80])
ax.set_yticklabels(['20', '40', '60', '80'], fontsize=10, color="gray")
ax.grid(True, linestyle='--', alpha=0.5)

# 图例与标题
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
ax.set_title("Per-category mAP50 Comparison (N-scale)", pad=30, fontsize=14, fontweight='bold')

plt.tight_layout()
# 保存图片
plt.savefig("radar_n_scale.png", dpi=300, bbox_inches='tight')
plt.show()
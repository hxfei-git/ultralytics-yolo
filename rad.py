import matplotlib.pyplot as plt
import numpy as np

categories = ['AL', 'BR', 'ST', 'SH', 
              'SP', 'VE', 'PE', 'WM']

yolo = np.array([61.2, 26.0, 70.5, 63.5, 13.3, 64.9, 27.1, 7.4])
mtfe = np.array([67.0, 28.4, 72.3, 66.0, 16.8, 66.8, 29.3, 13.3])

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

yolo = np.concatenate((yolo, [yolo[0]]))
mtfe = np.concatenate((mtfe, [mtfe[0]]))

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

# ⭐ 关键1：提高下限（放大差异）
ax.set_ylim(5, 80)

# ⭐ 关键2：柔和配色
color1 = "#577FC1"
color2 = "#DD8452"

ax.plot(angles, yolo, linewidth=2.5, label="YOLOv11-N", color=color1)
ax.fill(angles, yolo, alpha=0.15, color=color1)

ax.plot(angles, mtfe, linewidth=2.5, label="MTFE-YOLO-N", color=color2)
ax.fill(angles, mtfe, alpha=0.20, color=color2)

# 网格样式
ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

ax.set_yticks([20,40,60,80])
ax.set_yticklabels(['20','40','60','80'], fontsize=9)

# ⭐ 关键3：标注提升值
for i in range(len(categories)):
    diff = mtfe[i] - yolo[i]
    ax.text(angles[i], mtfe[i] + 2,
            f"+{diff:.1f}",
            ha='center',
            va='center',
            fontsize=9,
            color="darkblue")

ax.legend(loc='upper right', bbox_to_anchor=(1.15,1.1))
ax.set_title("mAP50 Comparison per Category", pad=20)

plt.tight_layout()
plt.savefig("radar_paper_style.png", dpi=300, bbox_inches='tight')
plt.show()
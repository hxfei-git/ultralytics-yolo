import warnings
import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案

warnings.filterwarnings('ignore')

from ultralytics import YOLO

# 启动新会话：tmux
# 启动命名会话：tmux new -s 会话名
# 分离会话（保持后台运行）：Ctrl+b d
# 进入已有会话：tmux attach -t 会话名
# 进入最后一个会话：tmux attach 或 tmux a
# 列出所有会话：tmux ls
# 关闭指定会话：tmux kill-session -t 会话名
# 关闭所有会话：tmux kill-server
# 训练后关机：python3 train.py; /usr/bin/shutdown -h now
# 删除本地分支：git branch -d xx
# 清理远端分支：git remote prune origin --dry-run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO11 with optional YAML and name")
    parser.add_argument('--yaml', type=str,
                        default='ultralytics/cfg/models/11/yolo11-FAFF.yaml')
    parser.add_argument('--name', type=str,
                        default='yolo11n-FAFF')
    args = parser.parse_args()

    model = YOLO(args.yaml)  # YOLO11
    # model.load('yolo11n.pt')  # loading pretrain weights
    model.train(
        data='../datasets/voc-ai-tod.yaml',
        cache=False,
        imgsz=640,
        epochs=10000,
        batch=12,
        close_mosaic=0,  # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
        workers=6,       # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
        # device='0,1',  # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
        optimizer='SGD',  # using SGD
        # patience=0,    # set 0 to close earlystop.
        # resume=True,   # 断点续训,YOLO初始化时选择last.pt,不懂就在百度云.txt找断点续训的视频
        # amp=False,     # close amp | loss出现nan可以关闭amp
        # fraction=0.2,
        project='runs/experiment',
        name=args.name,   # 使用命令行传入的名字（默认仍为 'yolo11n'）
        exist_ok=True,
    )
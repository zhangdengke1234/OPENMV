# 立创·庐山派-K230-CanMV开发板资料与相关扩展板软硬件资料官网全部开源
# 开发板官网：www.lckfb.com
# 技术支持常驻论坛，任何技术问题欢迎随时交流学习
# 立创论坛：www.jlc-bbs.com/lckfb
# 关注bilibili账号：【立创开发板】，掌握我们的最新动态！
# 编写者：LCKFB-AI-Asst
# --------------------------------------------------
# AI + 颜色双确认 v3
# 修正：红球带黑色花纹 → 不用"填充率"，改用：
#   1. 红占比 > 阈值（红像素占AI框比例）
#   2. 红色合并后(merge=True)最接近框中心 → 中心偏移约束
#   红色 blob 质心应靠近 AI 框中心(球在框中间)，人脸红区偏/散
# --------------------------------------------------

import os, gc, time
from libs.PlatTasks import DetectionApp
from libs.Utils import *
from media.sensor import *
from media.display import *
from media.media import *

VIDEO_W, VIDEO_H = 640, 360

sensor = Sensor(fps=30)
sensor.reset()
sensor.set_framesize(width=VIDEO_W, height=VIDEO_H, chn=CAM_CHN_ID_0)
sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_0)
sensor.set_framesize(width=VIDEO_W, height=VIDEO_H, chn=CAM_CHN_ID_2)
sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)

Display.init(Display.VIRT, width=VIDEO_W, height=VIDEO_H, fps=30, to_ide=True)
sensor.run()

root_path = "/sdcard/mp_deployment_source/"
deploy_conf = read_json(root_path + "deploy_config.json")
kmodel_path = root_path + deploy_conf["kmodel_path"]
labels = deploy_conf["categories"]
model_input_size = deploy_conf["img_size"]
model_type = deploy_conf["model_type"]
anchors = []
if model_type == "AnchorBaseDet":
    anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]

confidence_threshold = 0.4
nms_threshold = 0.5
det_app = DetectionApp("video", kmodel_path, labels, model_input_size, anchors,
                       model_type, confidence_threshold, nms_threshold,
                       [VIDEO_W, VIDEO_H], [VIDEO_W, VIDEO_H], debug_mode=0)
det_app.config_preprocess()

RED_LAB = [(19, 90, 15, 55, -5, 25)]
RED_RATIO_MIN = 0.15        # 红占比下限
CENTER_OFFSET_MAX = 0.25    # 红质心偏离AI框中心的比例上限

print(">>> AI+颜色 v3：红占比>%.2f + 红区质心居中" % RED_RATIO_MIN)
print(">>> 兼容带花纹红球，滤人脸")

def main_loop():
    while True:
        os.exitpoint()
        ai_img = sensor.snapshot(chn=CAM_CHN_ID_2)
        np_ai = ai_img.to_numpy_ref()
        frame = sensor.snapshot(chn=CAM_CHN_ID_0)
        res = det_app.run(np_ai)

        confirmed = []
        if res and "boxes" in res and len(res["boxes"]) > 0:
            for i in range(len(res["boxes"])):
                x1 = int(res["boxes"][i][0]); y1 = int(res["boxes"][i][1])
                x2 = int(res["boxes"][i][2]); y2 = int(res["boxes"][i][3])
                s = float(res["scores"][i])
                w = x2 - x1; h = y2 - y1
                if w < 10 or h < 10:
                    continue

                # 框内找红 blob（merge=True 合并黑线分割的红色块）
                blobs = frame.find_blobs(RED_LAB, pixels_threshold=20,
                                         area_threshold=20,
                                         roi=(x1, y1, w, h), merge=True)
                if not blobs:
                    print("滤掉: score=%.2f 框内无红" % s)
                    continue

                # ① 红占比
                total_red = sum(b.pixels() for b in blobs)
                ratio = total_red / (w * h)

                # ② 最大红blob质心与AI框中心的偏移
                best = max(blobs, key=lambda b: b.pixels())
                box_cx = x1 + w // 2
                box_cy = y1 + h // 2
                dx = (best.cx() - box_cx) / w
                dy = (best.cy() - box_cy) / h
                offset = (dx*dx + dy*dy) ** 0.5

                if ratio >= RED_RATIO_MIN and offset <= CENTER_OFFSET_MAX:
                    confirmed.append((x1, y1, x2, y2, s, ratio, offset))
                    print("✅ 红球: score=%.2f 红占=%.2f 偏移=%.2f" % (s, ratio, offset))
                else:
                    print("滤掉: score=%.2f 红占=%.2f 偏移=%.2f(非球)" % (s, ratio, offset))

        frame.draw_string_advanced(10, 10, 32, "v3", color=(0,255,0))
        for (x1, y1, x2, y2, s, ratio, offset) in confirmed:
            frame.draw_rectangle(x1, y1, x2-x1, y2-y1, color=(255,0,0), thickness=3)
            frame.draw_string_advanced(x1, y1-20, 24, "RED", color=(255,0,0))
        Display.show_image(frame)
        gc.collect()

try:
    main_loop()
except KeyboardInterrupt:
    print("已停止")
except Exception as e:
    print("异常:", e)
finally:
    try:
        det_app.deinit()
    except Exception:
        pass
    try:
        sensor.stop()
    except Exception:
        pass
    try:
        Display.deinit()
    except Exception:
        pass
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    try:
        MediaManager.deinit()
    except Exception:
        pass
    print("已释放资源")

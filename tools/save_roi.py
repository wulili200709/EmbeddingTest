# save_roi.py
import cv2, json

img_path = r"OK/Image_20260204164800760.bmp"  # 改成你的OK图（bmp/jpg都行）
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(img_path)

H, W = img.shape[:2]

# 最大显示尺寸（可改）
MAX_W, MAX_H = 1920, 1080

# 计算缩放比例（只缩小，不放大）
scale = min(MAX_W / W, MAX_H / H, 1.0)
disp = img if scale == 1.0 else cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.imshow("Select ROI", disp)

# 在缩放后的图上选 ROI
r = cv2.selectROI("Select ROI", disp, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x_s, y_s, w_s, h_s = map(int, r)
if w_s == 0 or h_s == 0:
    raise RuntimeError("ROI 为空：你可能按了取消或没框到区域")

# 换算回原图坐标（非常关键）
x = int(round(x_s / scale))
y = int(round(y_s / scale))
w = int(round(w_s / scale))
h = int(round(h_s / scale))

# 防越界
x = max(0, min(x, W - 1))
y = max(0, min(y, H - 1))
w = max(1, min(w, W - x))
h = max(1, min(h, H - y))

print("Display scale:", scale)
print("ROI (orig):", x, y, w, h)

with open("roi.json", "w", encoding="utf-8") as f:
    json.dump({"x": x, "y": y, "w": w, "h": h, "scale_for_display": scale}, f, indent=2)

print("saved roi.json")


"""Run YuNet once on the meeting frame and write dashboard/assets/faces.json.

Usage: <python with OpenCV ≥ 4.8> detect-faces.py <yunet.onnx> <frame.jpg> <out.json>
Boxes and landmarks are written as fractions of the frame (x, y, w, h), so they
scale with the stage. The explainer's Vision view draws these real detections.
"""
import json, sys
import cv2

model, frame_path, out = sys.argv[1:4]
img = cv2.imread(frame_path)
h, w = img.shape[:2]
# Detect on a 2× upscale: the seated faces are small in the 960×540 frame.
scale = 2
big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
det = cv2.FaceDetectorYN.create(model, '', (w * scale, h * scale), score_threshold=0.5, nms_threshold=0.3, top_k=50)
_, faces = det.detect(big)
res = []
for f in (faces if faces is not None else []):
    x, y, bw, bh = (float(v) / scale for v in f[:4])
    lm = [[float(f[4 + 2 * i]) / scale / w, float(f[5 + 2 * i]) / scale / h] for i in range(5)]
    res.append({'box': [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)],
                'landmarks': [[round(a, 4), round(b, 4)] for a, b in lm], 'score': round(float(f[14]), 3)})
res.sort(key=lambda r: r['box'][0])
json.dump({'model': 'YuNet face_detection_yunet_2023mar', 'frame': [w, h], 'faces': res}, open(out, 'w'), indent=1)
print(json.dumps(res, indent=None))

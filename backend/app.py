import os
import io
import base64
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
import cv2

app = Flask(__name__)

# 啟用 CORS 支援 GitHub Pages 跨域請求
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 系統通用限制：單一檔案大小不得超過 20MB (FR-1.3 / 第2節)
MAX_FILE_SIZE = 20 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'}

# 伺服器端暫存工作區（遵循非破壞性編輯原則，僅保存原始上傳影像與基礎中繼資料）
IMAGE_STORE = {
    "raw_bytes": None,
    "bgr_image": None,      # 供各模組進行 3 通道標準運算之基準影像 (uint8 BGR)
    "orig_image": None,     # 原始解碼影像 (保留原始通道與位元深度供 FR-2 解析)
    "filename": None,
    "ext": None,
    "file_size": 0
}

# YuNet 模型路徑與自動下載設定 (FR-5.1)
YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_yunet_model():
    """確保 YuNet ONNX 模型存在，若不存在則自動自官方儲存庫下載"""
    if not os.path.exists(YUNET_MODEL_PATH):
        print("正在下載 YuNet 人臉偵測模型權重...")
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
        print("YuNet 模型下載完成。")


def encode_image_to_base64(img_array):
    """將 OpenCV 影像陣列編碼為 PNG 格式之 Base64 Data URL (1.4 規範)"""
    success, buffer = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("影像編碼失敗")
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def ensure_uint8(img):
    """若原始影像為 16-bit，轉換為標準 8-bit 供後續視覺化與濾波運算"""
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """HTTP 413 保護機制 (第2節規範)"""
    return jsonify({
        "status": "error",
        "message": "檔案大小超過 20MB 上限 (HTTP 413 Payload Too Large)"
    }), 413


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Backend API is running"}), 200


# ==============================================================================
# 5.1 影像上傳與檔案解析 API (/api/upload)
# ==============================================================================
@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "未偵測到上傳之 image 欄位"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未選擇任何檔案"}), 400

    # 檢查副檔名
    filename = file.filename
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "status": "error",
            "message": f"不支援的影像格式 (.{ext})，僅支援 JPG, JPEG, PNG, BMP, WEBP, TIFF"
        }), 400

    raw_bytes = file.read()
    file_size = len(raw_bytes)
    if file_size > MAX_FILE_SIZE:
        return jsonify({
            "status": "error",
            "message": "單一影像檔案大小不得超過 20MB"
        }), 413

    # 使用 IMREAD_UNCHANGED 解析真實通道數與色彩深度
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    orig_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if orig_img is None:
        return jsonify({"status": "error", "message": "無法解析影像檔案，檔案可能已損毀"}), 400

    # 解析幾何資訊與通道結構 (FR-2.1 ~ FR-2.5)
    height, width = orig_img.shape[:2]
    total_pixels = int(width * height)

    if len(orig_img.shape) == 2:
        channels = 1
        color_mode = "Grayscale (灰階)"
    elif orig_img.shape[2] == 3:
        channels = 3
        color_mode = "RGB (彩色)"
    elif orig_img.shape[2] == 4:
        channels = 4
        color_mode = "RGBA (含透明通道彩色)"
    else:
        channels = int(orig_img.shape[2])
        color_mode = f"Multi-channel ({channels} 通道)"

    # 解析各通道位元深度 (Bit Depth)
    dtype_str = str(orig_img.dtype)
    bit_depth_map = {
        'uint8': '8-bit per channel (uint8)',
        'int8': '8-bit per channel (int8)',
        'uint16': '16-bit per channel (uint16)',
        'int16': '16-bit per channel (int16)',
        'float32': '32-bit float per channel (float32)'
    }
    bit_depth = bit_depth_map.get(dtype_str, f"{orig_img.dtype.itemsize * 8}-bit per channel")

    # 建立標準 3 通道 8-bit BGR 基準影像供後續通道分離、濾波與人臉偵測使用
    img_u8 = ensure_uint8(orig_img)
    if len(img_u8.shape) == 2:
        bgr_img = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    elif img_u8.shape[2] == 4:
        bgr_img = cv2.cvtColor(img_u8, cv2.COLOR_BGRA2BGR)
    else:
        bgr_img = img_u8.copy()

    # 儲存至暫存區（非破壞性編輯基準）
    IMAGE_STORE["raw_bytes"] = raw_bytes
    IMAGE_STORE["orig_image"] = orig_img
    IMAGE_STORE["bgr_image"] = bgr_img
    IMAGE_STORE["filename"] = filename
    IMAGE_STORE["ext"] = ext.upper()
    IMAGE_STORE["file_size"] = file_size

    return jsonify({
        "status": "success",
        "code": 200,
        "data": {
            "filename": filename,
            "format": ext.upper(),
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "color_mode": color_mode,
            "channels": channels,
            "bit_depth": bit_depth,
            "original_base64": encode_image_to_base64(bgr_img)
        }
    }), 200


# ==============================================================================
# 5.2 色彩通道分離與直方圖 API (/api/color-channel)
# ==============================================================================
@app.route('/api/color-channel', methods=['POST'])
def color_channel():
    if IMAGE_STORE["bgr_image"] is None:
        return jsonify({"status": "error", "message": "請先上傳影像檔案"}), 400

    payload = request.get_json(silent=True) or {}
    mode = payload.get("channel_mode", "Original")

    valid_modes = {"Original", "R", "G", "B", "Y", "U", "V"}
    if mode not in valid_modes:
        return jsonify({"status": "error", "message": f"無效的通道模式: {mode}"}), 400

    # 永遠由原始暫存影像讀取，確保非破壞性原則
    bgr_img = IMAGE_STORE["bgr_image"].copy()
    b_ch, g_ch, r_ch = cv2.split(bgr_img)

    if mode == "Original":
        target_img = bgr_img
        hist_r = cv2.calcHist([r_ch], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
        hist_g = cv2.calcHist([g_ch], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
        hist_b = cv2.calcHist([b_ch], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
        histograms = {
            "R": hist_r,
            "G": hist_g,
            "B": hist_b
        }
    else:
        if mode == "R":
            single_ch = r_ch
        elif mode == "G":
            single_ch = g_ch
        elif mode == "B":
            single_ch = b_ch
        else:
            # 轉換至 YUV 色彩空間取亮度與色度分量 (FR-3.1)
            yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
            y_ch, u_ch, v_ch = cv2.split(yuv_img)
            if mode == "Y":
                single_ch = y_ch
            elif mode == "U":
                single_ch = u_ch
            else:  # "V"
                single_ch = v_ch

        target_img = single_ch
        hist_single = cv2.calcHist([single_ch], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
        histograms = {
            mode: hist_single
        }

    return jsonify({
        "status": "success",
        "channel_mode": mode,
        "image_base64": encode_image_to_base64(target_img),
        "histogram": histograms
    }), 200


# ==============================================================================
# 5.3 卷積濾波與 PSNR 分析 API (/api/filter)
# ==============================================================================
@app.route('/api/filter', methods=['POST'])
def apply_filter():
    if IMAGE_STORE["bgr_image"] is None:
        return jsonify({"status": "error", "message": "請先上傳影像檔案"}), 400

    payload = request.get_json(silent=True) or {}
    kernel_size = payload.get("kernel_size")
    kernel_matrix = payload.get("kernel_matrix")
    is_normalized = bool(payload.get("is_normalized", False))

    if kernel_size not in (3, 5):
        return jsonify({"status": "error", "message": "kernel_size 必須為 3 或 5"}), 400

    try:
        kernel = np.array(kernel_matrix, dtype=np.float64)
        if kernel.shape != (kernel_size, kernel_size):
            return jsonify({"status": "error", "message": "矩陣維度與指定尺寸不符"}), 400
        if np.isnan(kernel).any() or np.isinf(kernel).any():
            return jsonify({"status": "error", "message": "矩陣包含無效數值"}), 400
    except Exception:
        return jsonify({"status": "error", "message": "無法解析卷積核矩陣格式"}), 400

    # 驗證不可為全 0 矩陣 (FR-4.4)
    if np.all(np.isclose(kernel, 0.0)):
        return jsonify({"status": "error", "message": "禁止使用全 0 矩陣執行卷積運算"}), 400

    # 正規化處理邏輯 (FR-4.5 & 5.3)
    kernel_sum = float(np.sum(kernel))
    norm_warning = None

    if is_normalized:
        if kernel_sum > 0:
            used_kernel = kernel / kernel_sum
        else:
            # 總和小於等於 0 (如拉普拉斯算子)，不適用標準除法正規化
            used_kernel = kernel.copy()
            norm_warning = "此濾波核總和小於等於 0，不適用標準除法正規化"
    else:
        used_kernel = kernel.copy()

    # 讀取原始影像並拆分 B, G, R 獨立套用卷積核 (FR-4.6)
    orig_bgr = IMAGE_STORE["bgr_image"].copy()
    b_ch, g_ch, r_ch = cv2.split(orig_bgr)

    # 明確指定 cv2.BORDER_REFLECT_101 反射邊界模式，以浮點數運算後進行 0~255 截斷 (Clip)
    filtered_b = cv2.filter2D(b_ch.astype(np.float64), cv2.CV_64F, used_kernel, borderType=cv2.BORDER_REFLECT_101)
    filtered_g = cv2.filter2D(g_ch.astype(np.float64), cv2.CV_64F, used_kernel, borderType=cv2.BORDER_REFLECT_101)
    filtered_r = cv2.filter2D(r_ch.astype(np.float64), cv2.CV_64F, used_kernel, borderType=cv2.BORDER_REFLECT_101)

    merged = cv2.merge([filtered_b, filtered_g, filtered_r])
    filtered_bgr = np.clip(np.round(merged), 0, 255).astype(np.uint8)

    # 計算 MSE 與 PSNR (FR-4.7 & 5.3)
    mse = float(np.mean((orig_bgr.astype(np.float64) - filtered_bgr.astype(np.float64)) ** 2))
    if mse == 0.0:
        psnr_val = "inf"
    else:
        psnr_val = round(float(10.0 * np.log10((255.0 ** 2) / mse)), 2)

    return jsonify({
        "status": "success",
        "filtered_image_base64": encode_image_to_base64(filtered_bgr),
        "original_image_base64": encode_image_to_base64(orig_bgr),
        "normalized_matrix": np.round(used_kernel, 6).tolist(),
        "mse": round(mse, 4),
        "psnr": psnr_val,
        "norm_warning": norm_warning
    }), 200


# ==============================================================================
# 5.4 人臉偵測 API (/api/face-detect)
# ==============================================================================
@app.route('/api/face-detect', methods=['POST'])
def face_detect():
    if IMAGE_STORE["bgr_image"] is None:
        return jsonify({"status": "error", "message": "請先上傳影像檔案"}), 400

    try:
        ensure_yunet_model()
    except Exception as e:
        return jsonify({"status": "error", "message": f"載入 YuNet 模型失敗: {str(e)}"}), 500

    # 於原始輸入影像複製本上進行推論與繪製 (FR-5.1 & FR-5.2)
    orig_bgr = IMAGE_STORE["bgr_image"].copy()
    height, width = orig_bgr.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL_PATH,
        config="",
        input_size=(width, height),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    _, faces = detector.detect(orig_bgr)

    confidences = []
    face_count = 0

    if faces is not None:
        face_count = int(len(faces))
        # 使用 enumerate 取得 index (idx)，用於標示 #n
        for idx, face in enumerate(faces):
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[14])
            conf_percent = round(score * 100.0, 2)
            confidences.append(conf_percent)
            
            cv2.rectangle(orig_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Y 座標加入 max(y - 8, 15) 防止文字超出圖片頂部邊界
            text = f"#{idx + 1}: {conf_percent}%"
            cv2.putText(orig_bgr, text, (x, max(y - 8, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return jsonify({
        "status": "success",
        "annotated_image_base64": encode_image_to_base64(orig_bgr),
        "face_count": face_count,
        "confidences": confidences
    }), 200


if __name__ == '__main__':
    ensure_yunet_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
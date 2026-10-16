# Tài Liệu API: Endpoint `/api/predict`

Tài liệu hướng dẫn sử dụng endpoint `/api/predict` dùng để nhận diện và đếm số lượng học sinh trong ảnh phòng học.

---

## 1. Tổng Quan

Endpoint tiếp nhận một tệp hình ảnh phòng học, thực hiện nhận diện học sinh bằng mô hình YOLO kết hợp lọc trùng lặp theo độ sâu (Depth-NMS với Depth Anything V2), và trả về kết quả số lượng người cùng chi tiết từng bounding box.

- **Đường dẫn (URL):** `/api/predict` (ví dụ: `http://localhost:8080/api/predict`)
- **Phương thức (HTTP Method):** `POST`
- **Định dạng dữ liệu gửi (Content-Type):** `multipart/form-data`
- **Định dạng dữ liệu nhận:** `application/json`

---

## 2. Dữ Liệu Đầu Vào (Request)

Yêu cầu gửi dữ liệu dưới dạng form-data với trường tệp tin:

| Trường (Field) | Kiểu (Type) | Bắt buộc | Mô tả |
|:---|:---|:---:|:---|
| `image` | `File` (Binary) | **Có** | Tệp hình ảnh cần nhận diện. |

### Định dạng ảnh được hỗ trợ
- `image/jpeg` (`.jpg`, `.jpeg`)
- `image/png` (`.png`)
- `image/webp` (`.webp`)

> **Lưu ý:** Tệp tải lên không được rỗng và phải là ảnh hợp lệ để OpenCV có thể giải mã.

---

## 3. Dữ Liệu Trả Về (Response)

### Mã trạng thái thành công: `200 OK`

Dữ liệu trả về có cấu trúc JSON như sau:

```json
{
  "id": "c1f7b7ad-4520-4e56-b08e-1f7c1184df9a",
  "people_count": 44,
  "kept_count": 44,
  "suppressed_count": 1,
  "detections": [
    {
      "box_id": 0,
      "bbox": [730, 70, 774, 110],
      "confidence": 0.7584,
      "depth": 0.8245,
      "depth_std": 0.0123,
      "kept": true,
      "suppressed_by": null,
      "suppression_reason": null
    },
    {
      "box_id": 12,
      "bbox": [732, 72, 775, 112],
      "confidence": 0.4120,
      "depth": 0.8260,
      "depth_std": 0.0115,
      "kept": false,
      "suppressed_by": 0,
      "suppression_reason": "Duplicate of #0 (IoU=0.88 >= 0.25, depth_diff=0.0015 < 0.0100)"
    }
  ]
}
```

### Chi tiết các trường thông tin:

#### Cấp cao nhất (`PredictResponse`):
- `id` (*string*): Mã định danh duy nhất (UUID) của phiên dự đoán.
- `people_count` (*integer*): **Số lượng học sinh thực tế đếm được** (chỉ tính các box hợp lệ được giữ lại).
- `kept_count` (*integer*): Tổng số bounding box được giữ lại (bằng với `people_count`).
- `suppressed_count` (*integer*): Số lượng bounding box bị loại bỏ do xác định là trùng lặp (false positive hoặc trùng vị trí theo độ sâu).
- `detections` (*array*): Danh sách chi tiết toàn bộ các phát hiện (bao gồm cả box giữ lại và box bị loại bỏ).

#### Từng phần tử trong `detections` (`DetectionItem`):
- `box_id` (*integer*): Mã định danh số thứ tự của box trong ảnh.
- `bbox` (*array of int*): Tọa độ hộp bao theo thứ tự `[x1, y1, x2, y2]` (pixel):
  - `x1`: Tọa độ mép trái
  - `y1`: Tọa độ mép trên
  - `x2`: Tọa độ mép phải
  - `y2`: Tọa độ mép dưới
- `confidence` (*float*): Điểm tin cậy của phát hiện (từ `0.0` đến `1.0`).
- `depth` (*float*): Giá trị độ sâu tương đối trung vị (median relative depth) trong vùng box (`0.0` là gần nhất, `1.0` là xa nhất).
- `depth_std` (*float*): Độ lệch chuẩn của độ sâu trong vùng box.
- `kept` (*boolean*):
  - `true`: Box hợp lệ, được tính vào tổng số người.
  - `false`: Box bị loại bỏ do trùng lặp.
- `suppressed_by` (*integer | null*): `box_id` của box ưu tiên hơn đã loại bỏ box này (nếu `kept = false`).
- `suppression_reason` (*string | null*): Lý do kỹ thuật giải thích vì sao box bị loại bỏ (ví dụ: chỉ số IoU và độ lệch độ sâu `depth_diff`).

---

## 4. Các Mã Lỗi Thường Gặp (Error Codes)

| Mã lỗi HTTP | Tình huống xảy ra | Chi tiết thông báo (`detail`) |
|:---:|:---|:---|
| **400 Bad Request** | Tải lên file sai định dạng (không phải JPEG, PNG, WebP) | `"Invalid file type. Only JPEG, PNG, and WebP images are allowed."` |
| **400 Bad Request** | File tải lên không có dữ liệu (dung lượng 0 byte) | `"Uploaded file is empty."` |
| **400 Bad Request** | File bị hỏng, OpenCV không thể giải mã thành ảnh | `"Failed to decode image."` |
| **500 Internal Error** | Lỗi xử lý suy luận trong mô hình | `"Inference failed: <chi tiết lỗi>"` |

---

## 5. Ví Dụ Gọi API

### 5.1 Sử dụng cURL (Command Line)

```bash
curl -X POST "http://localhost:8080/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@data/samples/classroom.png;type=image/png"
```

### 5.2 Sử dụng Python (`requests`)

```python
import requests

url = "http://localhost:8080/api/predict"
image_path = "data/samples/classroom.png"

with open(image_path, "rb") as f:
    files = {"image": ("classroom.png", f, "image/png")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print(f"Số lượng học sinh đếm được: {data['people_count']}")
    print(f"Số box bị loại bỏ: {data['suppressed_count']}")
    
    # Duyệt qua các học sinh được nhận diện
    for det in data["detections"]:
        if det["kept"]:
            print(f"- Box #{det['box_id']}: Tọa độ {det['bbox']}, Độ tin cậy: {det['confidence']:.2f}")
else:
    print(f"Lỗi {response.status_code}: {response.json()['detail']}")
```

### 5.3 Sử dụng JavaScript (Fetch API trên trình duyệt hoặc Node.js)

```javascript
const formData = new FormData();
// fileInput là thẻ <input type="file" /> từ giao diện
formData.append("image", fileInput.files[0]);

fetch("http://localhost:8080/api/predict", {
  method: "POST",
  body: formData,
})
  .then((res) => {
    if (!res.ok) {
      throw new Error(`HTTP error! Status: ${res.status}`);
    }
    return res.json();
  })
  .then((data) => {
    console.log("Số lượng học sinh:", data.people_count);
    console.log("Danh sách detections:", data.detections);
  })
  .catch((err) => console.error("Lỗi gửi ảnh:", err));
```

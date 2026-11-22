# Ghi chú PaDiM (B6)

Nguồn: Defard, Setkov, Loesch, Audigier, *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, arXiv:2011.08785.

## 1. Ý tưởng

- Học một chiều (one-class): chỉ dùng ảnh tốt để huấn luyện.
- Dùng backbone đã học sẵn (WideResNet50) để biến mỗi vùng ảnh (patch) thành một vector embedding.
- Tại **mỗi vị trí** `(i, j)` của lưới, mô tả "vùng bình thường trông thế nào" bằng một Gaussian đa biến, học từ N ảnh tốt.
- Ảnh test: với mỗi vị trí, đo vector embedding lệch khỏi Gaussian bình thường bao xa (khoảng cách Mahalanobis). Gộp mọi vị trí thành **bản đồ bất thường**; điểm cả ảnh = giá trị lớn nhất trên bản đồ.

## 2. Thứ được lưu ở mỗi vị trí

Với N ảnh train (bottle: N = 188), vị trí `(i, j)` có N vector `x_ij`, mỗi vector C chiều.

- Vector trung bình: `mu_ij = (1/N) * sum_k x_ij^k`, shape `(C,)`.
- Ma trận hiệp phương sai: `Sigma_ij = 1/(N-1) * sum_k (x_ij^k - mu_ij)(x_ij^k - mu_ij)^T + eps * I`, shape `(C, C)`.
  Thành phần `eps * I` (bài báo dùng eps = 0.01) giúp Sigma luôn khả nghịch. Cần thiết vì N = 188 nhỏ hơn C = 1536, nên Sigma ước lượng thuần sẽ bị suy biến (không khả nghịch).

Điểm bất thường của vector `x_ij` ở ảnh test:

```
M(x_ij) = sqrt( (x_ij - mu_ij)^T * Sigma_ij^{-1} * (x_ij - mu_ij) )
```

## 3. Ví dụ tính tay (C = 2, N = 4)

Dữ liệu train tại một vị trí: `x1 = (1,2)`, `x2 = (3,2)`, `x3 = (2,4)`, `x4 = (2,0)`.

**Trung bình:** `mu = (2, 2)`.

**Độ lệch:** `(-1,0)`, `(1,0)`, `(0,2)`, `(0,-2)`.

**Hiệp phương sai** (chia N - 1 = 3), tổng bốn ma trận `(x_k - mu)(x_k - mu)^T` là `[[2,0],[0,8]]`, nên:

```
Sigma = [[2/3, 0 ],      Sigma^{-1} = [[3/2, 0  ],
         [0,   8/3]]                   [0,   3/8]]
```

Độ lệch chuẩn: chiều x là `sqrt(2/3) ≈ 0.82`, chiều y là `sqrt(8/3) ≈ 1.63`.

**Hai điểm test:** `A = (4, 2)` và `B = (2, 4)`.

| | Độ lệch so với tâm | Euclid | Mahalanobis |
|---|---|---|---|
| A | (2, 0) | `sqrt(2^2 + 0^2) = 2` | `sqrt(2^2 * 3/2) = sqrt(6) ≈ 2.449` |
| B | (0, 2) | `sqrt(0^2 + 2^2) = 2` | `sqrt(2^2 * 3/8) = sqrt(1.5) ≈ 1.225` |

Kiểm tra bằng độ lệch chuẩn: `2 / 0.82 ≈ 2.45` (A) và `2 / 1.63 ≈ 1.22` (B), trùng với Mahalanobis.

## 4. Vì sao dùng Mahalanobis

A và B cách tâm bằng nhau theo Euclid (đều bằng 2), nhưng A đáng ngờ hơn B. A lệch theo chiều x, chiều này bình thường dao động rất ít (độ lệch chuẩn 0.82), nên lệch 2 là rất lớn so với bình thường (khoảng 2.45 độ lệch chuẩn). B lệch theo chiều y, chiều này bình thường dao động nhiều (độ lệch chuẩn 1.63), nên lệch 2 chỉ khoảng 1.22 độ lệch chuẩn.

Khoảng cách Mahalanobis đo độ lệch bằng đơn vị "số độ lệch chuẩn" của từng hướng, nên công bằng giữa các chiều có mức dao động khác nhau. Khoảng cách Euclid coi mọi chiều như nhau.

## 5. Chi phí bộ nhớ và giảm chiều

Dự án này dùng `layer2 + layer3` của WideResNet50: C = 512 + 1024 = 1536. Lưới 28 x 28 = 784 vị trí.

- Một ma trận Sigma: `1536^2 = 2,359,296` số (khoảng 9.4 MB với float32).
- Toàn bộ 784 vị trí: khoảng 1.85 tỉ số, khoảng **7.4 GB**. Quá nặng.

Bài báo giảm chiều bằng cách chọn ngẫu nhiên Rd chiều của embedding (Rd = 550 cho WideResNet50-2), dùng cho cả train và test. Với Rd = 550 thì Sigma cho cả lưới còn khoảng 0.95 GB; với Rd = 100 còn khoảng 31 MB.

## 6. Khác biệt giữa dự án này và bài báo

- Bài báo dùng `layer1 + layer2 + layer3`; dự án này dùng `layer2 + layer3` (C = 1536 thay vì 1792 với WideResNet50).
- Khi ghép các tầng có độ phân giải khác nhau, mỗi vector của tầng thấp độ phân giải được lặp lại cho các vị trí của lưới lớn hơn (`layer3` 14 x 14 lặp thành 28 x 28).
- Ghi vào `docs/decision-log.md` nếu chốt dùng `layer2 + layer3` cho PaDiM mini (B7).

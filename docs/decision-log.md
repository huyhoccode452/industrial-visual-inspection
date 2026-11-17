# Decision log

## 1. Không commit dataset, chỉ commit script chuẩn bị dữ liệu
- Bối cảnh: MVTec AD rất nặng (file zip 5.3 GB, 5 category giải nén ra còn khoảng 1.4 GB). Mình từng lỡ `git add` khi `.gitignore` chưa chặn `data/datasets/`, làm thư mục `.git` phình lên 527 MB dù chưa commit ảnh nào.
- Lựa chọn: `.gitignore` chặn `data/datasets/` và `*.zip`. Repo chỉ chứa `scripts/prepare_mvtec.py`; người dùng tự tải zip từ trang chính thức của MVTec AD rồi chạy script (README sẽ hướng dẫn).
- Lý do: repo nhẹ, clone nhanh, không phụ thuộc vào link Drive cá nhân có thể hỏng. Dữ liệu luôn lấy từ nguồn gốc nên cũng tránh vấn đề phát lại dataset.
- Phương án bỏ qua:
  - Commit thẳng ảnh vào repo: repo phình to, clone chậm, và dữ liệu đã vào lịch sử git thì rất khó gỡ ra.
  - Git LFS: thêm một công cụ phải cài, và có hạn mức dung lượng/băng thông.
  - Mirror lên Google Drive cá nhân: license MVTec AD (CC BY-NC-SA 4.0) cho phép chia sẻ lại nhưng chỉ vì mục đích phi thương mại, phải ghi nguồn và nếu chia sẻ bản đã chỉnh sửa thì giữ cùng loại license. Làm được, nhưng thêm việc phải duy trì link và tuân thủ điều kiện; dẫn người dùng về nguồn gốc thì đơn giản và an toàn hơn.

## 2. Không ghi local tag của PyTorch vào requirements
- Bối cảnh: torch trong venv là `2.15.0+cpu`, trong khi `requirements.txt` cũ (dump từ môi trường khác) ghi `torch==2.5.1+cu121`. Máy mình có GPU RTX 2050 nhưng đang cài bản CPU.
- Lựa chọn: ghi `torch==2.15.0` và `torchvision==0.29.2`, không ghi `+cpu` hay `+cu121`.
- Lý do: local tag gắn với một cấu hình phần cứng cụ thể. Người clone repo về mà có GPU, hoặc CUDA khác phiên bản, sẽ không cài được hoặc cài nhầm bản.
- Phương án bỏ qua: ghi đầy đủ local tag, vì chỉ chạy được trên máy giống máy mình.
- Hệ quả: README phải hướng dẫn người có GPU cài torch theo trang pytorch.org.

## 3. Script chuẩn bị dữ liệu phải báo lỗi khi category không có file
- Bối cảnh: bản script gốc khi gõ sai tên category (ví dụ `bottles`) chỉ in "0 file" rồi báo "Xong!", không có lỗi nào.
- Lựa chọn: nếu một category không có file nào trong zip thì dừng ngay với thông báo rõ ràng.
- Lý do: thất bại im lặng nguy hiểm hơn lỗi có thông báo. Nếu không báo, mình chỉ phát hiện dữ liệu thiếu khi đang train model.
- Phương án bỏ qua:
  - Giữ nguyên script gốc (không kiểm tra).
  - Chỉ in cảnh báo rồi chạy tiếp: dễ bị bỏ sót trong log.
- Việc còn mở: kiểm tra tất cả category trước khi giải nén bất kỳ cái nào, để không bị giải nén dở dang.

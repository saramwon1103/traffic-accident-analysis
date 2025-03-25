📂 data/                # Chứa dữ liệu về tai nạn giao thông  
│   ├── raw/           # Dữ liệu gốc (chưa xử lý)  
│   ├── processed/     # Dữ liệu đã tiền xử lý  
│   ├── graphs/        # Đồ thị mạng lưới giao thông  
│   └── metadata/      # Mô tả dữ liệu  
│  
📂 notebooks/           # Chứa notebook Jupyter  
│   ├── 01_exploration.ipynb   # Khám phá dữ liệu  
│   ├── 02_preprocessing.ipynb # Tiền xử lý dữ liệu  
│   ├── 03_modeling.ipynb      # Huấn luyện mô hình GNN  
│   └── 04_evaluation.ipynb    # Đánh giá mô hình  
│  
📂 src/                # Chứa mã nguồn chính  
│   ├── data_loader.py       # Load dữ liệu  
│   ├── preprocess.py        # Xử lý dữ liệu  
│   ├── graph_builder.py     # Xây dựng đồ thị từ dữ liệu  
│   ├── train.py             # Huấn luyện mô hình GNN  
│   ├── evaluate.py          # Đánh giá mô hình  
│   └── visualize.py         # Trực quan hóa dữ liệu  
│  
📂 tests/              # Chứa code kiểm thử  
│   ├── test_data_loader.py  
│   ├── test_preprocess.py  
│   ├── test_graph_builder.py  
│   └── test_model.py  
│  
📂 docs/               # Tài liệu dự án  
│   ├── report.md         # Báo cáo phân tích  
│   ├── references.md     # Tài liệu tham khảo  
│   ├── API_documentation.md  # Tài liệu API (nếu có)  
│   └── presentation.pptx  # Slide thuyết trình  
│  
📂 models/             # Chứa mô hình đã huấn luyện  
│   ├── GAT_model.pth        # Checkpoint của GNN  
│   ├── results.json         # Kết quả dự đoán  
│   └── model_config.json    # Cấu hình mô hình  
│  
📂 utils/              # Các công cụ hỗ trợ  
│   ├── helper.py           # Các hàm tiện ích  
│   ├── config.py           # Cấu hình tham số  
│   └── logger.py           # Ghi log  
│  
.gitignore            # Loại trừ file không cần push  
README.md             # Mô tả dự án  
requirements.txt      # Thư viện cần thiết  
LICENSE               # Giấy phép dự án (MIT, Apache, v.v.)  
setup.py              # Cấu hình package Python (nếu cần)  

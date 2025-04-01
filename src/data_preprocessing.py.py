import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Đọc dataset từ file CSV
df = pd.read_csv("D:/traffic_accidents.csv", encoding='utf-8')  # Thử với utf-8 hoặc ISO-8859-1

# Xóa cột "crash_date" nếu tồn tại
if "crash_date" in df.columns:
    df = df.drop(columns=["crash_date"])

# Các cột cần mã hóa
categorical_columns = [
    "traffic_control_device", "weather_condition", "lighting_condition", "first_crash_type", 
    "trafficway_type", "alignment", "roadway_surface_cond", "road_defect", "crash_type", 
    "intersection_related_i", "damage", "prim_contributory_cause", "most_severe_injury"
]

# Dictionary để lưu các label encoder
decoders = {}

# Mã hóa từng cột
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Thay đổi trực tiếp giá trị trong cột
    decoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))  # Lưu mapping cho báo cáo

# Xuất báo cáo mã hóa
encoding_report = """Báo cáo Mã hóa Categorical Data\n\n"""
for col, mapping in decoders.items():
    encoding_report += f"Cột: {col}\n"
    for key, value in mapping.items():
        encoding_report += f"  {key}: {value}\n"
    encoding_report += "\n"

# Xuất báo cáo mã hoá ra file
with open("D:/encoding_report.txt", "w", encoding="utf-8") as f:
    f.write(encoding_report)

# Lưu dataset đã mã hóa
encoded_file = "D:/encoded_dataset.csv"
df.to_csv(encoded_file, index=False)

print(f"Mã hóa hoàn tất! Dataset đã lưu vào: {encoded_file}")
print(f"Báo cáo mã hóa đã lưu vào: encoding_report.txt")

import pandas as pd
import numpy as np

# Số dòng dữ liệu mới bạn muốn tạo
n_samples = 100

# Tạo dữ liệu mẫu
df_new = pd.DataFrame({
    'traffic_control_device': np.random.choice(['SIGNAL', 'STOP', 'NONE'], n_samples),
    'weather_condition': np.random.choice(['CLEAR', 'RAIN', 'SNOW'], n_samples),
    'lighting_condition': np.random.choice(['DAYLIGHT', 'DARK'], n_samples),
    'first_crash_type': np.random.choice(['REAR END', 'ANGLE', 'SIDESWIPE'], n_samples),
    'trafficway_type': np.random.choice(['ONE-WAY', 'TWO-WAY'], n_samples),
    'alignment': np.random.choice(['STRAIGHT', 'CURVED'], n_samples),
    'roadway_surface_cond': np.random.choice(['DRY', 'WET'], n_samples),
    'road_defect': np.random.choice(['NONE', 'HOLE'], n_samples),
    'crash_type': np.random.choice(['COLLISION', 'NON-COLLISION'], n_samples),
    'intersection_related_i': np.random.choice(['Y', 'N'], n_samples),
    'damage': np.random.choice(['OVER $1500', 'UNDER $1500'], n_samples),
    'prim_contributory_cause': np.random.choice(['DISTRACTION', 'SPEEDING'], n_samples),
    'num_units': np.random.randint(1, 5, n_samples),
    'most_severe_injury': np.random.choice(['NO INJURY', 'INCAPACITATING INJURY'], n_samples),
    'injuries_total': np.random.randint(0, 4, n_samples),
    'injuries_fatal': np.random.randint(0, 1, n_samples),
    'injuries_incapacitating': np.random.randint(0, 2, n_samples),
    'injuries_non_incapacitating': np.random.randint(0, 2, n_samples),
    'injuries_reported_not_evident': np.random.randint(0, 2, n_samples),
    'injuries_no_indication': np.random.randint(0, 3, n_samples),
    'crash_hour': np.random.randint(0, 24, n_samples),
    'crash_day_of_week': np.random.randint(1, 8, n_samples),   # 1–7
    'crash_month': np.random.randint(1, 13, n_samples),        # 1–12
})

# Lưu file CSV
df_new.to_csv('new_data_simulated.csv', index=False)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Đọc dataset từ file CSV
df = pd.read_csv(r"D:\Năm 3 - HK2\Mạng xã hội\traffic-accident-analysis\data\new_data_simulated.csv", encoding='utf-8')  # Thử với utf-8 hoặc ISO-8859-1

# Xóa cột "crash_date" nếu tồn tại
if "crash_date" in df.columns:
    df = df.drop(columns=["crash_date"])

if "damage" in df.columns:
    df = df.drop(columns=["damage"])

# Các cột cần mã hóa
categorical_columns = [
    "traffic_control_device", "weather_condition", "lighting_condition", "first_crash_type", 
    "trafficway_type", "alignment", "roadway_surface_cond", "road_defect", "crash_type", 
    "intersection_related_i", "prim_contributory_cause", "most_severe_injury"
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

# Lưu dataset đã mã hóa
df.to_csv('new_data_cleaned.csv', index=False)

import torch 
import pandas as pd 
import networkx as nx 
from itertools import combinations
from torch_geometric.data import Data 
from model import GAT

# Đọc dataset
file_path = r'D:\Năm 3 - HK2\Mạng xã hội\traffic-accident-analysis\data\new_data_cleaned.csv'
df = pd.read_csv(file_path)

import networkx as nx
from itertools import combinations

G = nx.Graph()
for index, row in df.iterrows():
    G.add_node(index, **row.to_dict())

def is_similar(accident1, accident2):
    return (
        abs(accident1['crash_hour'] - accident2['crash_hour']) <= 1 or
        accident1['crash_month'] == accident2['crash_month'] or
        accident1['crash_day_of_week'] == accident2['crash_day_of_week'] or
        accident1['trafficway_type'] == accident2['trafficway_type'] or
        accident1['first_crash_type'] == accident2['first_crash_type'] or
        accident1['injuries_no_indication'] == accident2['injuries_no_indication']
    )

for u, v in combinations(G.nodes(data=True), 2):
    if is_similar(u[1], v[1]):
        G.add_edge(u[0], v[0])

print(f"✅ Đã tạo đồ thị với {G.number_of_nodes()} nút và {G.number_of_edges()} cạnh.")

import torch
from torch_geometric.data import Data

def networkx_to_pyg_inference(G):
    feature_attrs = list(next(iter(G.nodes(data=True)))[1].keys())

    features = [
        [float(data[attr]) for attr in feature_attrs]
        for _, data in G.nodes(data=True)
    ]

    x = torch.tensor(features, dtype=torch.float)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor(
        [[node_mapping[u], node_mapping[v]] for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    return Data(x=x, edge_index=edge_index)

data_new = networkx_to_pyg_inference(G)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GAT(in_features=22, hidden_dim=16, out_features=3, heads=8).to(device)
model.load_state_dict(torch.load('gat_model.pth', map_location=device))
model.eval()

with torch.no_grad():
    out = model(data_new.x.to(device), data_new.edge_index.to(device))
    pred = out.argmax(dim=1)

# Mapping số về chuỗi như trong cột damage gốc
damage_mapping = {
    0: "$500 OR LESS",
    1: "$501 - $1,500",
    2: "OVER $1,500"
}

# Tạo cột mới với nhãn dạng chuỗi
df['predicted_damage_label'] = df['predicted_damage'].map(damage_mapping)

# Hiển thị 5 dòng đầu để kiểm tra
print(df[['predicted_damage', 'predicted_damage_label']])

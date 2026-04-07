import torch
from ultralytics import YOLO
import os

model = YOLO("best.pt")  

source_path = r"D:\Desktop\Helmet Detection Model\Helmet_Detection_DataSet\test\images"

output_path = r"D:\Desktop\Helmet Detection Model\Helmet_Detection_DataSet\test_results"

results = model.predict(
    source=source_path,
    conf=0.5,
    save=True,
    project=output_path,
    name="predict"  
)

print("Detection completed successfully!")
print(f"Results saved in: {os.path.join(output_path, 'predict')}")
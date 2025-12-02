# filename: yolo_train.py
from ultralytics import YOLO

def train_yolo(
    data_yaml: str = "data.yaml",
    base_model: str = "yolov8s.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 16,
    device: int | str = 0,
    project: str = "runs/train",
    name: str = "tennis_yolo",
):
    """
    Train YOLO on your Roboflow-exported dataset.
    data.yaml should point to train/val image folders and class names:
      train: path/to/train/images
      val:   path/to/val/images
      nc: 3
      names: ["player", "racket", "ball"]
    """
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        lr0=0.001,
        patience=20,
        mosa ic=1.0,
        hsv_h=0.02,
        hsv_s=0.3,
        hsv_v=0.3,
    )

if __name__ == "__main__":
    train_yolo()

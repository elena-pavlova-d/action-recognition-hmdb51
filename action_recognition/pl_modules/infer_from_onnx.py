import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.io import read_video
from torchvision.transforms import Resize

video_path = Path(
    "data/train/cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi"
)
model_path = Path("video_action_model.onnx")
class_names_json_path = Path("index_to_class.json")
seq_len = 16
target_size = (128, 128)


def preprocess_video(video_path, seq_len=16, target_size=(128, 128)):
    vframes, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    vframes = vframes.float() / 255.0

    resize = Resize(target_size)
    vframes = torch.stack([resize(frame) for frame in vframes])

    total_frames = vframes.shape[0]

    if total_frames < seq_len:
        padding = seq_len - total_frames
        last_frame = vframes[-1].unsqueeze(0)
        vframes = torch.cat([vframes, last_frame.repeat(padding, 1, 1, 1)], dim=0)
    else:
        indices = torch.linspace(0, total_frames - 1, steps=seq_len).long()
        vframes = vframes[indices]

    vframes = vframes.unsqueeze(0).numpy()
    return vframes.astype(np.float32)


def run_onnx_inference(model_path, input_tensor):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": input_tensor})
    logits = outputs[0]
    probs = torch.softmax(torch.tensor(logits), dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs.squeeze().tolist()


def main():
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX модель не найдена: {model_path}")

    video_tensor = preprocess_video(
        video_path, seq_len=seq_len, target_size=target_size
    )
    pred_class, probs = run_onnx_inference(model_path, video_tensor)

    print(f"\n[RESULT] Предсказанный класс (индекс): {pred_class}")

    if class_names_json_path.exists():
        with open(class_names_json_path, "r", encoding="utf-8") as f:
            index_to_class = json.load(f)
        class_name = index_to_class.get(str(pred_class), None)
        if class_name:
            print(f"[INFO] Название класса: {class_name}")
        else:
            print("[INFO] Название класса не найдено в JSON")
    else:
        print("[INFO] JSON с названиями классов не найден")


if __name__ == "__main__":
    main()

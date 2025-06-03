import hydra
import torch
from omegaconf import DictConfig

from action_recognition.pl_modules.model import VideoActionModel


def export_to_onnx(ckpt_path, output_path, input_shape, num_classes):
    model = VideoActionModel.load_from_checkpoint(
        ckpt_path, num_classes=num_classes, learning_rate=1e-4
    )
    model.eval()

    dummy_input = torch.randn(*input_shape)
    print(f"dummy_input shape: {dummy_input.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[INFO] Модель экспортирована в: {output_path}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    ckpt_path = cfg.export.ckpt_path
    output_path = cfg.export.output_path
    num_classes = cfg.model.num_classes
    input_shape = tuple(cfg.export.input_shape)

    export_to_onnx(ckpt_path, output_path, input_shape, num_classes)


if __name__ == "__main__":
    main()

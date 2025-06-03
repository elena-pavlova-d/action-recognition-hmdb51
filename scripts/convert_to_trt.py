import sys
from pathlib import Path

import tensorrt as trt


def build_engine(onnx_path: Path, engine_path: Path, precision: str = "fp16") -> None:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with onnx_path.open("rb") as model_file:
        if not parser.parse(model_file.read()):
            for error_idx in range(parser.num_errors):
                print(parser.get_error(error_idx))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    engine = builder.build_engine(network, config)
    with engine_path.open("wb") as f:
        f.write(engine.serialize())

    print(f"Saved engine to {engine_path}")


if __name__ == "__main__":
    build_engine(
        onnx_path=Path(sys.argv[1]),
        engine_path=Path(sys.argv[2]),
        precision=sys.argv[3] if len(sys.argv) > 3 else "fp16",
    )

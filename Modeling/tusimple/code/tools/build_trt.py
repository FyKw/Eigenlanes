# build_trt.py

import tensorrt as trt
#depricated?
def build_trt_engine(onnx_path, plan_path="engine_fp16_sparse.plan", workspace_gb=2):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse err:", parser.get_error(i))
            raise SystemExit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb<<30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # 👇 This is the key for sparse Tensor Cores
    if hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    engine = builder.build_engine(network, config)
    with open(plan_path, "wb") as f:
        f.write(engine.serialize())
    print("✅ TensorRT engine saved to", plan_path)

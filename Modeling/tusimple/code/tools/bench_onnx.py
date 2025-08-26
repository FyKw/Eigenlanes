# tools/bench_onnx.py
import time, numpy as np, onnxruntime as ort
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def bench_onnx_cuda(onnx_path, H, W, iters=200, warmup=50, fp16=None):
    sess = ort.InferenceSession(
        onnx_path,
        providers=[("CUDAExecutionProvider", {})] if ort.get_device()=="GPU" else ["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    input_name = inp.name
    onnx_dtype = inp.type  # e.g., 'tensor(float)' or 'tensor(float16)'

    # decide dtype: if fp16 is None, follow the model; else force (and cast if compatible)
    if fp16 is None:
        dtype = np.float16 if onnx_dtype == "tensor(float16)" else np.float32
    else:
        dtype = np.float16 if fp16 else np.float32
        # if you force fp16 but the model expects float32, ORT will error.
        # So if mismatch, fall back to the model dtype:
        if (dtype == np.float16 and onnx_dtype == "tensor(float)") or \
           (dtype == np.float32 and onnx_dtype == "tensor(float16)"):
            print(f"⚠️ Overriding requested dtype to match ONNX ({onnx_dtype}).")
            dtype = np.float16 if onnx_dtype == "tensor(float16)" else np.float32

    x = np.random.randn(1, 3, H, W).astype(dtype)

    # warmup
    for _ in range(warmup):
        _ = sess.run(None, {input_name: x})

    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(None, {input_name: x})
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()

    ms = (time.perf_counter() - t0) * 1000.0 / iters
    prov = sess.get_providers()[0] if isinstance(sess.get_providers()[0], str) else sess.get_providers()[0][0]
    print(f"⏱ ONNX Runtime ({prov}) avg: {ms:.3f} ms  | dtype: {dtype}")
    return ms

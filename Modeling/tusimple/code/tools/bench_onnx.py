import os, time, csv, datetime, numpy as np, onnxruntime as ort
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def _ensure_csv_header(csv_path: str):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "model_name", "dtype", "avg_ms", "filesize_bytes"
            ])

def bench_onnx_cuda(onnx_path, H, W, iters=200, warmup=50, fp16=None,
                    csv_log_path=None, model_name="", tag_provider=None):
    """
    fp16=None => auto-match ONNX input dtype; fp16=True/False => force (falls back if mismatch)
    If csv_log_path is provided, append a log row.
    """
    providers = [("CUDAExecutionProvider", {})] if ort.get_device()=="GPU" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp = sess.get_inputs()[0]
    input_name = inp.name
    onnx_dtype = inp.type  # 'tensor(float)' or 'tensor(float16)'

    # decide dtype
    if fp16 is None:
        dtype = np.float16 if onnx_dtype == "tensor(float16)" else np.float32
    else:
        dtype = np.float16 if fp16 else np.float32
        # if forced dtype mismatches model, follow ONNX
        if (dtype is np.float16 and onnx_dtype == "tensor(float)") or \
           (dtype is np.float32 and onnx_dtype == "tensor(float16)"):
            dtype = np.float16 if onnx_dtype == "tensor(float16)" else np.float32

    x = np.random.randn(1, 3, H, W).astype(dtype)

    # warmup
    for _ in range(warmup):
        _ = sess.run(None, {input_name: x})

    # timed
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(None, {input_name: x})
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()

    ms = (time.perf_counter() - t0) * 1000.0 / iters
    provs = sess.get_providers()
    prov = provs[0][0] if isinstance(provs[0], tuple) else provs[0]
    if tag_provider:
        prov = f"{prov}:{tag_provider}"

    print(f"⏱ ONNX Runtime ({prov}) avg: {ms:.3f} ms | dtype: {dtype} | {os.path.basename(onnx_path)}")

    # optional CSV log
    if csv_log_path:
        _ensure_csv_header(csv_log_path)
        gpu = (torch.cuda.get_device_name(0) if (_HAS_TORCH and torch.cuda.is_available()) else "CPU")
        filesize = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else -1
        with open(csv_log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                model_name,
                "fp16" if dtype==np.float16 else "fp32",
                round(ms, 3),
                filesize,
            ])
    return ms

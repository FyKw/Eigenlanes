import torch, torch.onnx

def export_onnx(model, H, W, onnx_path, precision="fp32"):
    model = model.eval().to("cpu")
    dummy = torch.randn(1, 3, H, W)
    if precision == "fp16":
        model = model.half()
        dummy = dummy.half()

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)  # dict
            return (out["prob_logit"], out["height_prob_logit"])

    torch.onnx.export(
        Wrap(model), (dummy,), onnx_path,
        opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=["prob_logit","height_prob_logit"],
        dynamic_axes=None
    )
    print("✅ ONNX exported ->", onnx_path)
    return onnx_path

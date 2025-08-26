import torch, torch.onnx

def export_onnx(model, H, W, onnx_path="model.onnx"):
    model = model.eval().to("cpu")

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)  # your forward returns a dict
            return (out["prob_logit"], out["height_prob_logit"])  # tuple

    dummy = torch.randn(1, 3, H, W)
    torch.onnx.export(
        Wrap(model), (dummy,), onnx_path,
        opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=["prob_logit","height_prob_logit"],
        dynamic_axes=None  # keep static for speed
    )
    print("✅ ONNX exported ->", onnx_path)

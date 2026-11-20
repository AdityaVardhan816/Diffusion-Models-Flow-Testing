"""Interactive Presentation and Live Demonstration Server for Diffusion vs Flow Matching."""

import os
import io
import json
import time
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import torch

from src.model import CompactUNet
from src.diffusion import GaussianDiffusion
from src.flow_matching import FlowMatching
from src.reflow import Reflow
from src.evaluate import save_sample_grid
from torchvision.utils import make_grid
from PIL import Image

# Global model cache
MODELS = {}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_backbone(ckpt, device):
    b = CompactUNet(1, 1, 32, (1, 2, 4), 128).to(device)
    raw_state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    state = {}
    for k, v in raw_state.items():
        if k.startswith("model."):
            state[k[6:]] = v
        else:
            state[k] = v
    b.load_state_dict(state)
    b.eval()
    return b


def load_models():
    print(f"Loading trained models on device: {DEVICE}...")
    
    # 1. Flow Matching (1-RF)
    p_fm = "results/flow_matching_baseline/checkpoint_latest.pt"
    if os.path.exists(p_fm):
        try:
            ckpt = torch.load(p_fm, map_location=DEVICE)
            b = _load_backbone(ckpt, DEVICE)
            MODELS["1rf"] = FlowMatching(b)
            print("  ✓ Loaded 1-Rectified Flow model")
        except Exception as e:
            print(f"  Failed loading 1-RF: {e}")

    # 2. Reflow (2-RF)
    p_rf = "results/reflow_baseline/checkpoint_latest.pt"
    if os.path.exists(p_rf):
        try:
            ckpt = torch.load(p_rf, map_location=DEVICE)
            b = _load_backbone(ckpt, DEVICE)
            MODELS["2rf"] = Reflow(b)
            print("  ✓ Loaded 2-Rectified Flow model")
        except Exception as e:
            print(f"  Failed loading 2-RF: {e}")

    # 3. Diffusion
    p_diff = "results/diffusion_baseline/checkpoint_latest.pt"
    if os.path.exists(p_diff):
        try:
            ckpt = torch.load(p_diff, map_location=DEVICE)
            b = CompactUNet(1, 1, 32, (1, 2, 4), 128).to(DEVICE)
            diff = GaussianDiffusion(b, 1000).to(DEVICE)
            raw_state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            diff.load_state_dict(raw_state)
            diff.eval()
            MODELS["diffusion"] = diff
            print("  ✓ Loaded Diffusion model")
        except Exception as e:
            print(f"  Failed loading Diffusion: {e}")


class PresentationHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {
                "status": "online",
                "device": str(DEVICE),
                "author": "Aditya Vardhan Singh",
                "loaded_models": list(MODELS.keys())
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if parsed.path == "/api/generate":
            params = parse_qs(parsed.query)
            model_key = params.get("model", ["2rf"])[0].lower()
            steps = int(params.get("steps", [1])[0])
            seed = int(params.get("seed", [42])[0])
            batch_size = int(params.get("batch_size", [16])[0])

            if model_key not in MODELS:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Model '{model_key}' not loaded"}).encode("utf-8"))
                return

            t0 = time.perf_counter()
            torch.manual_seed(seed)

            with torch.no_grad():
                shape = (batch_size, 1, 32, 32)
                if model_key == "diffusion":
                    # eta = 1.0 if steps >= 50 else 0.0
                    samples, nfe = MODELS["diffusion"].sample_ddim(shape, device=DEVICE, steps=steps, eta=1.0 if steps >= 50 else 0.0)
                elif model_key == "2rf":
                    samples, nfe = MODELS["2rf"].sample_euler(shape, device=DEVICE, steps=steps)
                else:
                    samples, nfe = MODELS["1rf"].sample_euler(shape, device=DEVICE, steps=steps)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Convert to grid PNG
            imgs = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
            grid = make_grid(imgs.cpu(), nrow=4, padding=2)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            
            im = Image.fromarray(ndarr)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("X-Generation-Time-Ms", f"{elapsed_ms:.1f}")
            self.send_header("X-NFE", str(nfe))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(img_bytes)
            return

        # Serve static files (index.html, results, etc.)
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description="Presentation and Demo Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    load_models()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), PresentationHandler)
    print(f"\n=======================================================")
    print(f"Presentation Interface Running at: http://localhost:{args.port}")
    print(f"Author: Aditya Vardhan Singh")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()


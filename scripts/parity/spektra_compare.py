"""Compare Python vs Rust spektrafilm on a DNG.

Step 1: decode DNG via rawpy → linear ACES (Python) and write a
downsampled linear-ACES 32-bit float TIFF that both impls can consume.
Step 2: run Python simulate on that TIFF.
Step 3: print a note for the Rust caller to render the same TIFF.

Usage:
  python spektra_compare.py <raw.dng> <out_dir> [film] [paper] [max_dim]
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from spektrafilm import init_params, simulate
from spektrafilm.utils.raw_file_processor import load_and_process_raw_file


def downsample(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(h, w) / max_dim
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    # Simple area-average via Pillow with 32-bit float pivot.
    arr = (img.clip(0.0) * 1e3).astype(np.float32)
    out = np.zeros((new_h, new_w, 3), dtype=np.float32)
    for c in range(3):
        pil = Image.fromarray(arr[:, :, c], mode="F")
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        out[:, :, c] = np.asarray(pil)
    return (out / 1e3).astype(np.float32)


def main():
    raw_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    film = sys.argv[3] if len(sys.argv) > 3 else "kodak_gold_200"
    paper = sys.argv[4] if len(sys.argv) > 4 else "fujifilm_crystal_archive_typeii"
    max_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 1500

    print(f"[py] decoding {raw_path.name} via rawpy → linear ACES2065-1 …", flush=True)
    t = time.perf_counter()
    rgb_aces = load_and_process_raw_file(raw_path, output_colorspace="ACES2065-1")
    print(
        f"     done in {(time.perf_counter() - t) * 1000:.0f} ms  shape={rgb_aces.shape}  "
        f"dtype={rgb_aces.dtype}",
        flush=True,
    )

    print(f"[py] downsampling to ≤ {max_dim}px on the long edge …", flush=True)
    t = time.perf_counter()
    small = downsample(rgb_aces, max_dim)
    print(
        f"     {small.shape}  range=[{small.min():.4f}, {small.max():.4f}]  "
        f"in {(time.perf_counter() - t) * 1000:.0f} ms",
        flush=True,
    )

    # Save as 32-bit float TIFF via OpenImageIO (spektrafilm's own
    # I/O dependency).
    import OpenImageIO as oiio

    tiff_path = out_dir / "input_linear_aces.tif"
    spec = oiio.ImageSpec(small.shape[1], small.shape[0], 3, "float")
    spec.attribute("oiio:ColorSpace", "linear")
    buf = oiio.ImageBuf(spec)
    buf.set_pixels(oiio.ROI(0, small.shape[1], 0, small.shape[0]), small)
    buf.write(str(tiff_path))
    print(f"[py] wrote {tiff_path}", flush=True)

    print(f"[py] building params (film={film}, paper={paper}) …", flush=True)
    params = init_params(film, paper)
    params.io.input_color_space = "ACES2065-1"
    params.io.input_cctf_decoding = False
    # Auto-exposure on (matches our GUI default).

    print(f"[py] simulating …", flush=True)
    t = time.perf_counter()
    out = simulate(small, params)
    print(f"     done in {(time.perf_counter() - t) * 1000:.0f} ms", flush=True)

    arr = np.clip(out, 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    out_path = out_dir / f"py_{film}.png"
    Image.fromarray(arr8).save(out_path)
    print(f"[py] saved {out_path}", flush=True)
    print(f"\nNow render the same TIFF with Rust:", flush=True)
    print(
        f"  cargo run --release -p spektrafilm-cli -- process {tiff_path} "
        f"--output {out_dir}/rs_{film}.png --film {film}",
        flush=True,
    )


if __name__ == "__main__":
    main()

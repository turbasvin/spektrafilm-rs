"""Stage-by-stage Python dump for parity bisection."""

import numpy as np
import OpenImageIO as oiio
from spektrafilm import init_params, digest_params
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.utils.spectral_upsampling import rgb_to_raw_hanatos2025
from spektrafilm.model.density_curves import interpolate_exposure_to_density
from spektrafilm.model.emulsion import compute_density_spectral
from spektrafilm.utils.conversions import density_to_light
from spektrafilm.model.illuminants import standard_illuminant
from opt_einsum import contract


def dump(name, arr):
    path = f"/tmp/cmp/py_stage_{name}.f64"
    arr.astype(np.float64).ravel().tofile(path)
    print(f"wrote {path}  ({arr.size} doubles)")


# Same input as Rust probe
inp = oiio.ImageBuf("/tmp/cmp/input_linear_aces.tif")
spec = inp.spec()
img = np.array(inp.get_pixels(oiio.TypeDesc("float"))).reshape(spec.height, spec.width, 3)
dump("00_input", img)

# Bare-chain params
p = init_params("kodak_portra_400", "kodak_portra_endura")
p.io.input_color_space = "ACES2065-1"
p.io.input_cctf_decoding = False
p.camera.auto_exposure = False
p.film_render.grain.active = False
p.film_render.halation.active = False
p.film_render.dir_couplers.active = False
p.print_render.glare.active = False
p.scanner.unsharp_mask = (0.0, 0.0)
p = digest_params(p)

# Build pipeline so internal services (LUT, enlarger) are populated
pipe = SimulationPipeline(p)
_ = pipe.process(img)  # warm up + populate calibration

# ── 1. Filming expose (RGB → log_raw) ──────────────────────────────
# Use the pipeline's lut_service which has the windowed tc_lut.
fs = pipe._filming_stage
log_raw = fs.expose(img)  # full pipeline expose: handles AE, EV, halation, lens blur, log10
dump("01_log_raw", log_raw)

# ── 2. Film develop (density curve interp; no DIR/grain since disabled) ─
density_cmy_film = fs.develop(log_raw)
dump("02_density_cmy_film", density_cmy_film)

# ── 3. Printing — expose then develop ──────────────────────────────
ps = pipe._printing_stage
log_raw_print = ps.expose(density_cmy_film)
density_cmy_print = ps.develop(log_raw_print)
dump("03_density_cmy_print", density_cmy_print)

# ── 4. Scanning → final RGB (sRGB-encoded + clipped) ───────────────
scan_rgb = pipe._scanning_stage.scan(density_cmy_print)
dump("04_scan_rgb", scan_rgb)

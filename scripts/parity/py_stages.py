"""Per-stage Python dump matching rs_stages.rs."""

import numpy as np
from spektrafilm import init_params, digest_params
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.profiles.io import load_profile
from spektrafilm.utils.spectral_upsampling import rgb_to_raw_hanatos2025

TEST_COLORS = [
    [0.18, 0.18, 0.18],
    [0.40, 0.15, 0.08],
    [0.10, 0.40, 0.10],
    [0.05, 0.10, 0.40],
    [0.80, 0.60, 0.40],
]


def main():
    params = init_params("kodak_gold_200", "fujifilm_crystal_archive_typeii")
    params.camera.auto_exposure = False
    params.io.input_color_space = "ACES2065-1"
    params.io.input_cctf_decoding = False
    params.film_render.grain.active = False
    params.film_render.halation.active = False
    params.film_render.dir_couplers.active = False
    params.print_render.glare.active = False
    params.scanner.unsharp_mask = (0.0, 0.0)

    params = digest_params(params)
    pipe = SimulationPipeline(params)

    # The pipeline's components are exposed on _stage_*; we need
    # whichever stage runs the bare chain.
    print(f"# Per-stage dump (Python reference, bare chain).")
    print(
        f"# c/m/y filters: {params.enlarger.c_filter_neutral} "
        f"{params.enlarger.m_filter_neutral} {params.enlarger.y_filter_neutral}"
    )

    # Build sensitivity for the hanatos call.
    film = params.film
    log_sens = np.array(film.data.log_sensitivity)
    sens = np.nan_to_num(10.0 ** log_sens)

    # Build a single-pixel image for each test color.
    for rgb in TEST_COLORS:
        img = np.array([[rgb]], dtype=np.float64)
        # Process through the whole pipeline (bare chain), grab intermediate
        # values via instrumented stages.
        # Easier: call simulate, but also re-run each stage by hand.

        # 1. Hanatos raw
        raw = rgb_to_raw_hanatos2025(
            img,
            sens,
            color_space="ACES2065-1",
            apply_cctf_decoding=False,
            reference_illuminant=film.info.reference_illuminant,
        )
        raw_v = raw[0, 0]
        # 2. log10
        log_raw = np.log10(np.maximum(raw, 1e-10))
        lr_v = log_raw[0, 0]
        # 3. Film density curve
        from spektrafilm.model.density_curves import interpolate_exposure_to_density
        density_curves = np.array(film.data.density_curves)
        log_exposure = np.array(film.data.log_exposure)
        # normalize density curves (subtract nanmin per channel)
        norm_curves = density_curves - np.nanmin(density_curves, axis=0, keepdims=True)
        density_cmy = interpolate_exposure_to_density(
            log_raw,
            norm_curves,
            log_exposure,
            params.film_render.density_curve_gamma,
        )
        d_v = density_cmy[0, 0]
        # 4. Print stage: drive directly via the SimulationPipeline's
        # internal stages so we can grab the intermediate density.
        # Easiest: replicate manually using the model functions.
        from spektrafilm.model.emulsion import compute_density_spectral
        from spektrafilm.model.illuminants import standard_illuminant
        from spektrafilm.model.color_filters import color_enlarger

        # Build the enlarger illuminant the way Python's pipeline does.
        base_illuminant = standard_illuminant(params.enlarger.illuminant)
        print_illu = color_enlarger(
            base_illuminant,
            (
                params.enlarger.c_filter_neutral,
                params.enlarger.m_filter_neutral + params.enlarger.m_filter_shift,
                params.enlarger.y_filter_neutral + params.enlarger.y_filter_shift,
            ),
        )

        film_channel_density = np.array(params.film.data.channel_density)
        film_base_density = np.array(params.film.data.base_density)
        print_sens = np.nan_to_num(
            10.0 ** np.array(params.print.data.log_sensitivity)
        )
        # Spectral density on the film
        density_spectral = compute_density_spectral(
            film_channel_density,
            density_cmy,
            base_density=film_base_density,
        )
        # Light transmitted by the film, weighted by enlarger illuminant
        # and integrated against print sensitivity → raw_print.
        # density_to_light = illu * 10**(-density_spectral)
        light = print_illu[None, None, :] * np.power(10.0, -density_spectral)
        raw_print = np.einsum("ijk,kc->ijc", light, print_sens)
        # Normalize so midgray=1 (mirrors what the pipeline does for
        # `normalize_print_exposure`).
        # Compute midgray normalization factor: same as `compute_exposure_factor`.
        # Mid gray = 0.184 ACES through film → density → integrate.
        # Easier: just dump raw_print value and accept the absolute scale
        # depends on normalization step, then compare to Rust.
        rp = raw_print[0, 0]

        # Final RGB (full pipeline)
        result = pipe.process(img)
        s_v = result[0, 0]
        print(f"\nACES [{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}]")
        print(
            f"  raw          [{raw_v[0]:.6f} {raw_v[1]:.6f} {raw_v[2]:.6f}]"
        )
        print(
            f"  log_raw      [{lr_v[0]:.6f} {lr_v[1]:.6f} {lr_v[2]:.6f}]"
        )
        print(f"  film dens    [{d_v[0]:.6f} {d_v[1]:.6f} {d_v[2]:.6f}]")
        print(f"  raw_print    [{rp[0]:.6f} {rp[1]:.6f} {rp[2]:.6f}]")
        print(f"  final RGB    [{s_v[0]:.6f} {s_v[1]:.6f} {s_v[2]:.6f}]")


if __name__ == "__main__":
    main()

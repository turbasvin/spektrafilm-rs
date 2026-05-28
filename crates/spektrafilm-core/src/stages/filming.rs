/// Filming stage: expose the digital image onto virtual film.
///
/// Full Hanatos2025 path: RGB → XYZ → xy chromaticity → tc coordinates →
/// 2D LUT lookup (spectra × sensitivity) → per-channel film raw exposure.
use rayon::prelude::*;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::colorspace;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{from_f32, from_f64, to_f32};
use spektrafilm_math::spectral::{self, TcLut};

use crate::params::RuntimeParams;
use crate::profile::Profile;

/// Compute pixel size in micrometers from film format and image dimensions.
pub fn pixel_size_um(film_format_mm: f32, width: u32, height: u32) -> f32 {
    film_format_mm * 1000.0 / width.max(height) as f32
}

/// Auto-exposure: center-weighted luminance metering. Parallel reduce over
/// rows — each row contributes an independent (weighted_sum, weight_total)
/// pair which the final sum combines.
/// Auto-exposure compensation, Python-compatible (`center_weighted` method
/// in `spektrafilm/utils/autoexposure.py`).
///
/// Python downsamples the image to ≤ 256 px on the long edge using
/// `skimage.transform.rescale(order=0)` (nearest neighbour) before
/// measuring, then applies a Gaussian (σ = 0.2 normalised) centred at
/// the frame middle. We mirror that — the downsample is what makes the
/// Gaussian-weighted mean Python-parity-compatible; measuring on the
/// full-res image gives a different mean (different pixel set,
/// different Y values dominating the centre) and shifts the resulting
/// EV by tenths of a stop.
pub fn measure_autoexposure_ev(image: &ImageBuf, rgb_to_xyz: &[[f32; 3]; 3]) -> f32 {
    const MAX_SIZE: usize = 256;
    let w = image.width as usize;
    let h = image.height as usize;
    let max_dim = w.max(h);
    // skimage.rescale(scale, order=0) with scale = MAX/max_dim:
    // output shape = round(orig * scale); each output pixel takes the
    // nearest source via iy = round((oy + 0.5) / scale - 0.5)
    // (scipy.ndimage.zoom convention).
    let (sw, sh, ix, iy) = if max_dim > MAX_SIZE {
        let scale = (MAX_SIZE as f64) / (max_dim as f64);
        let sw = ((w as f64) * scale).round() as usize;
        let sh = ((h as f64) * scale).round() as usize;
        let map = |out_dim: usize, src_dim: usize| -> Vec<usize> {
            (0..out_dim)
                .map(|o| {
                    let f = ((o as f64 + 0.5) / scale - 0.5).round() as isize;
                    f.clamp(0, src_dim as isize - 1) as usize
                })
                .collect()
        };
        (sw, sh, map(sw, w), map(sh, h))
    } else {
        let ix: Vec<usize> = (0..w).collect();
        let iy: Vec<usize> = (0..h).collect();
        (w, h, ix, iy)
    };

    let sigma = 0.2f32;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let max_sdim = sw.max(sh) as f32;

    let x_terms: Vec<f32> = (0..sw)
        .map(|x| {
            let nx = (x as f32 / sw as f32 - 0.5) * (sw as f32 / max_sdim);
            nx * nx
        })
        .collect();

    let (weighted_sum, weight_total) = (0..sh)
        .into_par_iter()
        .map(|y| {
            let ny = (y as f32 / sh as f32 - 0.5) * (sh as f32 / max_sdim);
            let ny2 = ny * ny;
            let row_off = iy[y] * w * 3;
            let mut row_sum = 0.0f64;
            let mut row_w = 0.0f64;
            for x in 0..sw {
                let weight = (-(x_terms[x] + ny2) * inv_2sigma2).exp();
                let idx = row_off + ix[x] * 3;
                let y_lum = rgb_to_xyz[1][0] * to_f32(image.data[idx])
                    + rgb_to_xyz[1][1] * to_f32(image.data[idx + 1])
                    + rgb_to_xyz[1][2] * to_f32(image.data[idx + 2]);
                row_sum += (y_lum as f64) * (weight as f64);
                row_w += weight as f64;
            }
            (row_sum, row_w)
        })
        .reduce(|| (0.0f64, 0.0f64), |a, b| (a.0 + b.0, a.1 + b.1));

    let exposure = (weighted_sum / weight_total) as f32 / 0.184;
    if exposure <= 0.0 || exposure.is_infinite() {
        return 0.0;
    }
    let ev = -exposure.log2();
    tracing::info!(
        sw = sw,
        sh = sh,
        weighted_mean = (weighted_sum / weight_total),
        exposure_div_184 = exposure,
        ev = ev,
        "autoexposure"
    );
    ev
}

/// Expose: convert RGB to film raw exposure via Hanatos2025 spectral upsampling.
///
/// If a TC LUT is provided, uses the full spectral path.
/// Otherwise falls back to simplified RGB → log10.
#[allow(clippy::too_many_arguments)]
pub fn expose(
    image: &ImageBuf,
    film: &Profile,
    params: &RuntimeParams,
    backend: &dyn ComputeBackend,
    tc_lut: Option<&TcLut>,
) -> ImageBuf {
    let pix_um = pixel_size_um(params.camera.film_format_mm, image.width, image.height);
    let rgb_to_xyz = input_colorspace_to_xyz(&params.io.input_color_space);

    // Auto-exposure
    let mut rgb = image.clone();
    if params.camera.auto_exposure {
        let ae_ev = measure_autoexposure_ev(&rgb, &rgb_to_xyz);
        let scale = from_f32(2.0f32.powf(ae_ev));
        rgb.data.par_iter_mut().for_each(|v| *v *= scale);
    }

    // Exposure compensation
    let exp_comp = from_f32(2.0f32.powf(params.camera.exposure_compensation_ev));
    rgb.data.par_iter_mut().for_each(|v| *v *= exp_comp);

    // RGB → film raw exposure
    let mut raw = if let Some(lut) = tc_lut {
        // Full Hanatos2025 spectral upsampling with CAT02 adaptation.
        let ref_illuminant = select_illuminant(&film.info.reference_illuminant);
        backend.hanatos2025_rgb_to_raw(&rgb, lut, &params.io.input_color_space, ref_illuminant)
    } else {
        // Simplified fallback: treat RGB values as proportional to raw exposure
        rgb.clone()
    };

    // Halation (on linear raw)
    let halation = &params.film_render.halation;
    if halation.active {
        raw = spektrafilm_model::diffusion::apply_halation_um(
            &raw,
            pix_um,
            halation.scatter_amount,
            halation.scatter_spatial_scale,
            halation.scatter_core_um,
            halation.scatter_tail_um,
            halation.scatter_tail_weight,
            halation.halation_amount,
            halation.halation_spatial_scale,
            halation.halation_strength,
            halation.halation_first_sigma_um,
            halation.halation_n_bounces,
            halation.halation_bounce_decay,
            halation.halation_renormalize,
            backend,
        );
    }

    // Lens blur
    if params.camera.lens_blur_um > 0.0 {
        raw = spektrafilm_model::diffusion::apply_gaussian_blur_um(
            &raw,
            params.camera.lens_blur_um,
            pix_um,
            backend,
        );
    }

    // Convert to log10 exposure. Mirror Python's
    // `np.log10(np.fmax(raw, 0.0) + 1e-10)` exactly — adding 1e-10
    // after the floor-at-zero shifts every value by a constant in
    // log-space and is *not* the same as `log10(max(raw, 1e-10))`.
    let zero = from_f64(0.0);
    let eps = from_f64(1e-10);
    raw.data.par_iter_mut().for_each(|v| {
        *v = ((*v).max(zero) + eps).log10();
    });

    raw
}

/// Develop: log_raw → density_cmy via density curves + DIR couplers + grain.
pub fn develop(
    log_raw: &ImageBuf,
    film: &Profile,
    params: &RuntimeParams,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    let pix_um = pixel_size_um(params.camera.film_format_mm, log_raw.width, log_raw.height);
    // f64 chain for Python parity — curves are f64 in the profile JSON.
    let log_exposure_f64 = film.log_exposure_f64();
    let density_curves_f64 = film.density_curves_f64();
    // f32 versions kept for DIR couplers (still f32 API) and grain (legacy).
    let log_exposure = &film.log_exposure_f32();
    let density_curves = &film.density_curves_f32();
    let norm_curves = spektrafilm_model::density_curves::normalize_density_curves(density_curves);
    let gamma = params.film_render.density_curve_gamma;

    // Filming.develop uses NORMALIZED curves (Python `develop` subtracts nanmin).
    let norm_curves_f64 =
        spektrafilm_model::density_curves::normalize_density_curves_f64(&density_curves_f64);
    let mut density_cmy =
        backend.density_curve_interp(log_raw, &log_exposure_f64, &norm_curves_f64, gamma as f64);

    // DIR couplers
    let dir = &params.film_render.dir_couplers;
    if dir.active {
        let matrix = spektrafilm_model::couplers::compute_dir_couplers_matrix(
            dir.gamma_samelayer_rgb,
            dir.gamma_interlayer_r_to_gb,
            dir.gamma_interlayer_g_to_rb,
            dir.gamma_interlayer_b_to_rg,
            dir.inhibition_samelayer,
            dir.inhibition_interlayer,
        );
        density_cmy = spektrafilm_model::couplers::apply_density_correction(
            &density_cmy,
            log_raw,
            pix_um,
            log_exposure,
            density_curves,
            &matrix,
            dir.amount,
            dir.diffusion_size_um,
            dir.diffusion_tail_um,
            dir.diffusion_tail_weight,
            film.is_positive(),
            gamma,
            backend,
        );
    }

    // Grain
    let grain = &params.film_render.grain;
    if grain.active {
        // Use f64 throughout — Python reads these from JSON as f64; the
        // f32 storage in `GrainParams` would otherwise truncate to ~7
        // decimals and shift every Poisson lambda by ~5e-8, producing a
        // visibly different grain pattern.
        let norm_curves_f64 = spektrafilm_model::density_curves::normalize_density_curves_f64(
            &film.density_curves_f64(),
        );
        let density_max = spektrafilm_model::density_curves::max_density_f64(&norm_curves_f64);
        density_cmy = spektrafilm_model::grain::apply_grain_to_density(
            &density_cmy,
            pix_um,
            grain.agx_particle_area_um2,
            grain.agx_particle_scale,
            grain.density_min,
            density_max,
            grain.uniformity,
            grain.blur,
            grain.n_sub_layers,
            backend,
        );
    }

    density_cmy
}

/// Full filming stage: expose + develop.
pub fn process(
    image: &ImageBuf,
    film: &Profile,
    params: &RuntimeParams,
    backend: &dyn ComputeBackend,
    tc_lut: Option<&TcLut>,
) -> ImageBuf {
    let log_raw = expose(image, film, params, backend, tc_lut);
    develop(&log_raw, film, params, backend)
}

pub fn input_colorspace_to_xyz(name: &str) -> [[f32; 3]; 3] {
    match name {
        "sRGB" => colorspace::SRGB_TO_XYZ,
        "ProPhoto RGB" => colorspace::PROPHOTO_TO_XYZ,
        "Rec. 2020" | "Rec2020" | "ITU-R BT.2020" => colorspace::REC2020_TO_XYZ,
        "ACES2065-1" => colorspace::ACES_TO_XYZ,
        _ => colorspace::PROPHOTO_TO_XYZ,
    }
}

fn select_illuminant(name: &str) -> &'static [f32] {
    match name {
        "D50" => &spectral::ILLUMINANT_D50,
        "D55" => &spectral::ILLUMINANT_D55,
        "D65" => &spectral::ILLUMINANT_D65,
        _ => &spectral::ILLUMINANT_D55,
    }
}

// Halation, optical diffusion, and blur effects.

use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{Scalar, ZERO, from_f32, from_f64};

/// Apply unsharp mask to an image. `backend` provides the Gaussian blur
/// implementation (CPU rayon or wgpu compute shader).
pub fn apply_unsharp_mask(
    image: &ImageBuf,
    sigma: f32,
    amount: f32,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    if sigma <= 0.0 || amount <= 0.0 {
        return image.clone();
    }
    let blurred = backend.gaussian_blur(image, sigma);
    let amount_s = from_f32(amount);
    let mut result = image.clone();
    for (r, (o, b)) in result
        .data
        .iter_mut()
        .zip(image.data.iter().zip(blurred.data.iter()))
    {
        *r = o + amount_s * (o - b);
    }
    result
}

/// Apply Gaussian blur in physical units (micrometers).
pub fn apply_gaussian_blur_um(
    image: &ImageBuf,
    sigma_um: f32,
    pixel_size_um: f32,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    let sigma_px = sigma_um / pixel_size_um;
    if sigma_px > 0.0 {
        backend.gaussian_blur(image, sigma_px)
    } else {
        image.clone()
    }
}

/// Apply in-emulsion scatter and back-reflection halation.
///
/// Port of Python `apply_halation_um`.
///
/// Ordering: scatter → halation.
/// Scatter: energy-preserving mixture of Gaussian core + exponential tail.
/// Halation: additive multi-bounce sum of Gaussians with sqrt(k)-spaced widths.
#[allow(clippy::too_many_arguments)]
pub fn apply_halation_um(
    raw: &ImageBuf,
    pixel_size_um: f32,
    scatter_amount: f64,
    scatter_spatial_scale: f64,
    scatter_core_um: [f64; 3],
    scatter_tail_um: [f64; 3],
    scatter_tail_weight: [f64; 3],
    halation_amount: f64,
    halation_spatial_scale: f64,
    halation_strength: [f64; 3],
    halation_first_sigma_um: [f64; 3],
    halation_n_bounces: u32,
    halation_bounce_decay: f64,
    halation_renormalize: bool,
    _backend: &dyn ComputeBackend,
) -> ImageBuf {
    // Per-channel implementation matching Python's `apply_halation_um`.
    // The `_backend` argument is retained for API compat with the GPU
    // resident path; on the CPU export path we operate channel-by-
    // channel using `gaussian_blur_channel` and
    // `exponential_filter_channel` so that:
    //   * scatter blur σ is per-channel (Python passes a length-3
    //     `sigma_c_px` array to `fast_gaussian_filter`).
    //   * scatter tail is the proper exponential filter (3-Gaussian
    //     mixture, matching Python `fast_exponential_filter`).
    //   * halation σ is per-channel.
    //   * halation strength is per-channel.
    //   * tail weight is per-channel.
    // i.e. every place Python takes a length-3 array, we treat it as a
    // length-3 array, not as `(sum / 3.0)`.
    use spektrafilm_math::gaussian::{exponential_filter_channel, gaussian_blur_channel};

    let w = raw.width;
    let h = raw.height;
    let n_pix = (w as usize) * (h as usize);
    let mut channels: [Vec<Scalar>; 3] = [
        raw.extract_channel(0),
        raw.extract_channel(1),
        raw.extract_channel(2),
    ];

    // 1. Scatter pass — per-channel core gaussian + tail exponential.
    // f64 sigmas/lambdas — Python casts these via
    // `np.asarray(..., dtype=np.float64) * s_scale / pixel_size_um`.
    let pix_um_f64 = pixel_size_um as f64;
    if scatter_amount > 0.0 {
        for c in 0..3 {
            let sigma_c_px_f64 = scatter_core_um[c] * scatter_spatial_scale / pix_um_f64;
            let lambda_t_px_f64 = scatter_tail_um[c] * scatter_spatial_scale / pix_um_f64;
            if sigma_c_px_f64 <= 0.0 && lambda_t_px_f64 <= 0.0 {
                continue;
            }
            // gaussian_blur/exponential_filter take f32 sigma — narrow at the
            // boundary (the kernel itself produces the same bits for any
            // f32-representable sigma, this is just so we don't break the API).
            let sigma_c_px = sigma_c_px_f64.max(1e-6) as f32;
            let lambda_t_px = lambda_t_px_f64.max(1e-6) as f32;
            let core = gaussian_blur_channel(&channels[c], w, h, sigma_c_px);
            let tail = exponential_filter_channel(&channels[c], w, h, lambda_t_px);
            let one = from_f64(1.0);
            let wt = from_f64(scatter_tail_weight[c]);
            let sa = from_f64(scatter_amount);
            for i in 0..n_pix {
                let scattered = (one - wt) * core[i] + wt * tail[i];
                channels[c][i] = (one - sa) * channels[c][i] + sa * scattered;
            }
        }
    }

    // 2. Halation pass — per-channel σ + per-channel strength.
    let a_tot: [f64; 3] = [
        halation_strength[0] * halation_amount,
        halation_strength[1] * halation_amount,
        halation_strength[2] * halation_amount,
    ];

    if halation_n_bounces >= 1 && (a_tot[0] > 0.0 || a_tot[1] > 0.0 || a_tot[2] > 0.0) {
        let n_bounces = halation_n_bounces as usize;
        // Decay computed in f64 to match Python's `rho ** (k - 1)` and
        // subsequent normalize-by-sum at full f64 precision.
        let mut decay = vec![0.0f64; n_bounces];
        for (k, slot) in decay.iter_mut().enumerate() {
            *slot = halation_bounce_decay.powi(k as i32);
        }
        let decay_sum: f64 = decay.iter().sum();
        for d in &mut decay {
            *d /= decay_sum;
        }

        for c in 0..3 {
            if a_tot[c] == 0.0 {
                continue;
            }
            let sigma_first_px_f64 =
                halation_first_sigma_um[c] * halation_spatial_scale / pix_um_f64;
            if sigma_first_px_f64 <= 0.0 {
                continue;
            }
            let mut hb = vec![ZERO; n_pix];
            for (k, &wk) in decay.iter().enumerate() {
                let sigma_k =
                    (sigma_first_px_f64 * ((k as f64) + 1.0).sqrt()).max(1e-6) as f32;
                let blurred = gaussian_blur_channel(&channels[c], w, h, sigma_k);
                let wk_s = from_f64(wk);
                for i in 0..n_pix {
                    hb[i] += wk_s * blurred[i];
                }
            }
            let a_c = from_f64(a_tot[c]);
            for i in 0..n_pix {
                channels[c][i] += a_c * hb[i];
            }
        }

        if halation_renormalize {
            let one = from_f64(1.0);
            for c in 0..3 {
                let denom = one + from_f64(a_tot[c]);
                for v in channels[c].iter_mut() {
                    *v /= denom;
                }
            }
        }
    }

    // Pack channels back into a single interleaved ImageBuf.
    let mut out = ImageBuf::new(w, h);
    for c in 0..3 {
        out.write_channel(c, &channels[c]);
    }
    out
}

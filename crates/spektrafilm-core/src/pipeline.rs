/// 3-stage simulation pipeline: Filming → Printing → Scanning.
///
/// Full calibration chain:
///   1. Load spectra LUT → compute TC LUT for film sensitivity
///   2. Process virtual gray card through filming to get midgray spectral density
///   3. Compute print exposure normalization factor from midgray spectral density
///   4. Pass all calibration data to the pipeline stages
use std::path::Path;

use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::spectral::TcLut;

/// Diagnostic helper: when env var `$var` is set, dump the image's f64
/// pixel data to that path as a raw little-endian f64 blob. Used to
/// bisect the Python ↔ Rust drift one stage at a time without
/// modifying call sites. Silent no-op when the env var is unset.
fn dump_if_env(var: &str, image: &ImageBuf) {
    let Ok(path) = std::env::var(var) else {
        return;
    };
    let bytes: Vec<u8> = image
        .data
        .iter()
        .flat_map(|&v| (v as f64).to_le_bytes())
        .collect();
    match std::fs::write(&path, &bytes) {
        Ok(()) => tracing::info!(
            path = %path,
            count = image.data.len(),
            "dumped f64 buffer for {}",
            var
        ),
        Err(e) => tracing::error!("dump {var} → {path}: {e}"),
    }
}

use crate::enlarger;
use crate::params::RuntimeParams;
use crate::profile::Profile;
use crate::spectral_service;
use crate::stages;

pub struct Pipeline {
    pub film: Profile,
    pub print: Profile,
    pub params: RuntimeParams,
    tc_lut: Option<TcLut>,
    /// Print exposure normalization factor (1/geomean of midgray raw through enlarger).
    print_exposure_factor: f64,
    /// Filtered enlarger illuminant for printing stage (f64 for Python parity).
    print_illuminant: Vec<f64>,
}

impl Pipeline {
    /// Accessor for the pre-computed TC LUT (used by parity tests).
    pub fn tc_lut(&self) -> Option<&TcLut> {
        self.tc_lut.as_ref()
    }
    /// Accessor for the print exposure factor (parity tests).
    pub fn print_exposure_factor(&self) -> f64 {
        self.print_exposure_factor
    }
    /// Accessor for the filtered print illuminant (parity tests).
    pub fn print_illuminant_slice(&self) -> &[f64] {
        &self.print_illuminant
    }
}

impl Pipeline {
    /// Create pipeline without spectral LUT (simplified path).
    pub fn new(film: Profile, print: Profile, params: RuntimeParams) -> Self {
        let print_illuminant = enlarger::enlarger_filtered_illuminant_f64(
            &params.enlarger.illuminant,
            params.enlarger.c_filter_neutral as f64,
            (params.enlarger.m_filter_neutral + params.enlarger.m_filter_shift) as f64,
            (params.enlarger.y_filter_neutral + params.enlarger.y_filter_shift) as f64,
        );
        Self {
            film,
            print,
            params,
            tc_lut: None,
            print_exposure_factor: 1.0,
            print_illuminant,
        }
    }

    /// Create pipeline with full Hanatos2025 spectral upsampling and calibration.
    pub fn new_with_spectral(
        film: Profile,
        print: Profile,
        mut params: RuntimeParams,
        data_dir: &Path,
    ) -> Result<Self, String> {
        // Python parity: mirror `_apply_film_specifics` in
        // `spektrafilm/runtime/params_builder.py`. Python applies these
        // overrides inside `digest_params()` before the pipeline runs,
        // so a fresh `RuntimeParams::default()` does NOT match what
        // Python uses. The DIR-coupler gammas in particular differ
        // between positive and negative films.
        if film.is_positive() {
            params.film_render.dir_couplers.gamma_samelayer_rgb = [0.12, 0.08, 0.06];
            params.film_render.dir_couplers.gamma_interlayer_r_to_gb = [0.12, 0.06];
            params.film_render.dir_couplers.gamma_interlayer_g_to_rb = [0.08, 0.06];
            params.film_render.dir_couplers.gamma_interlayer_b_to_rg = [0.06, 0.06];
        } else if film.is_negative() {
            params.film_render.dir_couplers.gamma_samelayer_rgb = [0.336, 0.319, 0.273];
            params.film_render.dir_couplers.gamma_interlayer_r_to_gb = [0.353, 0.302];
            params.film_render.dir_couplers.gamma_interlayer_g_to_rb = [0.154, 0.353];
            params.film_render.dir_couplers.gamma_interlayer_b_to_rg = [0.168, 0.226];
        }

        // Python parity: look up per-(print, illuminant, film) neutral filter values from
        // the JSON database — matches `apply_database_neutral_print_filters`. Defaults to
        // params.enlarger.{c,m,y}_filter_neutral when the combo isn't in the database.
        // Keep the f64 lookup values around (params is f32) — narrowing to
        // f32 here costs ~4e-8 precision through the `10^(-cc/100)` step.
        let mut neutral_cmy_f64: Option<[f64; 3]> = None;
        if params.settings.neutral_print_filters_from_database {
            let db = crate::neutral_filters::NeutralFilters::load(data_dir);
            let print_stock = print.info.stock.as_deref().unwrap_or("");
            let film_stock = film.info.stock.as_deref().unwrap_or("");
            if let Some([c, m, y]) = db.lookup(print_stock, &params.enlarger.illuminant, film_stock)
            {
                params.enlarger.c_filter_neutral = c as f32;
                params.enlarger.m_filter_neutral = m as f32;
                params.enlarger.y_filter_neutral = y as f32;
                neutral_cmy_f64 = Some([c, m, y]);
            }
        }
        let cmy_f64 = neutral_cmy_f64.unwrap_or([
            params.enlarger.c_filter_neutral as f64,
            params.enlarger.m_filter_neutral as f64,
            params.enlarger.y_filter_neutral as f64,
        ]);
        let (c_neutral_f64, m_neutral_f64, y_neutral_f64) = (cmy_f64[0], cmy_f64[1], cmy_f64[2]);

        let spectra_lut = spectral_service::load_spectra_lut(data_dir)?;

        // Film sensitivity: Python `sensitivity = np.nan_to_num(10 ** log_sensitivity)` — f64.
        let log_sens = film.log_sensitivity_f64();
        let sensitivity: Vec<[f64; 3]> = log_sens
            .iter()
            .map(|row| {
                let mut out = [0.0f64; 3];
                for c in 0..3 {
                    let v = 10.0f64.powf(row[c]);
                    out[c] = if v.is_nan() { 0.0 } else { v };
                }
                out
            })
            .collect();

        // Compute TC LUT (optionally with bandpass window)
        let ref_illuminant = select_illuminant(&film.info.reference_illuminant);
        let ref_illuminant_f64 = select_illuminant_f64(&film.info.reference_illuminant);
        let window_params: Vec<f64> = film.data.hanatos2025_adaptation_window_params.clone();
        let tc_lut =
            if params.settings.apply_hanatos2025_adaptation_window && window_params.len() >= 4 {
                spectral_service::compute_tc_lut_with_window(
                    &spectra_lut,
                    &sensitivity,
                    &window_params,
                    ref_illuminant_f64,
                )
            } else {
                spectral_service::compute_tc_lut(&spectra_lut, &sensitivity)
            };

        // Python parity: sensitivities are pre-balanced in the profile so midgray ≈ 1.0.
        // The TC LUT is used unnormalized — Python's `rgb_to_raw_hanatos2025` does NOT scale it.
        // See `spektrafilm/utils/spectral_upsampling.py:rgb_to_raw_hanatos2025` (comment line 373).

        // Compute enlarger illuminant with dichroic filters — f64 for Python parity.
        // c/m/y come from the f64 lookup, shift values are f32 in params.
        let print_illuminant = enlarger::enlarger_filtered_illuminant_f64(
            &params.enlarger.illuminant,
            c_neutral_f64,
            m_neutral_f64 + params.enlarger.m_filter_shift as f64,
            y_neutral_f64 + params.enlarger.y_filter_shift as f64,
        );

        // Compute midgray spectral density (gray card through full filming path)
        let density_spectral_midgray =
            enlarger::compute_midgray_spectral_density(&tc_lut, &film, &params, ref_illuminant);

        // Print sensitivity: Python `sensitivity = np.nan_to_num(10 ** log_sensitivity)` — f64 with NaN→0.
        let print_log_sens = print.log_sensitivity_f64();
        let print_sensitivity: Vec<[f64; 3]> = print_log_sens
            .iter()
            .map(|row| {
                let mut out = [0.0f64; 3];
                for c in 0..3 {
                    let v = 10.0f64.powf(row[c]);
                    out[c] = if v.is_nan() { 0.0 } else { v };
                }
                out
            })
            .collect();

        // Print exposure normalization — mirror Python's
        // `_compute_exposure_factor_midgray` in
        // `spektrafilm/runtime/stages/printing.py`. There are two
        // candidate factors:
        //   * `factor_midgray`     = 1 / geomean(raw_midgray)
        //   * `factor_midgray_comp`= 1 / geomean(raw_midgray_with_neg_EV)
        // and four flag combinations of
        // `enlarger.normalize_print_exposure` × `enlarger.print_exposure_compensation`.
        //
        // With both flags ON (defaults) and no EV compensation,
        // Python returns `factor_midgray_comp == 1.0`. Rust used to
        // unconditionally apply `factor_midgray` here, which biased
        // the print exposure by ~3% on every render and was the root
        // cause of the residual Python-parity drift in the print stage.
        let factor_midgray = enlarger::compute_exposure_factor(
            &density_spectral_midgray,
            &print_illuminant,
            &print_sensitivity,
        );
        // Python builds `density_spectral_midgray_comp` whenever
        // `print_exposure_compensation` is on — even when EV == 0, in
        // which case `rgb_midgray_comp = rgb_midgray * 2^0` and so
        // `factor_midgray_comp == factor_midgray`. We have to mirror
        // that (NOT short-circuit to 1.0) because the
        // `factor_midgray_comp` branch is what gets returned by default.
        let factor_midgray_comp = if !params.enlarger.print_exposure_compensation {
            1.0
        } else if params.camera.exposure_compensation_ev == 0.0 {
            factor_midgray
        } else {
            let density_spectral_midgray_comp = enlarger::compute_midgray_spectral_density_comp(
                &tc_lut,
                &film,
                &params,
                ref_illuminant,
                params.camera.exposure_compensation_ev,
            );
            enlarger::compute_exposure_factor(
                &density_spectral_midgray_comp,
                &print_illuminant,
                &print_sensitivity,
            )
        };
        let print_exposure_factor = match (
            params.enlarger.normalize_print_exposure,
            params.enlarger.print_exposure_compensation,
        ) {
            (true, true) => factor_midgray_comp,
            (true, false) => factor_midgray,
            (false, true) => factor_midgray_comp / factor_midgray,
            (false, false) => 1.0,
        };

        tracing::info!(
            lut_size = tc_lut.size,
            film = film.info.stock.as_deref().unwrap_or("unknown"),
            print_exposure_factor = print_exposure_factor,
            "pipeline calibrated"
        );

        Ok(Self {
            film,
            print,
            params,
            tc_lut: Some(tc_lut),
            print_exposure_factor,
            print_illuminant,
        })
    }

    pub fn process(&self, image: ImageBuf, backend: &dyn ComputeBackend) -> ImageBuf {
        tracing::info!(backend = backend.name(), "pipeline: start");

        // GPU fast path: when none of the CPU-only effects are active and we
        // have a TC LUT, dispatch the whole filming→printing→scanning chain as
        // a single GPU command buffer (one upload + one readback total).
        // Halation and DIR couplers are supported in-resident; grain/glare/
        // etc. still force the per-stage path.
        if !self.params.io.scan_film
            && self.tc_lut.is_some()
            && self.params.camera.lens_blur_um == 0.0
            && !self.params.scanner.white_correction
            && !self.params.scanner.black_correction
            && self.params.scanner.lens_blur == 0.0
        {
            if let Some(out) = self.try_gpu_resident(&image, backend) {
                tracing::info!("pipeline: gpu-resident fast path complete");
                return self.apply_post_scan(out);
            }
        }

        // Parity bisection probe: split filming into expose+develop so we
        // can dump the intermediate log_raw (post-Hanatos + log10) buffer
        // when `$SPEKTRAFILM_DUMP_FILM_LOG_RAW` is set. This is the same
        // split Python's pipeline exposes via `output_film_log_raw`.
        let log_raw = stages::filming::expose(
            &image,
            &self.film,
            &self.params,
            backend,
            self.tc_lut.as_ref(),
        );
        dump_if_env("SPEKTRAFILM_DUMP_FILM_LOG_RAW", &log_raw);
        let filmed = stages::filming::develop(
            &log_raw,
            &self.film,
            &self.params,
            backend,
        );
        tracing::info!("pipeline: filming complete");
        dump_if_env("SPEKTRAFILM_DUMP_FILM_DENSITY", &filmed);

        if self.params.io.scan_film {
            let result = stages::scanning::process(&filmed, &self.film, &self.params, backend);
            tracing::info!("pipeline: scanning complete (film scan)");
            result
        } else {
            let printed = stages::printing::process_with_calibration(
                &filmed,
                &self.film,
                &self.print,
                &self.params,
                backend,
                &self.print_illuminant,
                self.print_exposure_factor,
            );
            tracing::info!("pipeline: printing complete");
            dump_if_env("SPEKTRAFILM_DUMP_PRINT_DENSITY", &printed);
            let result = stages::scanning::process(&printed, &self.print, &self.params, backend);
            tracing::info!("pipeline: scanning complete");
            result
        }
    }

    /// Try the GPU-resident fast path. Builds all the per-stage data and
    /// hands it to the backend's `try_run_film_chain`. The output is linear RGB
    /// (clipped to [0,1]) — sRGB encoding is applied by `apply_post_scan`.
    fn try_gpu_resident(&self, image: &ImageBuf, backend: &dyn ComputeBackend) -> Option<ImageBuf> {
        let tc_lut = self.tc_lut.as_ref()?;
        let ref_illuminant = select_illuminant(&self.film.info.reference_illuminant);
        let mut rgb_to_adapted = spektrafilm_math::spectral::build_rgb_to_adapted_xyz(
            &self.params.io.input_color_space,
            ref_illuminant,
        );

        // Bake the exposure scale (auto-exposure × manual EV compensation)
        // into rgb_to_adapted_xyz. Hanatos applies the matrix per pixel, so
        // scaling the matrix is equivalent to scaling the input RGB —
        // saves a separate "scale" compute pass at the head of the chain.
        // Auto-exposure metering itself stays on CPU (~30 ms at 6 MP after
        // the per-row rayon parallelization); the result is a single float
        // that's cheap to roll into the matrix.
        let mut exposure_scale_f64 = 1.0f64;
        if self.params.camera.auto_exposure {
            // Borrow back the f32 matrix shape that the CPU metering
            // function expects.
            let rgb_to_xyz_f32 =
                stages::filming::input_colorspace_to_xyz(&self.params.io.input_color_space);
            let ae_ev = stages::filming::measure_autoexposure_ev(image, &rgb_to_xyz_f32);
            exposure_scale_f64 *= 2.0f64.powf(ae_ev as f64);
        }
        if self.params.camera.exposure_compensation_ev != 0.0 {
            exposure_scale_f64 *=
                2.0f64.powf(self.params.camera.exposure_compensation_ev as f64);
        }
        if (exposure_scale_f64 - 1.0).abs() > 1e-9 {
            for row in &mut rgb_to_adapted {
                for v in row.iter_mut() {
                    *v *= exposure_scale_f64;
                }
            }
        }

        // Film density curves: normalized (filming.develop subtracts nanmin).
        let film_log_exp = self.film.log_exposure_f64();
        let film_curves = self.film.density_curves_f64();
        let film_curves_norm =
            spektrafilm_model::density_curves::normalize_density_curves_f64(&film_curves);
        let film_channel_density: Vec<[f64; 3]> = self
            .film
            .data
            .channel_density
            .iter()
            .map(|r| {
                [
                    r.first().copied().unwrap_or(0.0),
                    r.get(1).copied().unwrap_or(0.0),
                    r.get(2).copied().unwrap_or(0.0),
                ]
            })
            .collect();
        let film_base_density = self.film.data.base_density.clone();

        // Print sensitivity (10**log_sensitivity, NaN→0).
        let print_sens: Vec<[f64; 3]> = self
            .print
            .log_sensitivity_f64()
            .iter()
            .map(|row| {
                let mut o = [0.0; 3];
                for c in 0..3 {
                    let v = 10f64.powf(row[c]);
                    o[c] = if v.is_nan() { 0.0 } else { v };
                }
                o
            })
            .collect();
        let print_log_exp = self.print.log_exposure_f64();
        let print_curves = self.print.density_curves_f64();
        let print_channel_density: Vec<[f64; 3]> = self
            .print
            .data
            .channel_density
            .iter()
            .map(|r| {
                [
                    r.first().copied().unwrap_or(0.0),
                    r.get(1).copied().unwrap_or(0.0),
                    r.get(2).copied().unwrap_or(0.0),
                ]
            })
            .collect();
        let print_base_density = self.print.data.base_density.clone();

        // Scanning: viewing illuminant + normalization + combined XYZ→RGB matrix.
        let viewing_illu_f32 = match self.print.info.viewing_illuminant.as_str() {
            "D50" => &spektrafilm_math::spectral::ILLUMINANT_D50[..],
            "D55" => &spektrafilm_math::spectral::ILLUMINANT_D55[..],
            "D65" => &spektrafilm_math::spectral::ILLUMINANT_D65[..],
            _ => &spektrafilm_math::spectral::ILLUMINANT_D50[..],
        };
        let viewing_illu: Vec<f64> = viewing_illu_f32.iter().map(|&v| v as f64).collect();
        let n_wl = print_channel_density.len();
        let scan_norm: f64 = (0..n_wl)
            .map(|i| viewing_illu[i] * spektrafilm_math::spectral::CMF_Y[i] as f64)
            .sum();
        let viewing_white = spektrafilm_math::spectral::illuminant_xyz_f64(viewing_illu_f32);
        let output_white = spektrafilm_math::spectral::colorspace_white_xyz_f64(
            &self.params.io.output_color_space,
        );
        let adapt = spektrafilm_math::colorspace::chromatic_adaptation_matrix_f64(
            viewing_white,
            output_white,
        );
        let base_xyz_to_rgb = match self.params.io.output_color_space.as_str() {
            "sRGB" => spektrafilm_math::colorspace::XYZ_TO_SRGB_F64,
            "ProPhoto RGB" => spektrafilm_math::colorspace::XYZ_TO_PROPHOTO_F64,
            "Rec. 2020" | "Rec2020" | "ITU-R BT.2020" => {
                spektrafilm_math::colorspace::XYZ_TO_REC2020_F64
            }
            "ACES2065-1" => spektrafilm_math::colorspace::XYZ_TO_ACES_F64,
            _ => spektrafilm_math::colorspace::XYZ_TO_SRGB_F64,
        };
        let mut scan_xyz_to_rgb = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                scan_xyz_to_rgb[i][j] = base_xyz_to_rgb[i][0] * adapt[0][j]
                    + base_xyz_to_rgb[i][1] * adapt[1][j]
                    + base_xyz_to_rgb[i][2] * adapt[2][j];
            }
        }

        let print_norm_factor =
            self.print_exposure_factor * self.params.enlarger.print_exposure as f64;

        // Halation in the resident chain — only built when the halation
        // stage is active. Mirrors `apply_halation_um`: averages the
        // per-channel µm sigmas, converts to pixel space, and passes the
        // resulting scalars to the shaders.
        let halation = if self.params.film_render.halation.active {
            let pix_um = stages::filming::pixel_size_um(
                self.params.camera.film_format_mm,
                image.width,
                image.height,
            );
            let h = &self.params.film_render.halation;
            // GPU shaders take f32 — narrow at the boundary.
            let avg_f64 = |a: [f64; 3]| (a[0] + a[1] + a[2]) / 3.0;
            let strength_avg = (avg_f64(h.halation_strength) * h.halation_amount) as f32;
            let a_tot = [
                (h.halation_strength[0] * h.halation_amount) as f32,
                (h.halation_strength[1] * h.halation_amount) as f32,
                (h.halation_strength[2] * h.halation_amount) as f32,
            ];
            Some(spektrafilm_gpu::HalationGpuParams {
                scatter_amount: h.scatter_amount as f32,
                scatter_core_px: (avg_f64(h.scatter_core_um) * h.scatter_spatial_scale
                    / pix_um as f64) as f32,
                scatter_tail_px: (avg_f64(h.scatter_tail_um) * h.scatter_spatial_scale
                    / pix_um as f64) as f32,
                scatter_tail_weight: [
                    h.scatter_tail_weight[0] as f32,
                    h.scatter_tail_weight[1] as f32,
                    h.scatter_tail_weight[2] as f32,
                ],
                halation_amount: h.halation_amount as f32,
                halation_strength_avg: strength_avg,
                halation_a_tot: a_tot,
                halation_first_sigma_px: (avg_f64(h.halation_first_sigma_um)
                    * h.halation_spatial_scale
                    / pix_um as f64) as f32,
                halation_n_bounces: h.halation_n_bounces,
                halation_bounce_decay: h.halation_bounce_decay as f32,
                halation_renormalize: h.halation_renormalize,
            })
        } else {
            None
        };

        // DIR couplers in the resident chain. Mirrors CPU
        // `apply_density_correction`: build the scaled couplers matrix and
        // pre-compute the "density curves before DIR" once. The shader
        // re-interpolates these against `log_raw - correction`.
        // Held in this binding so `&density_curves_0_f64` outlives the
        // backend call.
        let dir_inputs = if self.params.film_render.dir_couplers.active {
            let pix_um = stages::filming::pixel_size_um(
                self.params.camera.film_format_mm,
                image.width,
                image.height,
            );
            let dir = &self.params.film_render.dir_couplers;
            let matrix = spektrafilm_model::couplers::compute_dir_couplers_matrix(
                dir.gamma_samelayer_rgb,
                dir.gamma_interlayer_r_to_gb,
                dir.gamma_interlayer_g_to_rb,
                dir.gamma_interlayer_b_to_rg,
                dir.inhibition_samelayer,
                dir.inhibition_interlayer,
            );
            let mut matrix_scaled = matrix;
            for row in &mut matrix_scaled {
                for v in row.iter_mut() {
                    *v *= dir.amount;
                }
            }
            let film_curves_f32 = self.film.density_curves_f32();
            let film_log_exp_f32 = self.film.log_exposure_f32();
            let norm_curves_f32 =
                spektrafilm_model::density_curves::normalize_density_curves(&film_curves_f32);
            // Narrow matrix to f32 just for the compute_curves_before_dir helper
            // (which still operates on f32 density curves from the profile).
            let matrix_scaled_f32: [[f32; 3]; 3] = [
                [matrix_scaled[0][0] as f32, matrix_scaled[0][1] as f32, matrix_scaled[0][2] as f32],
                [matrix_scaled[1][0] as f32, matrix_scaled[1][1] as f32, matrix_scaled[1][2] as f32],
                [matrix_scaled[2][0] as f32, matrix_scaled[2][1] as f32, matrix_scaled[2][2] as f32],
            ];
            let density_curves_0_f32 = spektrafilm_model::couplers::compute_curves_before_dir(
                &norm_curves_f32,
                &film_log_exp_f32,
                &matrix_scaled_f32,
                self.film.is_positive(),
            );
            let density_curves_0_f64: Vec<[f64; 3]> = density_curves_0_f32
                .iter()
                .map(|row| [row[0] as f64, row[1] as f64, row[2] as f64])
                .collect();
            let density_max_f32 =
                spektrafilm_model::density_curves::max_density(&norm_curves_f32);
            Some((
                density_curves_0_f64,
                matrix_scaled,
                density_max_f32,
                pix_um,
                dir.diffusion_size_um,
                dir.diffusion_tail_um,
                dir.diffusion_tail_weight,
                self.film.is_positive(),
                self.params.film_render.density_curve_gamma as f64,
            ))
        } else {
            None
        };
        let dir_couplers = dir_inputs.as_ref().map(|d| {
            // GPU shader path is f32 — narrow at the boundary.
            let matrix_f32: [[f32; 3]; 3] = [
                [d.1[0][0] as f32, d.1[0][1] as f32, d.1[0][2] as f32],
                [d.1[1][0] as f32, d.1[1][1] as f32, d.1[1][2] as f32],
                [d.1[2][0] as f32, d.1[2][1] as f32, d.1[2][2] as f32],
            ];
            spektrafilm_gpu::DirCouplersGpuParams {
                couplers_matrix_scaled: matrix_f32,
                density_max: d.2,
                is_positive: d.7,
                diffusion_size_px: (d.4 / d.3 as f64) as f32,
                diffusion_tail_px: (d.5 / d.3 as f64) as f32,
                diffusion_tail_weight: d.6 as f32,
                density_curves_0: &d.0,
                log_exposure: &film_log_exp,
                gamma_factor: d.8,
            }
        });

        // Grain in the resident chain — same per-channel particle math as
        // `apply_grain_to_density`. The GPU uses normal-approximation
        // sampling; CPU does the same whenever λ > 30 / var > 9, which is
        // the typical regime for ≥ 1 MP images.
        let grain = if self.params.film_render.grain.active {
            let pix_um = stages::filming::pixel_size_um(
                self.params.camera.film_format_mm,
                image.width,
                image.height,
            );
            let g = &self.params.film_render.grain;
            let pixel_area = pix_um * pix_um;
            let n_sub = g.n_sub_layers.max(1);
            // GPU shaders are f32 — narrow the f64 grain params at the boundary.
            let mut npp = [0.0f32; 3];
            for c in 0..3 {
                let particle_area = g.agx_particle_area_um2 * g.agx_particle_scale[c];
                npp[c] = ((pixel_area as f64 / particle_area) / n_sub as f64) as f32;
            }
            let film_curves_f32 = self.film.density_curves_f32();
            let norm_curves_f32 =
                spektrafilm_model::density_curves::normalize_density_curves(&film_curves_f32);
            let dmax_curves =
                spektrafilm_model::density_curves::max_density(&norm_curves_f32);
            let mut density_max = [0.0f32; 3];
            for c in 0..3 {
                density_max[c] = dmax_curves[c] + g.density_min[c] as f32;
            }
            Some(spektrafilm_gpu::GrainGpuParams {
                density_min: [
                    g.density_min[0] as f32,
                    g.density_min[1] as f32,
                    g.density_min[2] as f32,
                ],
                density_max,
                n_particles_per_pixel: npp,
                grain_uniformity: [
                    g.uniformity[0] as f32,
                    g.uniformity[1] as f32,
                    g.uniformity[2] as f32,
                ],
                n_sub_layers: n_sub,
                base_seed: 0,
                grain_blur: g.blur,
            })
        } else {
            None
        };

        // Glare in the resident chain — applied after scan_spectral on the
        // final RGB buffer. Mirrors the CPU lognormal + blur + add. The
        // CPU code reads `print_render.glare` for the print scan path,
        // not `film_render.glare`.
        let glare = if self.params.print_render.glare.active
            && self.params.print_render.glare.percent > 0.0
        {
            let g = &self.params.print_render.glare;
            // LogNormal parameters (same derivation as `compute_random_glare_amount`).
            let m = g.percent as f64;
            let s = (g.roughness * g.percent) as f64;
            let sigma2 = (1.0 + (s * s) / (m * m)).ln();
            let sigma = sigma2.sqrt();
            let mu = m.ln() - sigma2 / 2.0;
            // glare_rgb_offset = (XYZ→RGB) · illuminant_xyz / 100.
            let mut illu_xyz = [0.0f64; 3];
            let n_wl_loc = print_channel_density.len();
            for i in 0..n_wl_loc {
                illu_xyz[0] += viewing_illu[i] * spektrafilm_math::spectral::CMF_X[i] as f64;
                illu_xyz[1] += viewing_illu[i] * spektrafilm_math::spectral::CMF_Y[i] as f64;
                illu_xyz[2] += viewing_illu[i] * spektrafilm_math::spectral::CMF_Z[i] as f64;
            }
            for c in 0..3 {
                illu_xyz[c] /= scan_norm;
            }
            let mut offset_rgb = [0.0f32; 3];
            for i in 0..3 {
                let v = scan_xyz_to_rgb[i][0] * illu_xyz[0]
                    + scan_xyz_to_rgb[i][1] * illu_xyz[1]
                    + scan_xyz_to_rgb[i][2] * illu_xyz[2];
                offset_rgb[i] = (v / 100.0) as f32;
            }
            Some(spektrafilm_gpu::GlareGpuParams {
                mu: mu as f32,
                sigma: sigma as f32,
                blur_px: g.blur,
                base_seed: 42,
                rgb_offset: offset_rgb,
            })
        } else {
            None
        };

        // Unsharp mask: scanner.unsharp_mask = [sigma, amount].
        let [usm_sigma, usm_amount] = self.params.scanner.unsharp_mask;
        let unsharp = if usm_sigma > 0.0 && usm_amount > 0.0 {
            Some(spektrafilm_gpu::UnsharpGpuParams {
                sigma_px: usm_sigma,
                amount: usm_amount,
            })
        } else {
            None
        };

        let params = spektrafilm_gpu::FilmChainParams {
            image,
            tc_lut,
            rgb_to_adapted_xyz: &rgb_to_adapted,
            film_log_exposure: &film_log_exp,
            film_density_curves_normalized: &film_curves_norm,
            film_gamma: self.params.film_render.density_curve_gamma as f64,
            film_channel_density: &film_channel_density,
            film_base_density: &film_base_density,
            print_illuminant: &self.print_illuminant,
            print_sensitivity: &print_sens,
            print_normalization_factor: print_norm_factor,
            print_log_exposure: &print_log_exp,
            print_density_curves: &print_curves,
            print_gamma: self.params.print_render.density_curve_gamma as f64,
            print_channel_density: &print_channel_density,
            print_base_density: &print_base_density,
            viewing_illuminant: &viewing_illu,
            scan_normalization: scan_norm,
            scan_xyz_to_rgb: &scan_xyz_to_rgb,
            halation,
            dir_couplers,
            grain,
            glare,
            unsharp,
        };
        backend.try_run_film_chain(&params)
    }

    /// After the GPU fast path returns linear RGB, apply sRGB encoding + clip
    /// (the only post-scan op when no scanner blur/unsharp/correction/glare is active).
    fn apply_post_scan(&self, mut rgb: ImageBuf) -> ImageBuf {
        use rayon::prelude::*;
        use spektrafilm_math::precision::from_f64;
        let zero = from_f64(0.0);
        let one = from_f64(1.0);
        if self.params.io.output_cctf_encoding {
            rgb.data.par_iter_mut().for_each(|v| {
                *v = spektrafilm_math::precision::srgb_encode((*v).clamp(zero, one));
            });
        } else {
            rgb.data
                .par_iter_mut()
                .for_each(|v| *v = (*v).clamp(zero, one));
        }
        rgb
    }
}

fn select_illuminant(name: &str) -> &'static [f32] {
    use spektrafilm_math::spectral;
    match name {
        "D50" => &spectral::ILLUMINANT_D50,
        "D55" => &spectral::ILLUMINANT_D55,
        "D65" => &spectral::ILLUMINANT_D65,
        _ => &spectral::ILLUMINANT_D55,
    }
}

fn select_illuminant_f64(name: &str) -> &'static [f64] {
    use spektrafilm_math::spectral;
    match name {
        "D50" => &spectral::ILLUMINANT_D50_F64,
        "D55" => &spectral::ILLUMINANT_D55_F64,
        "D65" => &spectral::ILLUMINANT_D65_F64,
        _ => &spectral::ILLUMINANT_D55_F64,
    }
}

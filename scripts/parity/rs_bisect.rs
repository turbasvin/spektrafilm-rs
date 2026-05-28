// Stage-by-stage dump for parity bisection. Loads the same linear-ACES
// TIFF, runs the bare chain stage by stage, and writes each stage's
// f64 output to disk for diffing against Python.

use std::path::Path;
use std::fs::File;
use std::io::Write;

fn dump_f64(name: &str, data: &[f64]) {
    let path = format!("/tmp/cmp/stage_{name}.f64");
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|v| v.to_le_bytes().to_vec())
        .collect();
    File::create(&path).unwrap().write_all(&bytes).unwrap();
    eprintln!("wrote {path}  ({} doubles)", data.len());
}

fn imagebuf_to_f64(img: &spektrafilm_math::image::ImageBuf) -> Vec<f64> {
    img.data.iter().map(|&v| spektrafilm_math::precision::to_f32(v) as f64).collect()
}

fn main() {
    use spektrafilm_core::params::RuntimeParams;
    use spektrafilm_core::profile;
    use spektrafilm_core::spectral_service;
    use spektrafilm_core::stages;
    use spektrafilm_core::neutral_filters::NeutralFilters;
    use spektrafilm_core::enlarger;
    use spektrafilm_gpu::cpu_backend::CpuBackend;
    use spektrafilm_math::image::ImageBuf;
    use spektrafilm_math::precision::{Scalar, from_f32};
    use spektrafilm_math::spectral::build_rgb_to_adapted_xyz;

    let data_dir = Path::new("/Users/sasha/Desktop/spektrafilm-rs/data");
    let film = profile::load_profile_by_name(data_dir, "kodak_portra_400").expect("film");
    let print = profile::load_profile_by_name(data_dir, "kodak_portra_endura").expect("print");

    // Load input TIFF as f32 → Scalar
    let img = image::open("/tmp/cmp/input_linear_aces.tif").expect("tiff").to_rgb32f();
    let (w, h) = (img.width(), img.height());
    let scalars: Vec<Scalar> = img.into_raw().into_iter().map(from_f32).collect();
    let input = ImageBuf::from_data(w, h, scalars);
    dump_f64("00_input", &imagebuf_to_f64(&input));

    // Params: bare chain
    let mut params = RuntimeParams::default();
    params.camera.auto_exposure = false;
    params.film_render.halation.active = false;
    params.film_render.dir_couplers.active = false;
    params.film_render.grain.active = false;
    params.print_render.glare.active = false;
    params.scanner.unsharp_mask = [0.0, 0.0];
    params.io.input_color_space = "ACES2065-1".into();
    params.io.input_cctf_decoding = false;

    // Database lookup for c/m/y
    if params.settings.neutral_print_filters_from_database {
        let db = NeutralFilters::load(data_dir);
        if let Some([c, m, y]) = db.lookup(
            print.info.stock.as_deref().unwrap_or(""),
            &params.enlarger.illuminant,
            film.info.stock.as_deref().unwrap_or(""),
        ) {
            params.enlarger.c_filter_neutral = c as f32;
            params.enlarger.m_filter_neutral = m as f32;
            params.enlarger.y_filter_neutral = y as f32;
        }
    }

    // Build TC LUT (with window, like the pipeline does)
    let log_sens = film.log_sensitivity_f64();
    let sensitivity: Vec<[f64; 3]> = log_sens
        .iter()
        .map(|row| {
            let mut o = [0.0f64; 3];
            for c in 0..3 {
                let v = 10.0f64.powf(row[c]);
                o[c] = if v.is_nan() { 0.0 } else { v };
            }
            o
        })
        .collect();
    let spectra_lut = spectral_service::load_spectra_lut(data_dir).expect("LUT");
    let window_params: Vec<f64> = film.data.hanatos2025_adaptation_window_params.clone();
    let ref_illu: &[f32] = match film.info.reference_illuminant.as_str() {
        "D50" => &spektrafilm_math::spectral::ILLUMINANT_D50,
        "D65" => &spektrafilm_math::spectral::ILLUMINANT_D65,
        _ => &spektrafilm_math::spectral::ILLUMINANT_D55,
    };
    let tc_lut = if params.settings.apply_hanatos2025_adaptation_window && window_params.len() >= 4 {
        spectral_service::compute_tc_lut_with_window(&spectra_lut, &sensitivity, &window_params, ref_illu)
    } else {
        spectral_service::compute_tc_lut(&spectra_lut, &sensitivity)
    };

    // Print exposure factor branching (replicate pipeline.rs logic)
    let density_spectral_midgray = enlarger::compute_midgray_spectral_density(&tc_lut, &film, &params, ref_illu);
    let print_log_sens = print.log_sensitivity_f64();
    let print_sensitivity: Vec<[f64; 3]> = print_log_sens
        .iter()
        .map(|row| {
            let mut o = [0.0f64; 3];
            for c in 0..3 { let v = 10.0f64.powf(row[c]); o[c] = if v.is_nan() { 0.0 } else { v }; }
            o
        })
        .collect();
    let print_illuminant = enlarger::enlarger_filtered_illuminant_f64(
        &params.enlarger.illuminant,
        params.enlarger.c_filter_neutral,
        params.enlarger.m_filter_neutral + params.enlarger.m_filter_shift,
        params.enlarger.y_filter_neutral + params.enlarger.y_filter_shift,
    );
    let factor_midgray = enlarger::compute_exposure_factor(&density_spectral_midgray, &print_illuminant, &print_sensitivity);
    let print_exposure_factor = factor_midgray;
    eprintln!("print_exposure_factor = {print_exposure_factor:.17}");

    let backend = CpuBackend;

    // ── 1. Filming ─────────────────────────────────────────────────
    let log_raw = stages::filming::expose(&input, &film, &params, &backend, Some(&tc_lut));
    dump_f64("01_log_raw", &imagebuf_to_f64(&log_raw));

    // ── 2. Film develop (density curve interp) ─────────────────────
    let density_cmy_film = stages::filming::develop(&log_raw, &film, &params, &backend);
    dump_f64("02_density_cmy_film", &imagebuf_to_f64(&density_cmy_film));

    // ── 3. Printing (process_with_calibration) ─────────────────────
    let printed = stages::printing::process_with_calibration(
        &density_cmy_film, &film, &print, &params, &backend,
        &print_illuminant, print_exposure_factor,
    );
    dump_f64("03_density_cmy_print", &imagebuf_to_f64(&printed));

    // ── 4. Scanning ────────────────────────────────────────────────
    let result = stages::scanning::process(&printed, &print, &params, &backend);
    dump_f64("04_scan_rgb", &imagebuf_to_f64(&result));
}

mod debug_compare;
pub mod filming;
pub mod printing;
pub mod scanning;

#[cfg(test)]
mod integration_tests {
    use crate::params::RuntimeParams;
    use crate::pipeline::Pipeline;
    use crate::profile;
    use spektrafilm_math::image::ImageBuf;
    use spektrafilm_math::precision::{Scalar, from_f64};
    use std::path::Path;

    fn data_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("data")
    }

    #[test]
    fn test_full_pipeline_portra_400_to_endura() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();

        let mut params = RuntimeParams::default();
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let img = ImageBuf::from_data(8, 8, vec![from_f64(0.184); 8 * 8 * 3]);

        let pipeline = Pipeline::new(film, print, params);
        let result = pipeline.process(img, &backend);

        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        let px = result.get(4, 4);
        for c in 0..3 {
            assert!(
                px[c] >= from_f64(0.0) && px[c] <= from_f64(1.0),
                "channel {c} out of range: {}",
                px[c]
            );
        }
        let mean: Scalar =
            result.data.iter().copied().sum::<Scalar>() / result.data.len() as Scalar;
        assert!(mean > from_f64(0.01), "output near-black: mean={mean}");
        assert!(mean < from_f64(0.99), "output near-white: mean={mean}");
    }

    #[test]
    fn test_full_pipeline_with_spectral_lut() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();

        let mut params = RuntimeParams::default();
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;
        params.io.input_color_space = "sRGB".to_string();

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let img = ImageBuf::from_data(8, 8, vec![from_f64(0.184); 8 * 8 * 3]);

        let pipeline = Pipeline::new_with_spectral(film, print, params, &dir);
        match pipeline {
            Ok(p) => {
                let result = p.process(img, &backend);
                let mean: Scalar =
                    result.data.iter().copied().sum::<Scalar>() / result.data.len() as Scalar;
                eprintln!("Spectral pipeline output mean: {mean}");
                let px = result.get(4, 4);
                eprintln!("Spectral pipeline pixel(4,4): {:?}", px);
                assert!(
                    mean > from_f64(0.01),
                    "spectral output near-black: mean={mean}"
                );
                assert!(
                    mean < from_f64(0.99),
                    "spectral output near-white: mean={mean}"
                );
            }
            Err(e) => {
                eprintln!("Spectral LUT not available: {e} — skipping test");
            }
        }
    }

    #[test]
    fn test_film_scan_pipeline() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = film.clone();

        let mut params = RuntimeParams::default();
        params.io.scan_film = true;
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let img = ImageBuf::from_data(4, 4, vec![from_f64(0.184); 4 * 4 * 3]);

        let pipeline = Pipeline::new(film, print, params);
        let result = pipeline.process(img, &backend);
        let mean: Scalar =
            result.data.iter().copied().sum::<Scalar>() / result.data.len() as Scalar;
        assert!(mean > from_f64(0.01), "film scan near-black: mean={mean}");
        assert!(mean < from_f64(0.99), "film scan near-white: mean={mean}");
    }
}

#[cfg(test)]
mod debug_tests {
    use crate::params::RuntimeParams;
    use crate::pipeline::Pipeline;
    use crate::profile;
    use crate::stages;
    use spektrafilm_math::image::ImageBuf;
    use spektrafilm_math::precision::from_f64;
    use std::path::Path;

    #[test]
    fn debug_pipeline_values() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("data");
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();

        let mut params = RuntimeParams::default();
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;
        params.io.input_color_space = "sRGB".to_string();

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let gray = from_f64(0.184);
        let img = ImageBuf::from_data(1, 1, vec![gray, gray, gray]);
        eprintln!("Input: {:?}", img.get(0, 0));

        let log_raw = stages::filming::expose(&img, &film, &params, &backend, None);
        eprintln!("log_raw: {:?}", log_raw.get(0, 0));

        let density_cmy = stages::filming::develop(&log_raw, &film, &params, &backend);
        eprintln!("density_cmy: {:?}", density_cmy.get(0, 0));

        // Use simplified printing path for debug trace
        let printed = stages::printing::process(&density_cmy, &film, &print, &params, &backend);
        eprintln!("density_print: {:?}", printed.get(0, 0));
        let density_print = printed;
        let rgb_out = stages::scanning::scan(&density_print, &print, &params, &backend);
        eprintln!("rgb_out: {:?}", rgb_out.get(0, 0));
    }
}

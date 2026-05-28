// Minimal interactive preview for spektrafilm.
//
// Loads an image, exposes the most impactful runtime parameters as
// egui controls, and re-renders the full GPU-resident pipeline whenever
// a control changes. The output texture is uploaded to egui once per
// render; the panel scales it to fit.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use eframe::egui;
use rayon::prelude::*;
use spektrafilm_core::params::RuntimeParams;
use spektrafilm_core::pipeline::Pipeline;
use spektrafilm_core::profile;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{Scalar, from_f32, srgb_decode, to_f32};

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let backend = spektrafilm_gpu::select_backend();
    let backend: Arc<dyn ComputeBackend> = Arc::from(backend);
    let initial_image = std::env::args().nth(1).map(PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
        // Force the wgpu renderer (Metal-backed CAMetalLayer on macOS)
        // instead of glow. We need a CAMetalLayer so we can tag its
        // colorspace as sRGB — see `tag_metal_layer_srgb` below.
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "spektrafilm",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc, backend, initial_image)))),
    )
}

/// One entry in the film / paper combo box. `stock` is the filename
/// stem (the unique key the profile loader expects); `display` is the
/// human-readable label from the profile's `info.name`, falling back
/// to the stock id when missing.
#[derive(Debug, Clone)]
struct ProfileEntry {
    stock: String,
    display: String,
}

struct App {
    backend: Arc<dyn ComputeBackend>,
    data_dir: PathBuf,
    /// Profiles where `info.support == "film"`.
    films: Vec<ProfileEntry>,
    /// Profiles where `info.support == "paper"` (or other print-stage
    /// supports).
    papers: Vec<ProfileEntry>,
    film_name: String,
    print_name: String,
    params: RuntimeParams,
    image_path: Option<PathBuf>,
    image: Option<ImageBuf>,
    /// Last rendered pipeline output (post sRGB encode + clip). Retained
    /// so the Save button can write it without re-running the pipeline.
    output_image: Option<ImageBuf>,
    output_tex: Option<egui::TextureHandle>,
    last_render_ms: f32,
    last_pipeline_build_ms: f32,
    status: String,
    /// Set when any control change should trigger a re-render. The
    /// next `update()` tick dispatches the render onto a worker
    /// thread so the UI stays responsive while the pipeline (up to
    /// ~500 ms on 6 MP) runs in the background.
    dirty: bool,
    /// Marked when the user changes a param while a previous render
    /// is still in flight. After that render finishes we re-arm
    /// `dirty` so the latest state gets a fresh pass instead of
    /// dropping the user's mid-render edits on the floor.
    pending_dirty: bool,
    /// Current preview zoom multiplier on top of the fit-to-panel
    /// scale (1.0 = fit). Mouse wheel adjusts it around the cursor,
    /// drag pans, double-click resets.
    zoom: f32,
    /// Pan offset (screen pixels) added to the preview centre.
    pan: egui::Vec2,
    /// macOS-only: tag the wgpu CAMetalLayer's `colorspace` as sRGB on
    /// the first `update()` tick (it's not yet wired up at
    /// `App::new` time). Set to `true` once the call succeeds so we
    /// don't retry every frame.
    #[cfg(target_os = "macos")]
    metal_colorspace_tagged: bool,
    /// In-flight pipeline render. `Some` while the worker thread is
    /// running; main thread polls the receiver each frame and
    /// uploads the resulting texture once it lands. Decoupling the
    /// render from the UI thread is what keeps sliders responsive —
    /// the 250–500 ms pipeline used to block input handling.
    render_job: Option<RenderJob>,
    /// In-flight f64 export job. `Some` while the subprocess is
    /// running; the `update()` loop polls the receiver each frame and
    /// surfaces success/failure in `status` when the worker thread
    /// completes. Joined eagerly to release the thread.
    export_job: Option<ExportJob>,
}

/// One in-flight preview render. The worker owns a Pipeline + the
/// ImageBuf clone and, when it finishes, sends back the output buffer
/// plus the two timings the status bar shows.
struct RenderJob {
    rx: mpsc::Receiver<RenderResult>,
    handle: Option<JoinHandle<()>>,
}

struct RenderResult {
    output: ImageBuf,
    pipeline_build_ms: f32,
    render_ms: f32,
}

/// One in-flight f64 export. The worker thread owns the child
/// process and polls `cancel` in its wait loop. On completion the
/// worker sends `Ok(elapsed_seconds, output_filename)` or `Err(msg)`;
/// `Err("cancelled")` is also produced when the cancel flag is set.
/// The join handle is held so we can `join()` after consuming the
/// message and on `on_exit` to drain the worker before the process dies.
struct ExportJob {
    rx: mpsc::Receiver<Result<(f32, String), String>>,
    handle: Option<JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
    started_at: Instant,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        backend: Arc<dyn ComputeBackend>,
        initial_image: Option<PathBuf>,
    ) -> Self {
        let _ = cc;

        let data_dir = pick_data_dir();
        let (films, papers) = scan_profiles(&data_dir);
        let film_name = pick_default_stock(&films, "kodak_gold_200");
        // Prefer the film's own `target_print` for the initial paper —
        // each profile is tuned against a specific paper, so guessing
        // a generic default produces a colour-shifted preview on
        // start-up (Kodak Gold's normal pair is Portra Endura, not
        // Fujifilm Crystal Archive).
        let print_name = profile::load_profile_by_name(&data_dir, &film_name)
            .ok()
            .and_then(|f| f.info.target_print.clone())
            .filter(|t| papers.iter().any(|p| &p.stock == t))
            .unwrap_or_else(|| pick_default_stock(&papers, "fujifilm_crystal_archive_typeii"));

        let mut app = Self {
            backend,
            data_dir,
            films,
            papers,
            film_name,
            print_name,
            params: RuntimeParams::default(),
            image_path: None,
            image: None,
            output_image: None,
            output_tex: None,
            last_render_ms: 0.0,
            last_pipeline_build_ms: 0.0,
            pending_dirty: false,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            render_job: None,
            status: String::from("Load an image to start."),
            dirty: false,
            #[cfg(target_os = "macos")]
            metal_colorspace_tagged: false,
            export_job: None,
        };
        if let Some(p) = initial_image {
            app.load_image_from_path(&p);
        }
        app
    }

    fn load_image_from_path(&mut self, path: &Path) {
        let t = Instant::now();
        match load_image(path) {
            Ok(img) => {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                // PNG (post sRGB-decode) and RAW (linear sRGB after
                // disabling imagepipe's gamma+basecurve) both deliver
                // linear sRGB. Tell the pipeline accordingly so it
                // doesn't double-decode the gamma.
                if ext == "png" || is_raw_extension(&ext) {
                    self.params.io.input_color_space = "sRGB".to_string();
                    self.params.io.input_cctf_decoding = false;
                }
                self.status = format!(
                    "Loaded {} × {} ({:.1} MP) in {:.0} ms",
                    img.width,
                    img.height,
                    (img.pixel_count() as f64 / 1e6),
                    t.elapsed().as_secs_f32() * 1000.0
                );
                self.image = Some(img);
                self.image_path = Some(path.to_path_buf());
                self.dirty = true;
            }
            Err(e) => {
                self.status = format!("Load error: {e:#}");
            }
        }
    }

    /// Spawn a render on a worker thread. The UI stays interactive
    /// during the 250–500 ms pipeline run (previously this blocked
    /// the main thread, dropping mid-drag slider events). If a job
    /// is already in flight, mark `pending_dirty` so the latest
    /// params get re-rendered as soon as the in-flight one returns.
    fn dispatch_render(&mut self, ctx: &egui::Context) {
        if self.render_job.is_some() {
            self.pending_dirty = true;
            return;
        }
        let Some(image) = self.image.clone() else { return; };
        let film_name = self.film_name.clone();
        let print_name = self.print_name.clone();
        let params = self.params.clone();
        let data_dir = self.data_dir.clone();
        let backend = self.backend.clone();
        let (tx, rx) = mpsc::channel();
        let ctx_for_worker = ctx.clone();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-render".into())
            .spawn(move || {
                let t_build = Instant::now();
                let film = match profile::load_profile_by_name(&data_dir, &film_name) {
                    Ok(f) => f,
                    Err(_) => return,
                };
                let print = match profile::load_profile_by_name(&data_dir, &print_name) {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let pipeline = match Pipeline::new_with_spectral(film, print, params, &data_dir) {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let pipeline_build_ms = t_build.elapsed().as_secs_f32() * 1000.0;
                let t = Instant::now();
                let output = pipeline.process(image, backend.as_ref());
                let render_ms = t.elapsed().as_secs_f32() * 1000.0;
                let _ = tx.send(RenderResult {
                    output,
                    pipeline_build_ms,
                    render_ms,
                });
                ctx_for_worker.request_repaint();
            })
            .expect("OS thread spawn");
        self.render_job = Some(RenderJob {
            rx,
            handle: Some(handle),
        });
    }

    /// Central-panel preview with mouse-wheel zoom (centred on the
    /// cursor) and click-drag pan. Double-click resets to fit.
    /// `self.zoom = 1.0` means fit-to-panel; values >1 zoom in.
    fn draw_preview_with_zoom(&mut self, ui: &mut egui::Ui, tex: &egui::TextureHandle) {
        let avail = ui.available_size();
        let img_size = tex.size_vec2();
        let fit_scale = (avail.x / img_size.x).min(avail.y / img_size.y).min(1.0);
        let scale = fit_scale * self.zoom;
        let drawn_size = img_size * scale;

        let response = ui.allocate_response(avail, egui::Sense::click_and_drag());
        let rect = response.rect;

        // Two zoom sources, never combined:
        //   1. trackpad pinch (`zoom_delta` — already a multiplier
        //      around 1.0, smoothed by egui per frame)
        //   2. mouse wheel scroll (`smooth_scroll_delta.y` — egui's
        //      per-frame integrated scroll). Don't add `raw_scroll_delta`
        //      on top; `smooth` already accumulates the raw events.
        // Both are anchored on the cursor so the pixel under the
        // pointer stays put.
        let (pinch, scroll) = ui.input(|i| {
            if response.hovered() {
                (i.zoom_delta(), i.smooth_scroll_delta.y)
            } else {
                (1.0, 0.0)
            }
        });
        // Scroll → small log-zoom step per notch. 0.0015 keeps a single
        // trackpad swipe feeling like a smooth zoom rather than a leap.
        let scroll_factor = if scroll.abs() > 0.01 {
            (scroll * 0.0015).exp()
        } else {
            1.0
        };
        let factor = pinch * scroll_factor;
        if (factor - 1.0).abs() > 1e-4 {
            let cursor = ui.input(|i| i.pointer.hover_pos());
            let old_zoom = self.zoom;
            self.zoom = (self.zoom * factor).clamp(0.1, 32.0);
            if let Some(cur) = cursor {
                // Keep the image point under the cursor anchored.
                let centre = rect.center() + self.pan;
                let offset_from_centre = cur - centre;
                let scale_ratio = self.zoom / old_zoom;
                self.pan += offset_from_centre * (1.0 - scale_ratio);
            }
        }

        if response.dragged() {
            self.pan += response.drag_delta();
        }
        if response.double_clicked() {
            self.zoom = 1.0;
            self.pan = egui::Vec2::ZERO;
        }

        let centre = rect.center() + self.pan;
        let draw_rect = egui::Rect::from_center_size(centre, drawn_size);
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        // Clip to the central panel so panned-off pixels don't leak.
        let painter = ui.painter_at(rect);
        painter.image(tex.id(), draw_rect, uv, egui::Color32::WHITE);

        // Zoom indicator + reset chip in the corner.
        let pct = (self.zoom * fit_scale * 100.0).round() as i32;
        let text = if (self.zoom - 1.0).abs() < 0.001 && self.pan.length_sq() < 1.0 {
            "fit".to_string()
        } else {
            format!("{pct} %  (double-click to reset)")
        };
        let pos = rect.left_top() + egui::vec2(8.0, 8.0);
        painter.text(
            pos,
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::monospace(12.0),
            egui::Color32::from_rgba_premultiplied(220, 220, 220, 200),
        );
    }

    /// Called once per `update()`. If the in-flight render finished,
    /// upload the texture and unblock the next pass. If more changes
    /// arrived during the render, re-arm `dirty`.
    fn poll_render_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.render_job.as_mut() else { return; };
        let result = match job.rx.try_recv() {
            Ok(r) => r,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.render_job = None;
                self.status = "Render error: worker thread vanished".into();
                return;
            }
        };
        if let Some(h) = job.handle.take() {
            let _ = h.join();
        }
        self.render_job = None;
        self.last_pipeline_build_ms = result.pipeline_build_ms;
        self.last_render_ms = result.render_ms;
        self.output_tex = Some(make_texture(ctx, &result.output));
        self.output_image = Some(result.output);
        if self.pending_dirty {
            self.pending_dirty = false;
            self.dirty = true;
        }
    }

    /// Open a save dialog and write the most recent rendered output to
    /// disk. Suggested filename is the input stem + the chosen film
    /// stock + the chosen extension; default extension is PNG (8-bit
    /// sRGB-encoded, matching what's on screen).
    fn save_dialog(&mut self) {
        let Some(out) = self.output_image.clone() else {
            self.status = "Nothing to save yet — load an image first.".into();
            return;
        };
        let default_name = match &self.image_path {
            Some(p) => {
                let stem = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("spektrafilm");
                format!("{stem}_{}_spektra.png", self.film_name)
            }
            None => "spektrafilm.png".into(),
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "tif", "tiff"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };
        let t = Instant::now();
        match save_image(&out, &path) {
            Ok(()) => {
                self.status = format!(
                    "Saved {} in {:.0} ms",
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("(file)"),
                    t.elapsed().as_secs_f32() * 1000.0
                );
            }
            Err(e) => {
                self.status = format!("Save error: {e:#}");
            }
        }
    }

    /// Export the current frame by subprocessing the f64-built
    /// `spektrafilm` CLI. The GUI runs the pipeline at f32 on the GPU
    /// for fast iteration; export re-runs the pipeline at f64 (CPU)
    /// using the same params and writes a standard PNG/TIFF/JPEG.
    ///
    /// The f64 CLI is located via, in order: `$SPEKTRAFILM_F64_CLI`,
    /// then `spektrafilm-f64` on `PATH`, then `target/release/spektrafilm-f64`
    /// relative to `CARGO_MANIFEST_DIR`. Build it with
    /// `cargo build --release --features precision-f64 -p spektrafilm-cli`
    /// and either rename / symlink it to `spektrafilm-f64` or set the env var.
    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(input_path) = self.image_path.clone() else {
            self.status = "Load an image before exporting.".into();
            return;
        };
        let cli_path = match locate_f64_cli() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("Export: {e}");
                return;
            }
        };
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("spektrafilm");
        let default_name = format!("{stem}_{}_spektra_f64.png", self.film_name);
        let Some(out_path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "tif", "tiff", "jpg", "jpeg"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };

        // Spawn the export on a worker thread so the egui event loop
        // keeps drawing. The cancel flag is shared with the worker so
        // the Cancel button (and `on_exit`) can kill the child cleanly
        // instead of letting it orphan after the GUI window closes.
        //
        // Enlarger + scanner LUTs are toggled on for the export only.
        // The GUI preview runs on wgpu (fast at full spectral
        // integration; LUT round-trip via CPU PCHIP would only slow
        // it down), but the f64 CPU export gets a 5-10× speedup from
        // the LUT path — matching Python's typical export config.
        let film = self.film_name.clone();
        let paper = self.print_name.clone();
        let mut params = self.params.clone();
        params.settings.use_enlarger_lut = true;
        params.settings.use_scanner_lut = true;
        let data_dir = self.data_dir.clone();
        let (tx, rx) = mpsc::channel();
        let ctx_for_worker = ctx.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_for_worker = Arc::clone(&cancel);
        let started_at = Instant::now();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-export".into())
            .spawn(move || {
                let res = run_f64_export(
                    &cli_path,
                    &input_path,
                    &out_path,
                    &film,
                    &paper,
                    &params,
                    &data_dir,
                    &cancel_for_worker,
                );
                let name = out_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("(file)")
                    .to_string();
                let msg = match res {
                    Ok(()) => Ok((started_at.elapsed().as_secs_f32(), name)),
                    Err(e) => Err(format!("{e:#}")),
                };
                let _ = tx.send(msg);
                ctx_for_worker.request_repaint();
            })
            .expect("OS thread spawn");
        self.status = "Exporting at f64 (CPU)…".into();
        self.export_job = Some(ExportJob {
            rx,
            handle: Some(handle),
            cancel,
            started_at,
        });
    }

    /// Request cancellation of the in-flight export. The worker thread
    /// observes the flag in its `try_wait` poll loop and SIGKILLs the
    /// child; `poll_export_job` then drains the resulting error message
    /// on the next `update()` tick.
    fn cancel_export(&mut self) {
        let Some(job) = self.export_job.as_ref() else {
            return;
        };
        if !job.cancel.swap(true, Ordering::SeqCst) {
            self.status = "Cancelling export…".into();
        }
    }

    /// Called once per `update()`. If the export is still in-flight,
    /// refreshes the status with elapsed time and requests a repaint a
    /// second from now (so the timer ticks without us busy-looping).
    /// If the job finished, drains the result into the status bar and
    /// joins the worker thread.
    fn poll_export_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.export_job.as_mut() else {
            return;
        };
        let msg = match job.rx.try_recv() {
            Ok(m) => m,
            Err(mpsc::TryRecvError::Empty) => {
                let secs = job.started_at.elapsed().as_secs();
                self.status = if job.cancel.load(Ordering::SeqCst) {
                    format!("Cancelling export… ({secs}s)")
                } else {
                    format!("Exporting at f64 (CPU)… ({secs}s)")
                };
                ctx.request_repaint_after(Duration::from_secs(1));
                return;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.status = "Export error: worker thread vanished".into();
                self.export_job = None;
                return;
            }
        };
        if let Some(h) = job.handle.take() {
            let _ = h.join();
        }
        self.export_job = None;
        self.status = match msg {
            Ok((secs, name)) => format!("Exported (f64 CPU) {name} in {secs:.1} s"),
            Err(e) if e.contains("cancelled") => "Export cancelled.".into(),
            Err(e) => format!("Export error: {e}"),
        };
    }

    fn controls_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("spektrafilm");
        ui.add_space(6.0);

        // ── File ────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            if ui.button("Open…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(
                        "Image",
                        &[
                            "png", "tif", "tiff", // standard
                            "dng", "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "raf",
                            "orf", "rw2", "pef", "srw", "x3f", "iiq", "3fr", "crw", "rwl",
                            "mrw", "mef", "kdc",
                        ],
                    )
                    .pick_file()
                {
                    self.load_image_from_path(&path);
                }
            }
            let save_enabled = self.output_image.is_some();
            if ui
                .add_enabled(save_enabled, egui::Button::new("Save…"))
                .on_disabled_hover_text("Render an image first")
                .clicked()
            {
                self.save_dialog();
            }
            let export_busy = self.export_job.is_some();
            if export_busy {
                if ui
                    .button("Cancel")
                    .on_hover_text("Stop the in-flight f64 CPU export and kill the child process.")
                    .clicked()
                {
                    self.cancel_export();
                }
            } else {
                let export_enabled = self.image_path.is_some();
                if ui
                    .add_enabled(export_enabled, egui::Button::new("Export…"))
                    .on_hover_text(
                        "Re-run the pipeline at f64 precision (CPU, via the spektrafilm-f64 \
                         CLI subprocess) and write a PNG/TIFF/JPEG. The live preview stays \
                         at f32 GPU for speed; export trades time for precision.",
                    )
                    .on_disabled_hover_text("Load an image first")
                    .clicked()
                {
                    self.export_dialog(ctx);
                }
            }
        });
        if let Some(p) = &self.image_path {
            ui.label(
                egui::RichText::new(p.file_name().and_then(|s| s.to_str()).unwrap_or(""))
                    .small(),
            );
        }
        ui.add_space(4.0);

        // ── Profiles ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Profiles")
            .default_open(true)
            .show(ui, |ui| {
                let film_changed = profile_combo(
                    ui,
                    "film",
                    "Film stock",
                    &self.films,
                    &mut self.film_name,
                );
                if film_changed {
                    // Follow the film's `target_print` so a fresh film
                    // pick lands on the paper the profile was tuned for.
                    if let Ok(film) =
                        profile::load_profile_by_name(&self.data_dir, &self.film_name)
                    {
                        if let Some(target) = film.info.target_print.as_deref() {
                            if self.papers.iter().any(|p| p.stock == target) {
                                self.print_name = target.to_string();
                            }
                        }
                    }
                    self.dirty = true;
                }
                if profile_combo(
                    ui,
                    "paper",
                    "Print paper",
                    &self.papers,
                    &mut self.print_name,
                ) {
                    self.dirty = true;
                }
            });

        // ── Exposure ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Exposure")
            .default_open(true)
            .show(ui, |ui| {
                let mut changed = false;
                changed |= ui
                    .checkbox(&mut self.params.camera.auto_exposure, "Auto exposure")
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut self.params.camera.exposure_compensation_ev,
                            -5.0..=5.0,
                        )
                        .text("EV compensation")
                        .step_by(0.1),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.params.camera.film_format_mm, 4.0..=120.0)
                            .text("Film format (mm)")
                            .logarithmic(true),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.params.camera.lens_blur_um, 0.0..=100.0)
                            .text("Lens blur (µm)"),
                    )
                    .changed();
                if changed {
                    self.dirty = true;
                }
            });

        // ── Halation ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Halation")
            .default_open(true)
            .show(ui, |ui| {
                let h = &mut self.params.film_render.halation;
                let mut changed = false;
                changed |= ui.checkbox(&mut h.active, "Active").changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut h.halation_amount, 0.0..=3.0)
                            .text("Halation amount"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut h.halation_spatial_scale, 0.1..=5.0)
                            .text("Halation scale"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut h.scatter_amount, 0.0..=3.0).text("Scatter amount"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut h.scatter_spatial_scale, 0.1..=5.0)
                            .text("Scatter scale"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut h.halation_n_bounces, 1..=5).text("Bounces"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut h.halation_bounce_decay, 0.0..=1.0)
                            .text("Bounce decay"),
                    )
                    .changed();
                changed |= ui.checkbox(&mut h.halation_renormalize, "Renormalize").changed();
                if changed {
                    self.dirty = true;
                }
            });

        // ── DIR couplers ────────────────────────────────────────────────
        egui::CollapsingHeader::new("DIR couplers")
            .default_open(true)
            .show(ui, |ui| {
                let d = &mut self.params.film_render.dir_couplers;
                let mut changed = false;
                changed |= ui.checkbox(&mut d.active, "Active").changed();
                changed |= ui
                    .add(egui::Slider::new(&mut d.amount, 0.0..=2.0).text("Amount"))
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut d.diffusion_size_um, 0.0..=100.0)
                            .text("Diffusion size (µm)"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut d.diffusion_tail_um, 0.0..=400.0)
                            .text("Diffusion tail (µm)"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut d.diffusion_tail_weight, 0.0..=1.0)
                            .text("Tail weight"),
                    )
                    .changed();
                if changed {
                    self.dirty = true;
                }
            });

        // ── Grain ───────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Grain")
            .default_open(true)
            .show(ui, |ui| {
                let g = &mut self.params.film_render.grain;
                let mut changed = false;
                changed |= ui.checkbox(&mut g.active, "Active").changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut g.agx_particle_area_um2, 0.05..=1.0)
                            .text("Particle area (µm²)"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut g.blur, 0.0..=3.0).text("Post-blur σ"),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut g.n_sub_layers, 1..=4).text("Sub-layers"))
                    .changed();
                if changed {
                    self.dirty = true;
                }
            });

        // ── Glare ───────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Glare")
            .default_open(false)
            .show(ui, |ui| {
                let g = &mut self.params.print_render.glare;
                let mut changed = false;
                changed |= ui.checkbox(&mut g.active, "Active").changed();
                changed |= ui
                    .add(egui::Slider::new(&mut g.percent, 0.0..=0.2).text("Percent"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut g.roughness, 0.0..=2.0).text("Roughness"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut g.blur, 0.0..=5.0).text("Blur σ (px)"))
                    .changed();
                if changed {
                    self.dirty = true;
                }
            });

        // ── Scanner ─────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Scanner")
            .default_open(false)
            .show(ui, |ui| {
                let s = &mut self.params.scanner;
                let mut changed = false;
                changed |= ui
                    .add(
                        egui::Slider::new(&mut s.lens_blur, 0.0..=5.0).text("Lens blur σ (px)"),
                    )
                    .changed();
                let [mut sigma, mut amount] = s.unsharp_mask;
                changed |= ui
                    .add(egui::Slider::new(&mut sigma, 0.0..=3.0).text("Unsharp σ (px)"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut amount, 0.0..=2.0).text("Unsharp amount"))
                    .changed();
                if changed {
                    s.unsharp_mask = [sigma, amount];
                }
                changed |= ui
                    .checkbox(&mut s.white_correction, "White correction")
                    .changed();
                if s.white_correction {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut s.white_level, 0.5..=1.0)
                                .text("White level"),
                        )
                        .changed();
                }
                changed |= ui
                    .checkbox(&mut s.black_correction, "Black correction")
                    .changed();
                if s.black_correction {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut s.black_level, 0.0..=0.5)
                                .text("Black level"),
                        )
                        .changed();
                }
                if changed {
                    self.dirty = true;
                }
            });

        // ── Enlarger ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Enlarger")
            .default_open(false)
            .show(ui, |ui| {
                let e = &mut self.params.enlarger;
                let mut changed = false;
                changed |= ui
                    .add(
                        egui::Slider::new(&mut e.print_exposure, 0.1..=5.0)
                            .text("Print exposure")
                            .logarithmic(true),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut e.m_filter_shift, -50.0..=50.0)
                            .text("Magenta filter shift"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut e.y_filter_shift, -50.0..=50.0)
                            .text("Yellow filter shift"),
                    )
                    .changed();
                if changed {
                    self.dirty = true;
                }
            });

        // ── Output ──────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Output")
            .default_open(false)
            .show(ui, |ui| {
                let io = &mut self.params.io;
                let mut changed = false;
                let mut scan_film = io.scan_film;
                changed |= ui.checkbox(&mut scan_film, "Scan film (skip printing)").changed();
                if changed {
                    io.scan_film = scan_film;
                }
                changed |= ui
                    .checkbox(&mut io.output_cctf_encoding, "Output sRGB encoded")
                    .changed();
                if changed {
                    self.dirty = true;
                }
            });

        // ── Metrics ─────────────────────────────────────────────────────
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);
        ui.monospace(format!(
            "render:        {:>6.1} ms   {}",
            self.last_render_ms,
            fps_label(self.last_render_ms)
        ));
        ui.monospace(format!(
            "pipeline build:{:>6.1} ms",
            self.last_pipeline_build_ms
        ));
        ui.monospace(format!("backend: {}", self.backend.name()));
        ui.label(egui::RichText::new(&self.status).small());
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // First-paint hook: tag the wgpu Metal layer's colorspace as sRGB.
        // Has to happen here (not in `App::new`) because the CAMetalLayer
        // isn't wired up at construction time.
        #[cfg(target_os = "macos")]
        if !self.metal_colorspace_tagged {
            match tag_metal_layer_srgb(frame) {
                Ok(()) => {
                    eprintln!("[spektrafilm] CAMetalLayer.colorspace = sRGB — tagged OK");
                    self.metal_colorspace_tagged = true;
                }
                Err(e) => {
                    eprintln!("[spektrafilm] colorspace tag attempt: {e}");
                }
            }
        }
        let _ = frame;

        if self.dirty && self.image.is_some() {
            self.dispatch_render(ctx);
            self.dirty = false;
        }
        self.poll_render_job(ctx);
        self.poll_export_job(ctx);
        egui::SidePanel::right("controls")
            .resizable(false)
            .exact_width(340.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls_panel(ui, ctx));
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            // Clone the texture handle (cheap — internally `Arc`) so we
            // can take `&mut self` to mutate `zoom` / `pan` from the
            // closure without a borrow conflict on `self.output_tex`.
            let tex = self.output_tex.clone();
            if let Some(tex) = tex {
                self.draw_preview_with_zoom(ui, &tex);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Drop or open an image to start.").size(18.0));
                });
            }
        });

        // Accept drag-and-dropped image files.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if let Some(path) = dropped.into_iter().next() {
            self.load_image_from_path(&path);
        }
    }

    /// Called once when the window is closing. If an export is still
    /// in-flight, set the cancel flag and join the worker thread so
    /// the child process is SIGKILLed before the GUI exits. Without
    /// this the f64 CLI orphans into its own process group and keeps
    /// hammering the CPU long after the window is gone.
    fn on_exit(&mut self) {
        let Some(job) = self.export_job.take() else {
            return;
        };
        job.cancel.store(true, Ordering::SeqCst);
        if let Some(h) = job.handle {
            let _ = h.join();
        }
    }
}

/// Find the data directory. Tries `./data`, then the CARGO_MANIFEST_DIR
/// relative path (handy when running via `cargo run -p spektrafilm-gui`).
fn pick_data_dir() -> PathBuf {
    let cwd = PathBuf::from("data");
    if cwd.is_dir() {
        return cwd;
    }
    // Workspace root: this crate lives at crates/spektrafilm-gui, so
    // ../.. from CARGO_MANIFEST_DIR is the workspace root.
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest).join("..").join("..").join("data");
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from("data")
}

/// Scan `<data_dir>/profiles/*.json`, parse each profile's `info`, and
/// bucket the results by `info.support`. Films go into the first vec,
/// papers (and any other print-stage supports) into the second.
/// Each entry carries the filename stem (the unique loader key) plus a
/// human-readable display label.
fn scan_profiles(data_dir: &Path) -> (Vec<ProfileEntry>, Vec<ProfileEntry>) {
    let mut films = Vec::new();
    let mut papers = Vec::new();
    let dir = data_dir.join("profiles");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return (films, papers);
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy().to_string();
        let Some(stem) = name_lossy.strip_suffix(".json") else {
            continue;
        };
        let stock = stem.to_string();
        // Cheap probe: just deserialize the file's `info` field. We
        // could skip the rest of the profile but `Profile` already does
        // the right thing — and we pay this once at startup.
        let (display, is_paper) = match profile::load_profile_by_name(data_dir, &stock) {
            Ok(p) => {
                let display = p.info.name.clone().unwrap_or_else(|| stock.clone());
                let is_paper = p.info.support == "paper" || p.info.stage == "printing";
                (display, is_paper)
            }
            Err(_) => (stock.clone(), false),
        };
        let entry = ProfileEntry { stock, display };
        if is_paper {
            papers.push(entry);
        } else {
            films.push(entry);
        }
    }
    films.sort_by(|a, b| a.display.cmp(&b.display));
    papers.sort_by(|a, b| a.display.cmp(&b.display));
    (films, papers)
}

/// Combo box that picks one of `entries` by its `stock` id (the
/// underlying file stem) while showing `display` (the human-readable
/// name) as the label. Falls back to showing the raw stock id if no
/// entry with the current `selected_stock` exists.
fn profile_combo(
    ui: &mut egui::Ui,
    salt: &str,
    label: &str,
    entries: &[ProfileEntry],
    selected_stock: &mut String,
) -> bool {
    ui.label(label);
    let display = entries
        .iter()
        .find(|e| &e.stock == selected_stock)
        .map(|e| e.display.clone())
        .unwrap_or_else(|| selected_stock.clone());
    let prev = selected_stock.clone();
    egui::ComboBox::from_id_salt(salt)
        .selected_text(&display)
        .width(ui.available_width().min(280.0))
        .show_ui(ui, |ui| {
            for entry in entries {
                ui.selectable_value(selected_stock, entry.stock.clone(), &entry.display);
            }
        });
    prev != *selected_stock
}

fn fps_label(ms: f32) -> &'static str {
    if ms < 16.0 {
        "60 fps"
    } else if ms < 33.0 {
        "30 fps"
    } else if ms < 67.0 {
        "15 fps"
    } else if ms < 200.0 {
        "5 fps"
    } else if ms < 400.0 {
        "2 fps"
    } else {
        ""
    }
}

fn pick_default_stock(entries: &[ProfileEntry], preferred: &str) -> String {
    if entries.iter().any(|e| e.stock == preferred) {
        preferred.to_string()
    } else {
        entries
            .first()
            .map(|e| e.stock.clone())
            .unwrap_or_else(|| preferred.to_string())
    }
}

fn load_image(path: &Path) -> Result<ImageBuf> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if is_raw_extension(&ext) {
        return load_raw(path);
    }
    let img = image::open(path).with_context(|| format!("opening image: {}", path.display()))?;
    let rgb = img.to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    let data: Vec<f32> = rgb.into_raw();
    let scalars: Vec<Scalar> = match ext.as_str() {
        "png" => data
            .into_par_iter()
            .map(|v| srgb_decode(from_f32(v)))
            .collect(),
        _ => data.into_par_iter().map(from_f32).collect(),
    };
    Ok(ImageBuf::from_data(w, h, scalars))
}

/// Camera RAW extensions supported via `imagepipe` / `rawloader`. The
/// underlying decoder handles many proprietary formats; this list is
/// the dispatch trigger only (we still try the decoder for unknown
/// extensions).
fn is_raw_extension(ext: &str) -> bool {
    matches!(
        ext,
        "dng"
            | "cr2"
            | "cr3"
            | "nef"
            | "nrw"
            | "arw"
            | "srf"
            | "sr2"
            | "raf"
            | "orf"
            | "rw2"
            | "pef"
            | "ptx"
            | "srw"
            | "x3f"
            | "iiq"
            | "3fr"
            | "ari"
            | "bay"
            | "crw"
            | "dcr"
            | "drf"
            | "erf"
            | "fff"
            | "k25"
            | "kdc"
            | "mef"
            | "mos"
            | "mrw"
            | "rwl"
    )
}

/// Decode a camera RAW file and return a **linear** image in sRGB
/// primaries. The pipeline runs `rawler` (a modern fork of rawloader
/// with broader camera coverage — Sony ARW, newer Nikon NEF, recent
/// DNGs) instead of the older `imagepipe` chain.
///
/// `rawler::RawDevelop::default()` already covers rescale → demosaic →
/// crop → white balance → camera→sRGB matrix. We drop the final
/// `SRgb` gamma step so we get LINEAR sRGB out (the spektrafilm
/// pipeline applies its own sRGB encoding at the very end).
///
/// Limitations: Lightroom-exported "lossy DNG" (`ljpeg sof.precision 8`)
/// is not supported by rawler — those files come back as an error,
/// which we surface in the GUI status bar.
fn load_raw(path: &Path) -> Result<ImageBuf> {
    use rawler::{decode_file, imgop::develop::{ProcessingStep, RawDevelop}};
    let raw = decode_file(path)
        .map_err(|e| anyhow::anyhow!("RAW decode failed: {e:?}"))?;
    let mut dev = RawDevelop::default();
    // Drop the sRGB gamma step — we want linear sRGB primaries, the
    // spektrafilm pipeline applies its own sRGB OETF at the end.
    dev.steps.retain(|s| !matches!(s, ProcessingStep::SRgb));
    let intermediate = dev
        .develop_intermediate(&raw)
        .map_err(|e| anyhow::anyhow!("RAW develop failed: {e:?}"))?;
    let dyn_img = intermediate
        .to_dynamic_image()
        .ok_or_else(|| anyhow::anyhow!("RAW develop: empty image"))?;
    // Force to RGB16 then promote to Scalar at full f32 precision.
    let rgb16 = dyn_img.to_rgb16();
    let (w, h) = (rgb16.width(), rgb16.height());
    let inv_max = 1.0f32 / 65535.0;
    let scalars: Vec<Scalar> = rgb16
        .as_raw()
        .par_iter()
        .map(|&v| from_f32(v as f32 * inv_max))
        .collect();
    Ok(ImageBuf::from_data(w, h, scalars))
}

/// Write the pipeline's RGB ImageBuf to disk. PNG is 8-bit (matches what
/// the preview shows); TIFF is 16-bit (more headroom — recommended for
/// further editing). The pipeline already emits sRGB-encoded values
/// clamped to [0, 1] when `output_cctf_encoding` is on, which is the
/// default for the GUI.
fn save_image(out: &ImageBuf, path: &Path) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_lowercase();
    let w = out.width;
    let h = out.height;
    match ext.as_str() {
        "tif" | "tiff" => {
            // Pack into 16-bit RGB.
            let n = (w as usize) * (h as usize) * 3;
            let mut buf = vec![0u16; n];
            buf.par_iter_mut()
                .zip(out.data.par_iter())
                .for_each(|(dst, &src)| {
                    *dst = (to_f32(src).clamp(0.0, 1.0) * 65535.0).round() as u16;
                });
            let img = image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(w, h, buf)
                .context("packing 16-bit TIFF buffer")?;
            img.save_with_format(path, image::ImageFormat::Tiff)
                .with_context(|| format!("writing TIFF {}", path.display()))?;
        }
        _ => {
            let n = (w as usize) * (h as usize) * 3;
            let mut buf = vec![0u8; n];
            buf.par_iter_mut()
                .zip(out.data.par_iter())
                .for_each(|(dst, &src)| {
                    *dst = (to_f32(src).clamp(0.0, 1.0) * 255.0).round() as u8;
                });
            let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w, h, buf)
                .context("packing 8-bit PNG buffer")?;
            img.save_with_format(path, image::ImageFormat::Png)
                .with_context(|| format!("writing PNG {}", path.display()))?;
        }
    }
    Ok(())
}

/// Convert the pipeline's post-sRGB-encoded RGB ImageBuf into an egui
/// ColorImage and upload as a texture. The pipeline emits values already
/// clamped to [0, 1] and sRGB-encoded when `output_cctf_encoding` is on,
/// so we just scale to 8-bit.
fn make_texture(ctx: &egui::Context, out: &ImageBuf) -> egui::TextureHandle {
    let w = out.width as usize;
    let h = out.height as usize;
    let mut rgba = vec![0u8; w * h * 4];
    rgba.par_chunks_exact_mut(4)
        .zip(out.data.par_chunks_exact(3))
        .for_each(|(dst, px)| {
            dst[0] = (to_f32(px[0]).clamp(0.0, 1.0) * 255.0).round() as u8;
            dst[1] = (to_f32(px[1]).clamp(0.0, 1.0) * 255.0).round() as u8;
            dst[2] = (to_f32(px[2]).clamp(0.0, 1.0) * 255.0).round() as u8;
            dst[3] = 255;
        });
    let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
    ctx.load_texture(
        "spektrafilm-output",
        color_image,
        egui::TextureOptions::LINEAR,
    )
}

/// Locate the f64-built `spektrafilm` CLI binary. Search order:
///   1. `$SPEKTRAFILM_F64_CLI` — explicit override, full path.
///   2. `spektrafilm-f64` next to the running GUI executable (release
///      builds: both binaries live in `target/release/`).
///   3. `spektrafilm-f64` on `$PATH`.
///   4. `target/release/spektrafilm-f64` relative to `CARGO_MANIFEST_DIR`
///      (handy when running via `cargo run`).
///
/// Returns a path that exists and is executable, or an explanatory
/// error pointing to the build command.
fn locate_f64_cli() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("SPEKTRAFILM_F64_CLI") {
        let path = PathBuf::from(&p);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!("SPEKTRAFILM_F64_CLI points at {p} but no such file"));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let p = dir.join("spektrafilm-f64");
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(p) = which_on_path("spektrafilm-f64") {
        return Ok(p);
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest)
            .join("..")
            .join("..")
            .join("target")
            .join("release")
            .join("spektrafilm-f64");
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(
        "no f64 CLI found. Build it with \
         `cargo build --release --features precision-f64 -p spektrafilm-cli`, \
         rename `target/release/spektrafilm` to `spektrafilm-f64`, \
         and either put it on PATH or set $SPEKTRAFILM_F64_CLI."
            .into(),
    )
}

/// Minimal PATH lookup so we don't pull in the `which` crate for one call.
fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// RAII guard so the params tempfile is unlinked on every code path
/// (early `?` return, panic during `Command::output`, normal exit).
/// Without this, a failed `serde_json::to_writer` or a panic mid-spawn
/// would leak user-state JSON into the OS temp dir.
struct TempPath(PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Run the f64 CLI as a child process. Writes the current
/// `RuntimeParams` to a temp JSON file, invokes
/// `spektrafilm-f64 process …`, polls for completion while watching
/// `cancel`, and surfaces the CLI's stderr verbatim on failure. If
/// `cancel` is set the child is SIGKILLed and `Err("cancelled")` is
/// returned — this prevents the orphan-process / system-freeze bug
/// where closing the GUI mid-export left the CPU pipeline running.
fn run_f64_export(
    cli_path: &Path,
    input: &Path,
    output: &Path,
    film: &str,
    paper: &str,
    params: &spektrafilm_core::params::RuntimeParams,
    data_dir: &Path,
    cancel: &AtomicBool,
) -> Result<()> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let temp = TempPath(std::env::temp_dir().join(format!(
        "spektrafilm-export-{}-{nanos}.json",
        std::process::id()
    )));
    {
        let f = std::fs::File::create(&temp.0)
            .with_context(|| format!("creating params tempfile {}", temp.0.display()))?;
        serde_json::to_writer(std::io::BufWriter::new(f), params)
            .context("serializing params to JSON")?;
    }

    let mut cmd = std::process::Command::new(cli_path);
    // Force the CPU backend: the wgpu shaders run f32 even in a
    // precision-f64 build (WGSL has no f64). Letting the child default
    // to GPU would silently demote the export back to f32 math.
    //
    // Pin Accelerate BLAS to single-threaded mode. Our CPU spectral
    // pipeline already chunks the large-M dgemm calls across rayon
    // (see `dgemm_row_parallel` in `cpu_backend.rs`). With Accelerate's
    // internal threading enabled, every per-chunk dgemm call contends
    // on Accelerate's global lock and the rayon parallelism collapses
    // to ~1.5 cores instead of saturating all of them. One-thread
    // Accelerate + rayon-level chunking is the configuration that
    // actually scales.
    cmd.env("SPEKTRAFILM_BACKEND", "cpu")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .arg("process")
        .arg(input)
        .arg("-o")
        .arg(output)
        .arg("--film")
        .arg(film)
        .arg("--paper")
        .arg(paper)
        .arg("--params")
        .arg(&temp.0)
        .arg("--data-dir")
        .arg(data_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if params.io.scan_film {
        cmd.arg("--scan-film");
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", cli_path.display()))?;

    // Poll loop. 100 ms is responsive to a Cancel click without
    // burning CPU while the export grinds through CPU f64 math.
    let status = loop {
        if let Some(s) = child.try_wait().context("waiting for f64 CLI")? {
            break s;
        }
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("cancelled");
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    // The child has exited; piped buffers are drained safely.
    let mut stderr_buf = String::new();
    if let Some(mut s) = child.stderr.take() {
        use std::io::Read;
        let _ = s.read_to_string(&mut stderr_buf);
    }

    if !status.success() {
        let trimmed = stderr_buf.trim();
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "(signal)".into());
        anyhow::bail!(
            "f64 CLI exited {code} — {}",
            if trimmed.is_empty() { "(no stderr)" } else { trimmed }
        );
    }
    Ok(())
}

/// macOS only: walk from the eframe `RawWindowHandle` down to the
/// `CAMetalLayer` and tag its colorspace as sRGB. Without this the
/// metal layer's `colorspace` is null and macOS treats the framebuffer
/// pixels as raw display primaries — sRGB content rendered into a
/// wide-gamut panel comes out oversaturated. Setting the layer's
/// colorspace makes the OS gamut-map exactly like it does for a
/// regular sRGB-tagged PNG opened in Preview, so what you see in the
/// GUI matches what you export.
#[cfg(target_os = "macos")]
fn tag_metal_layer_srgb<H: raw_window_handle::HasWindowHandle>(h: &H) -> Result<(), String> {
    use core_graphics::color_space::{CGColorSpace, kCGColorSpaceSRGB};
    use foreign_types::ForeignType;
    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    use raw_window_handle::RawWindowHandle;

    let handle = h
        .window_handle()
        .map_err(|e| format!("no window handle: {e}"))?;
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return Err("not an AppKit window".into());
    };
    let ns_view: *mut AnyObject = appkit.ns_view.as_ptr().cast();
    if ns_view.is_null() {
        return Err("ns_view is null".into());
    }

    // `kCGColorSpaceSRGB` is an extern static CFStringRef, accessing it
    // is unsafe in the 2024 edition's stricter model.
    let srgb_cs = unsafe { CGColorSpace::create_with_name(kCGColorSpaceSRGB) }
        .ok_or_else(|| "CGColorSpaceCreateWithName(sRGB) returned null".to_string())?;
    // `foreign_types::ForeignType::as_ptr` returns the opaque
    // `CGColorSpaceRef` (i.e. `*mut sys::CGColorSpace`). objc2's
    // `msg_send!` only knows how to pass standard pointer types as
    // arguments, so we cast through `*mut c_void` here. ObjC reads
    // it as a CGColorSpaceRef on the receiving side.
    let cs_ref = srgb_cs.as_ptr() as *mut std::ffi::c_void;

    unsafe {
        // NSView -> CALayer (the CAMetalLayer wgpu created).
        let layer: *mut AnyObject = msg_send![ns_view, layer];
        if layer.is_null() {
            return Err("ns_view.layer is null".into());
        }
        // Defensive: only call `setColorspace:` if the layer actually
        // responds to it. CAMetalLayer does, but `_NSOpenGLViewBackingLayer`
        // doesn't (and crashes the app with `unrecognized selector`).
        // This guard means the patch is a no-op for unexpected renderers
        // rather than a hard crash.
        let selector = objc2::sel!(setColorspace:);
        let responds: bool = msg_send![layer, respondsToSelector: selector];
        if !responds {
            return Err("layer does not respond to setColorspace: (renderer is not wgpu/Metal)".into());
        }
        // CAMetalLayer.colorspace is `retain`-strong, so ObjC takes its
        // own reference; we can let `srgb_cs` drop after the call.
        let _: () = msg_send![layer, setColorspace: cs_ref];
        tracing::info!("tagged CAMetalLayer.colorspace = sRGB");
    }
    Ok(())
}

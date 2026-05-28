/// Film and paper profile loading from JSON.
///
/// Mirrors Python `profiles/io.py`. Profiles contain spectral sensitivity data,
/// density curves, and metadata for each film stock and paper type.
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Profile {
    pub metadata: ProfileMetadata,
    pub info: ProfileInfo,
    pub data: ProfileData,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileMetadata {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub copyright: String,
    #[serde(default)]
    pub created: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub citation: String,
    #[serde(default)]
    pub datasource: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileInfo {
    pub stock: Option<String>,
    pub name: Option<String>,
    #[serde(rename = "type", default = "default_negative")]
    pub film_type: String,
    #[serde(default = "default_film")]
    pub support: String,
    #[serde(default = "default_filming")]
    pub stage: String,
    #[serde(rename = "use", default = "default_still")]
    pub usage: String,
    #[serde(default = "default_weak")]
    pub antihalation: String,
    pub target_print: Option<String>,
    #[serde(default = "default_color")]
    pub channel_model: String,
    #[serde(default = "default_status_m")]
    pub densitometer: String,
    #[serde(default = "default_log_sens")]
    pub log_sensitivity_density_over_min: f64,
    #[serde(default = "default_d55")]
    pub reference_illuminant: String,
    #[serde(default = "default_d50")]
    pub viewing_illuminant: String,
}

fn default_negative() -> String {
    "negative".into()
}
fn default_film() -> String {
    "film".into()
}
fn default_filming() -> String {
    "filming".into()
}
fn default_still() -> String {
    "still".into()
}
fn default_weak() -> String {
    "weak".into()
}
fn default_color() -> String {
    "color".into()
}
fn default_status_m() -> String {
    "status_M".into()
}
fn default_log_sens() -> f64 {
    0.2
}
fn default_d55() -> String {
    "D55".into()
}
fn default_d50() -> String {
    "D50".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileData {
    #[serde(default)]
    pub wavelengths: Vec<f64>,
    #[serde(default, deserialize_with = "deser_zero_matrix")]
    pub log_sensitivity: Vec<Vec<f64>>,
    #[serde(default, deserialize_with = "deser_zero_vec")]
    pub hanatos2025_adaptation_window_params: Vec<f64>,
    #[serde(default, deserialize_with = "deser_zero_matrix")]
    pub hanatos2025_adaptation_surface_params: Vec<Vec<f64>>,
    /// NaN-preserving: null values mean "no data at this wavelength"
    #[serde(default, deserialize_with = "deser_nullable_matrix")]
    pub channel_density: Vec<Vec<f64>>,
    /// NaN-preserving: null values mean "no data at this wavelength"
    #[serde(default, deserialize_with = "deser_nullable_vec")]
    pub base_density: Vec<f64>,
    /// NaN-preserving
    #[serde(default, deserialize_with = "deser_nullable_vec")]
    pub midscale_neutral_density: Vec<f64>,
    #[serde(default)]
    pub log_exposure: Vec<f64>,
    #[serde(default, deserialize_with = "deser_zero_matrix")]
    pub density_curves: Vec<Vec<f64>>,
    #[serde(default, deserialize_with = "deser_zero_tensor")]
    pub density_curves_layers: Vec<Vec<Vec<f64>>>,
}

/// Deserialize with null → 0.0 (for data that must be finite: sensitivity, density curves).
fn deser_zero_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<Option<f64>> = Deserialize::deserialize(deserializer)?;
    Ok(v.into_iter().map(|x| x.unwrap_or(0.0)).collect())
}
fn deser_zero_matrix<'de, D>(deserializer: D) -> Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<Vec<Option<f64>>> = Deserialize::deserialize(deserializer)?;
    Ok(v.into_iter()
        .map(|row| row.into_iter().map(|x| x.unwrap_or(0.0)).collect())
        .collect())
}
fn deser_zero_tensor<'de, D>(deserializer: D) -> Result<Vec<Vec<Vec<f64>>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<Vec<Vec<Option<f64>>>> = Deserialize::deserialize(deserializer)?;
    Ok(v.into_iter()
        .map(|m| {
            m.into_iter()
                .map(|r| r.into_iter().map(|x| x.unwrap_or(0.0)).collect())
                .collect()
        })
        .collect())
}

/// Deserialize with null → NaN (for spectral data where null means "no measurement").
fn deser_nullable_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<Option<f64>> = Deserialize::deserialize(deserializer)?;
    Ok(v.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
}

/// Deserialize a Vec<Vec<f64>> where elements may be null.
/// For channel_density, null means "no data at this wavelength" — we use NaN
/// to propagate this correctly through spectral calculations (matching Python).
fn deser_nullable_matrix<'de, D>(deserializer: D) -> Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Vec<Vec<Option<f64>>> = Deserialize::deserialize(deserializer)?;
    Ok(v.into_iter()
        .map(|row| row.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
        .collect())
}

impl Profile {
    pub fn is_negative(&self) -> bool {
        self.info.film_type == "negative"
    }
    pub fn is_positive(&self) -> bool {
        self.info.film_type == "positive"
    }
    pub fn is_film(&self) -> bool {
        self.info.support == "film"
    }
    pub fn is_paper(&self) -> bool {
        self.info.support == "paper"
    }
    pub fn is_color(&self) -> bool {
        self.info.channel_model == "color"
    }
    pub fn is_bw(&self) -> bool {
        self.info.channel_model == "bw"
    }
    pub fn is_filming(&self) -> bool {
        self.info.stage == "filming"
    }
    pub fn is_printing(&self) -> bool {
        self.info.stage == "printing"
    }

    /// Get density curves as [N][3] f64 array for calibration precision.
    pub fn density_curves_f64(&self) -> Vec<[f64; 3]> {
        self.data
            .density_curves
            .iter()
            .map(|row| {
                [
                    row.get(0).copied().unwrap_or(0.0),
                    row.get(1).copied().unwrap_or(0.0),
                    row.get(2).copied().unwrap_or(0.0),
                ]
            })
            .collect()
    }

    /// Get log_exposure as f64 slice.
    pub fn log_exposure_f64(&self) -> Vec<f64> {
        self.data.log_exposure.clone()
    }

    /// Get density curves as [N][3] f32 array for fast interpolation.
    pub fn density_curves_f32(&self) -> Vec<[f32; 3]> {
        self.data
            .density_curves
            .iter()
            .map(|row| {
                [
                    row.get(0).copied().unwrap_or(0.0) as f32,
                    row.get(1).copied().unwrap_or(0.0) as f32,
                    row.get(2).copied().unwrap_or(0.0) as f32,
                ]
            })
            .collect()
    }

    /// Get log_exposure as f32 slice.
    pub fn log_exposure_f32(&self) -> Vec<f32> {
        self.data.log_exposure.iter().map(|&v| v as f32).collect()
    }

    /// Get log_sensitivity as [81][3] f32 array.
    pub fn log_sensitivity_f32(&self) -> Vec<[f32; 3]> {
        self.data
            .log_sensitivity
            .iter()
            .map(|row| {
                [
                    row.get(0).copied().unwrap_or(0.0) as f32,
                    row.get(1).copied().unwrap_or(0.0) as f32,
                    row.get(2).copied().unwrap_or(0.0) as f32,
                ]
            })
            .collect()
    }

    /// Get log_sensitivity as [81][3] f64 array (precision-preserving).
    pub fn log_sensitivity_f64(&self) -> Vec<[f64; 3]> {
        self.data
            .log_sensitivity
            .iter()
            .map(|row| {
                [
                    row.get(0).copied().unwrap_or(0.0),
                    row.get(1).copied().unwrap_or(0.0),
                    row.get(2).copied().unwrap_or(0.0),
                ]
            })
            .collect()
    }
}

/// Load a profile from a JSON file on disk.
pub fn load_profile(path: &Path) -> Result<Profile, ProfileError> {
    let file =
        std::fs::File::open(path).map_err(|e| ProfileError::Io(path.display().to_string(), e))?;
    let reader = std::io::BufReader::new(file);
    let profile: Profile = serde_json::from_reader(reader)
        .map_err(|e| ProfileError::Parse(path.display().to_string(), e))?;
    validate_profile(&profile)?;
    Ok(profile)
}

/// Load a profile by stock name from a data directory.
pub fn load_profile_by_name(data_dir: &Path, stock: &str) -> Result<Profile, ProfileError> {
    let path = data_dir.join("profiles").join(format!("{stock}.json"));
    load_profile(&path)
}

fn validate_profile(profile: &Profile) -> Result<(), ProfileError> {
    let data = &profile.data;
    if data.log_exposure.is_empty() {
        return Err(ProfileError::Validation("log_exposure is empty".into()));
    }
    if data.density_curves.len() != data.log_exposure.len() {
        return Err(ProfileError::Validation(
            "density_curves length must match log_exposure length".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("loading profile {0}: {1}")]
    Io(String, std::io::Error),
    #[error("parsing profile {0}: {1}")]
    Parse(String, serde_json::Error),
    #[error("invalid profile: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_kodak_portra_400() {
        // This test requires the data directory to be present
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("data/profiles/kodak_portra_400.json");
        if !path.exists() {
            eprintln!("Skipping test — profile not found at {}", path.display());
            return;
        }
        let profile = load_profile(&path).unwrap();
        assert_eq!(profile.info.stock.as_deref(), Some("kodak_portra_400"));
        assert_eq!(profile.info.film_type, "negative");
        assert_eq!(profile.info.support, "film");
        assert_eq!(profile.data.wavelengths.len(), 81);
        assert_eq!(profile.data.log_exposure.len(), 256);
        assert_eq!(profile.data.density_curves.len(), 256);
    }
}

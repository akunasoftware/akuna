use std::path::Path;

use burn_wgpu::Wgpu;

use crate::types::extraction::{
    DetectedFileType, ExtractionMetadata, FileExtractionError,
};

pub(super) struct FileMetadata {
    pub(super) detected_type: DetectedFileType,
    pub(super) metadata: ExtractionMetadata,
}

/// Uses and returns Magika filetype detection, alongside basic useful file meta
pub(super) async fn extract_metadata(
    file_path: &Path,
) -> Result<FileMetadata, FileExtractionError> {
    let detected_type = detect_file_type(file_path).await?;
    let extension = file_path
        .extension()
        .map(|extension| extension.to_string_lossy().into_owned());
    let stem = file_path
        .file_stem()
        .map(|file_stem| file_stem.to_string_lossy().into_owned());

    Ok(FileMetadata {
        metadata: ExtractionMetadata {
            stem,
            extension,
            label: detected_type.label.clone(),
            mime_type: detected_type.mime_type.clone(),
            description: detected_type.description.clone(),
        },
        detected_type,
    })
}

/// Uses Google "Magika" ML model to intelligently infer file type from given path
async fn detect_file_type(
    file_path: &Path,
) -> Result<DetectedFileType, FileExtractionError> {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    let mut magika = burn_magika::Session::<Wgpu>::new(&device)?;
    let type_info = magika.identify_file_async(file_path).await?.info();

    Ok(DetectedFileType {
        mime_type: type_info.mime_type.to_string(),
        label: type_info.label.to_string(),
        description: type_info.description.to_string(),
        is_text: type_info.is_text,
    })
}

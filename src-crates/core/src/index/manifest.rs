use std::fs;
use std::io::{ErrorKind, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::chunking::ChunkingOptions;
use crate::embedding::EmbeddingModel;
use crate::index::IndexOptions;

const MANIFEST_FILE: &str = "manifest.yaml";
const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StoredConfig {
    pub(super) embedding_model: EmbeddingModel,
    pub(super) chunking: ChunkingOptions,
    pub(super) fulltext: bool,
    pub(super) graph: bool,
}

#[derive(Deserialize, Serialize)]
struct ManifestFile {
    schema_version: u32,
    embedding_model: String,
    chunking: ChunkingOptions,
    fulltext: bool,
    graph: bool,
}

impl StoredConfig {
    /// Builds stored configuration from index options.
    pub(super) fn from_options(options: &IndexOptions) -> Self {
        Self {
            embedding_model: options.embedding_model,
            chunking: options.chunking.clone(),
            fulltext: options.fulltext,
            graph: options.graph,
        }
    }
}

/// Creates or validates the index manifest.
pub(super) fn ensure(root: &Path, requested: &StoredConfig) -> Result<()> {
    let non_empty = root.exists() && directory_non_empty(root)?;
    fs::create_dir_all(root).with_context(|| {
        format!("failed to create index root {}", root.display())
    })?;

    let path = root.join(MANIFEST_FILE);
    match fs::read_to_string(&path) {
        Ok(contents) => validate(&path, &contents, requested),
        Err(error) if error.kind() == ErrorKind::NotFound && non_empty => {
            bail!(
                "index manifest missing for non-empty root {}",
                root.display()
            )
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            write(&path, requested)
        }
        Err(error) => Err(error).with_context(|| {
            format!("failed to read index manifest {}", path.display())
        }),
    }
}

/// Returns whether a directory has stored entries.
fn directory_non_empty(path: &Path) -> Result<bool> {
    let mut entries = fs::read_dir(path).with_context(|| {
        format!("failed to read index root {}", path.display())
    })?;
    entries
        .next()
        .transpose()
        .with_context(|| {
            format!("failed to read index root {}", path.display())
        })
        .map(|entry| entry.is_some())
}

/// Writes a new manifest file.
fn write(path: &Path, config: &StoredConfig) -> Result<()> {
    let manifest = ManifestFile::from_config(config);
    let contents = serde_norway::to_string(&manifest)
        .context("failed to serialize index manifest")?;
    let parent = path.parent().context("index manifest path has no parent")?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).with_context(|| {
            format!("failed to create index manifest {}", path.display())
        })?;
    temporary.write_all(contents.as_bytes()).with_context(|| {
        format!("failed to write index manifest {}", path.display())
    })?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| {
            format!("failed to write index manifest {}", path.display())
        })?;

    Ok(())
}

/// Validates existing manifest contents.
fn validate(
    path: &Path,
    contents: &str,
    requested: &StoredConfig,
) -> Result<()> {
    let manifest = serde_norway::from_str::<ManifestFile>(contents)
        .with_context(|| {
            format!("index manifest corrupt at {}", path.display())
        })?;
    if manifest.schema_version != SCHEMA_VERSION {
        bail!(
            "index manifest mismatch for schema_version: stored {}, requested {}",
            manifest.schema_version,
            SCHEMA_VERSION,
        );
    }
    let stored = manifest.into_config()?;
    compare(
        stored.embedding_model,
        requested.embedding_model,
        "embedding_model",
    )?;
    compare(stored.fulltext, requested.fulltext, "fulltext")?;
    compare(stored.graph, requested.graph, "graph")?;
    if stored.chunking != requested.chunking {
        bail!(
            "index manifest mismatch for chunking: stored {:?}, requested {:?}",
            stored.chunking,
            requested.chunking,
        );
    }

    Ok(())
}

/// Compares one manifest field.
fn compare<T>(stored: T, requested: T, field: &str) -> Result<()>
where
    T: std::fmt::Debug + PartialEq,
{
    if stored == requested {
        return Ok(());
    }

    bail!(
        "index manifest mismatch for {field}: stored {stored:?}, requested {requested:?}",
    )
}

impl ManifestFile {
    /// Builds a manifest file from stored configuration.
    fn from_config(config: &StoredConfig) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            embedding_model: embedding_model_key(config.embedding_model)
                .to_string(),
            chunking: config.chunking.clone(),
            fulltext: config.fulltext,
            graph: config.graph,
        }
    }

    /// Converts a manifest file into stored configuration.
    fn into_config(self) -> Result<StoredConfig> {
        Ok(StoredConfig {
            embedding_model: parse_embedding_model(&self.embedding_model)?,
            chunking: self.chunking,
            fulltext: self.fulltext,
            graph: self.graph,
        })
    }
}

/// Returns the manifest key for an embedding model.
fn embedding_model_key(model: EmbeddingModel) -> &'static str {
    match model {
        EmbeddingModel::MiniLmL6 => "mini_lm_l6",
        EmbeddingModel::MiniLmL12 => "mini_lm_l12",
        EmbeddingModel::BgeSmallEnV15 => "bge_small_en_v15",
        EmbeddingModel::BgeBaseEnV15 => "bge_base_en_v15",
        EmbeddingModel::BgeLargeEnV15 => "bge_large_en_v15",
        EmbeddingModel::AllMpnetBaseV2 => "all_mpnet_base_v2",
        EmbeddingModel::BgeM3 => "bge_m3",
    }
}

/// Parses an embedding model manifest key.
fn parse_embedding_model(value: &str) -> Result<EmbeddingModel> {
    match value {
        "mini_lm_l6" => Ok(EmbeddingModel::MiniLmL6),
        "mini_lm_l12" => Ok(EmbeddingModel::MiniLmL12),
        "bge_small_en_v15" => Ok(EmbeddingModel::BgeSmallEnV15),
        "bge_base_en_v15" => Ok(EmbeddingModel::BgeBaseEnV15),
        "bge_large_en_v15" => Ok(EmbeddingModel::BgeLargeEnV15),
        "all_mpnet_base_v2" => Ok(EmbeddingModel::AllMpnetBaseV2),
        "bge_m3" => Ok(EmbeddingModel::BgeM3),
        _ => bail!("index manifest has unknown embedding_model {value:?}"),
    }
}

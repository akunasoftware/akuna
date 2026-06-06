use crate::extraction::{
    DocumentContent, ExtractionMetadata, ExtractionPart, PartKind, pipeline,
    provenance,
};
use std::collections::HashMap;

/// Extract structured syntax parts from text content when language is supported.
pub(in crate::extraction) fn extract(
    text: &str,
    metadata: &ExtractionMetadata,
) -> Option<DocumentContent> {
    let extension = metadata.extension.as_deref();
    let started = std::time::Instant::now();
    let ranges = syntax_part_ranges(text, extension)?;
    let parts = ranges
        .into_iter()
        .enumerate()
        .filter_map(|(index, range)| {
            let part_text = text.get(range.range.clone())?.trim();
            if part_text.is_empty() {
                return None;
            }

            Some(ExtractionPart {
                index,
                kind: PartKind::Code,
                text: Some(text.get(range.range.clone())?.to_owned()),
                provenance: Some(provenance::from_byte_range(
                    range.range.start,
                    range.range.end,
                )),
            })
        })
        .collect::<Vec<_>>();

    let part_count = parts.len();
    let duration_ms = started.elapsed().as_millis() as u64;
    (part_count > 1).then_some(DocumentContent {
        canonical_text: Some(text.to_owned()),
        parts,
        pipeline: vec![pipeline::step(
            "parsing",
            "tree-sitter",
            duration_ms,
            HashMap::from([("parts".to_owned(), part_count)]),
        )],
    })
}

/// Tree-sitter byte range for a leaf named syntax node.
struct SyntaxPartRange {
    range: std::ops::Range<usize>,
}

/// Extract lowest-depth named syntax node ranges.
fn syntax_part_ranges(
    content: &str,
    file_extension: Option<&str>,
) -> Option<Vec<SyntaxPartRange>> {
    let language = language_for_extension(file_extension?)?;
    let tree = syntax_tree(content, language)?;

    if tree.root_node().has_error() {
        return None;
    }

    let mut ranges = Vec::new();
    collect_leaf_named_ranges(tree.root_node(), &mut ranges);

    (!ranges.is_empty()).then_some(ranges)
}

/// Collect named nodes with no named children.
fn collect_leaf_named_ranges(
    node: tree_sitter::Node<'_>,
    ranges: &mut Vec<SyntaxPartRange>,
) {
    if node.parent().is_some() && node.is_named() && !has_named_children(node) {
        ranges.push(SyntaxPartRange {
            range: node.byte_range(),
        });
        return;
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_leaf_named_ranges(child, ranges);
    }
}

/// Returns whether a node contains named children.
fn has_named_children(node: tree_sitter::Node<'_>) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).next().is_some()
}

/// Parses content into a tree-sitter syntax tree.
fn syntax_tree(
    content: &str,
    language: tree_sitter::Language,
) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();

    if parser.set_language(&language).is_err() {
        return None;
    }

    parser.parse(content, None)
}

/// Maps a file extension to its tree-sitter language, if supported.
fn language_for_extension(
    file_extension: &str,
) -> Option<tree_sitter::Language> {
    match file_extension {
        "bash" | "sh" => Some(tree_sitter_bash::LANGUAGE.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "containerfile" | "dockerfile" => {
            Some(tree_sitter_containerfile::LANGUAGE.into())
        }
        "cs" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        "css" => Some(tree_sitter_css::LANGUAGE.into()),
        "dart" => Some(tree_sitter_dart::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "graphql" | "gql" => Some(tree_sitter_graphql::LANGUAGE.into()),
        "groovy" | "gradle" => Some(tree_sitter_groovy::LANGUAGE.into()),
        "hcl" | "tf" | "terraform" => Some(tree_sitter_hcl::LANGUAGE.into()),
        "html" => Some(tree_sitter_html::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "json" => Some(tree_sitter_json::LANGUAGE.into()),
        "lua" => Some(tree_sitter_lua::LANGUAGE.into()),
        "md" | "markdown" => Some(tree_sitter_md::LANGUAGE.into()),
        "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "rb" | "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "scala" | "sc" => Some(tree_sitter_scala::LANGUAGE.into()),
        "scss" => Some(tree_sitter_scss::language()),
        "sql" => Some(tree_sitter_sequel::LANGUAGE.into()),
        "svelte" => Some(tree_sitter_svelte_next::LANGUAGE.into()),
        "swift" => Some(tree_sitter_swift::LANGUAGE.into()),
        "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "vue" => Some(tree_sitter_vue_updated::language()),
        "yaml" | "yml" => Some(tree_sitter_yaml::LANGUAGE.into()),
        _ => None,
    }
}

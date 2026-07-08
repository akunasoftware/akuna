use crate::chunking::{PartKind, Segment};

/// Extract structured code segments when language support exists.
pub(crate) fn segment_code<'a>(
    content: &'a str,
    file_extension: Option<&str>,
) -> Option<Vec<Segment<'a>>> {
    let language = language_for_extension(file_extension?)?;
    let tree = syntax_tree(content, language)?;

    if tree.root_node().has_error() {
        return None;
    }

    let mut ranges = Vec::new();
    collect_leaf_named_ranges(tree.root_node(), &mut ranges);

    let segments = ranges
        .into_iter()
        .filter_map(|range| {
            let text = content.get(range.clone())?;
            (!text.trim().is_empty()).then_some(Segment {
                text,
                kind: PartKind::Code,
                byte_range: range,
            })
        })
        .collect::<Vec<_>>();

    (!segments.is_empty()).then_some(segments)
}

/// Returns whether an extension has a supported syntax grammar.
pub(crate) fn supports_code_extension(file_extension: Option<&str>) -> bool {
    file_extension.and_then(language_for_extension).is_some()
}

/// Collect source ranges from leaf syntax nodes.
fn collect_leaf_named_ranges(
    node: tree_sitter::Node<'_>,
    ranges: &mut Vec<std::ops::Range<usize>>,
) {
    if node.parent().is_some() && node.is_named() && !has_named_children(node) {
        ranges.push(node.byte_range());
        return;
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_leaf_named_ranges(child, ranges);
    }
}

/// Returns whether a syntax node contains named children.
fn has_named_children(node: tree_sitter::Node<'_>) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).next().is_some()
}

/// Build a syntax tree for source content.
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

/// Resolve parser support from a file extension.
fn language_for_extension(
    file_extension: &str,
) -> Option<tree_sitter::Language> {
    let file_extension = file_extension.to_ascii_lowercase();
    match file_extension.as_str() {
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

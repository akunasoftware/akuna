use std::ops::Range;

pub(super) fn chunk_content<'a>(
    content: &'a str,
    delimiters: Option<&[u8]>,
    target_size: usize,
    file_extension: Option<&str>,
) -> Option<Vec<&'a str>> {
    let language = language_for_extension(file_extension?)?;
    let tree = syntax_tree(content, language)?;

    if tree.root_node().has_error() {
        return None;
    }

    let mut ranges = Vec::new();
    collect_chunk_ranges(tree.root_node(), target_size, &mut ranges);

    if ranges.is_empty() {
        return None;
    }

    let mut chunks = Vec::new();
    let mut chunk_start = 0;
    let mut chunk_end = ranges[0].end;

    for range in ranges.into_iter().skip(1) {
        if range.end - chunk_start > target_size {
            push_chunk(
                &mut chunks,
                content,
                chunk_start,
                chunk_end,
                delimiters,
                target_size,
            );
            chunk_start = chunk_end;
        }

        chunk_end = range.end;
    }

    push_chunk(
        &mut chunks,
        content,
        chunk_start,
        content.len(),
        delimiters,
        target_size,
    );

    Some(chunks)
}

fn collect_chunk_ranges(
    node: tree_sitter::Node<'_>,
    target_size: usize,
    ranges: &mut Vec<Range<usize>>,
) {
    let mut cursor = node.walk();
    let children = node.named_children(&mut cursor).collect::<Vec<_>>();

    if children.is_empty() {
        ranges.push(node.byte_range());
        return;
    }

    if node.parent().is_some() && node.byte_range().len() <= target_size {
        ranges.push(node.byte_range());
        return;
    }

    for child in children {
        collect_chunk_ranges(child, target_size, ranges);
    }
}

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

fn push_chunk<'a>(
    chunks: &mut Vec<&'a str>,
    content: &'a str,
    start: usize,
    end: usize,
    delimiters: Option<&[u8]>,
    target_size: usize,
) {
    let text = &content[start..end];

    if text.len() <= target_size {
        chunks.push(text);
        return;
    }

    chunks.extend(super::chunk_with_delimiters(text, delimiters, target_size));
}

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

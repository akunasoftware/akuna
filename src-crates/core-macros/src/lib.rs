//! Proc macros for core domain types.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, parse_macro_input};

/// Derives `akuna_core::graph::primitives::GraphNode`.
///
/// Use `#[graph(node_type(name = "Name"))]` on fixed node types.
/// Use `#[graph(id)]`, `#[graph(name)]`, `#[graph(description)]`, and `#[graph(metadata)]` on named fields.
#[proc_macro_derive(GraphNode, attributes(graph))]
pub fn derive_graph_node(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    expand_graph_node(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derives `akuna_core::graph::primitives::GraphEdge`.
///
/// Use `#[graph(source_labels)]`, `#[graph(source)]`,
/// `#[graph(predicate)]`, `#[graph(target)]`, and
/// `#[graph(target_labels)]` on named fields.
#[proc_macro_derive(GraphEdge, attributes(graph))]
pub fn derive_graph_edge(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    expand_graph_edge(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_graph_node(
    input: DeriveInput,
) -> syn::Result<proc_macro2::TokenStream> {
    let name = input.ident;
    let config = graph_node_config(&input.attrs)?;
    let fields = graph_node_fields(&input.data)?;
    let id_field = fields.id;
    let labels = config.labels;
    let labels_field = fields.labels;
    let name_field = fields.name;
    let description_field = fields.description;
    let metadata_field = fields.metadata;
    let metadata_ty = fields.metadata_ty;
    let (impl_generics, type_generics, where_clause) =
        input.generics.split_for_impl();
    let labels_method = match (labels, &labels_field) {
        (Some(labels), _) => quote! { vec![#(#labels),*] },
        (None, Some(field)) => quote! {
            self.#field.iter().map(String::as_str).collect()
        },
        (None, None) => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "GraphNode requires #[graph(node_type(name = \"...\"))] or #[graph(labels)] field",
            ));
        }
    };
    let labels_assignment = labels_field
        .as_ref()
        .map(|field| quote! { #field: labels, });
    let name_method = match (config.name, &name_field) {
        (Some(name), _) => quote! { #name },
        (None, Some(field)) => quote! { self.#field.as_ref() },
        (None, None) => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "GraphNode requires #[graph(node_type(name = \"...\"))] or #[graph(name)] field",
            ));
        }
    };
    let description_method = match (config.description, &description_field) {
        (Some(description), _) => quote! { Some(#description) },
        (None, Some(field)) => quote! { self.#field.as_deref() },
        (None, None) => quote! { None },
    };
    let name_assignment =
        name_field.as_ref().map(|field| quote! { #field: name, });
    let description_assignment = description_field
        .as_ref()
        .map(|field| quote! { #field: description, });

    Ok(quote! {
        impl #impl_generics akuna_core::graph::primitives::GraphNode for #name #type_generics #where_clause {
            type Metadata = #metadata_ty;

            fn labels(&self) -> Vec<&str> {
                #labels_method
            }

            fn id(&self) -> &str {
                self.#id_field.as_ref()
            }

            fn name(&self) -> &str {
                #name_method
            }

            fn description(&self) -> Option<&str> {
                #description_method
            }

            fn metadata(&self) -> Option<&Self::Metadata> {
                self.#metadata_field.as_ref()
            }

            fn from_graph_parts(
                id: String,
                labels: Vec<String>,
                name: String,
                description: Option<String>,
                metadata: Option<Self::Metadata>,
            ) -> Self {
                Self {
                    #id_field: id,
                    #labels_assignment
                    #name_assignment
                    #description_assignment
                    #metadata_field: metadata,
                }
            }
        }
    })
}

fn expand_graph_edge(
    input: DeriveInput,
) -> syn::Result<proc_macro2::TokenStream> {
    reject_graph_edge_type_attrs(&input.attrs)?;

    let name = input.ident;
    let fields = graph_edge_fields(&input.data)?;
    let source_labels_field = fields.source_labels;
    let source_field = fields.source;
    let predicate_field = fields.predicate;
    let target_field = fields.target;
    let target_labels_field = fields.target_labels;
    let (impl_generics, type_generics, where_clause) =
        input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics akuna_core::graph::primitives::GraphEdge for #name #type_generics #where_clause {
            fn source_labels(&self) -> Vec<&str> {
                self.#source_labels_field.iter().map(String::as_str).collect()
            }

            fn source(&self) -> &str {
                self.#source_field.as_ref()
            }

            fn predicate(&self) -> &str {
                self.#predicate_field.as_ref()
            }

            fn target(&self) -> &str {
                self.#target_field.as_ref()
            }

            fn target_labels(&self) -> Vec<&str> {
                self.#target_labels_field.iter().map(String::as_str).collect()
            }
        }
    })
}

fn reject_graph_edge_type_attrs(attrs: &[syn::Attribute]) -> syn::Result<()> {
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("graph")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("node_type") {
                return Err(meta.error(
                    "GraphEdge does not support graph node type attributes",
                ));
            }

            Err(meta.error("unsupported graph edge attribute"))
        })?;
    }

    Ok(())
}

struct GraphNodeFields {
    id: syn::Ident,
    labels: Option<syn::Ident>,
    name: Option<syn::Ident>,
    description: Option<syn::Ident>,
    metadata: syn::Ident,
    metadata_ty: syn::Type,
}

struct GraphNodeConfig {
    labels: Option<Vec<String>>,
    name: Option<String>,
    description: Option<String>,
}

fn graph_node_config(attrs: &[syn::Attribute]) -> syn::Result<GraphNodeConfig> {
    let mut labels = None;
    let mut name = None;
    let mut description = None;

    for attr in attrs.iter().filter(|attr| attr.path().is_ident("graph")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("node_type") {
                meta.parse_nested_meta(|meta| {
                    if meta.path.is_ident("name") {
                        let value = meta.value()?;
                        name = Some(value.parse::<LitStr>()?.value());
                        return Ok(());
                    }

                    if meta.path.is_ident("description") {
                        let value = meta.value()?;
                        description = Some(value.parse::<LitStr>()?.value());
                        return Ok(());
                    }

                    Err(meta.error("unsupported graph node type attribute"))
                })?;
                return Ok(());
            }

            Err(meta.error("unsupported graph attribute"))
        })?;
    }

    labels = labels.or_else(|| name.clone().map(|name| vec![name]));

    Ok(GraphNodeConfig {
        labels,
        name,
        description,
    })
}

fn graph_node_fields(data: &Data) -> syn::Result<GraphNodeFields> {
    let Data::Struct(data) = data else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "GraphNode can only be derived for structs",
        ));
    };

    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &data.fields,
            "GraphNode requires named fields",
        ));
    };

    let mut id = None;
    let mut labels = None;
    let mut name = None;
    let mut description = None;
    let mut metadata = None;
    let mut metadata_ty = None;

    for field in &fields.named {
        for attr in field
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("graph"))
        {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("id") {
                    set_graph_node_field(&mut id, field, "id")?;
                    return Ok(());
                }

                if meta.path.is_ident("labels") {
                    set_graph_node_field(&mut labels, field, "labels")?;
                    return Ok(());
                }

                if meta.path.is_ident("name") {
                    set_graph_node_field(&mut name, field, "name")?;
                    return Ok(());
                }

                if meta.path.is_ident("description") {
                    set_graph_node_field(
                        &mut description,
                        field,
                        "description",
                    )?;
                    return Ok(());
                }

                if meta.path.is_ident("metadata") {
                    set_graph_node_field(&mut metadata, field, "metadata")?;
                    metadata_ty = option_inner_type(&field.ty).cloned();
                    return Ok(());
                }

                Err(meta.error("unsupported graph field attribute"))
            })?;
        }
    }

    Ok(GraphNodeFields {
        id: required_graph_node_field(id, fields, "id")?,
        labels,
        name,
        description,
        metadata: required_graph_node_field(metadata, fields, "metadata")?,
        metadata_ty: metadata_ty.ok_or_else(|| {
            syn::Error::new_spanned(
                fields,
                "GraphNode metadata field must be Option<T>",
            )
        })?,
    })
}

fn option_inner_type(ty: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;

    if segment.ident != "Option" {
        return None;
    }

    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let syn::GenericArgument::Type(inner) = args.args.first()? else {
        return None;
    };

    Some(inner)
}

fn set_graph_node_field(
    target: &mut Option<syn::Ident>,
    field: &syn::Field,
    role: &str,
) -> syn::Result<()> {
    if target.is_some() {
        return Err(syn::Error::new_spanned(
            field,
            format!("GraphNode requires exactly one #[graph({role})] field"),
        ));
    }

    *target = field.ident.clone();
    Ok(())
}

fn required_graph_node_field(
    field: Option<syn::Ident>,
    fields: &syn::FieldsNamed,
    role: &str,
) -> syn::Result<syn::Ident> {
    field.ok_or_else(|| {
        syn::Error::new_spanned(
            fields,
            format!("GraphNode requires exactly one #[graph({role})] field"),
        )
    })
}

struct GraphEdgeFields {
    source_labels: syn::Ident,
    source: syn::Ident,
    predicate: syn::Ident,
    target: syn::Ident,
    target_labels: syn::Ident,
}

fn graph_edge_fields(data: &Data) -> syn::Result<GraphEdgeFields> {
    let Data::Struct(data) = data else {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "GraphEdge can only be derived for structs",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &data.fields,
            "GraphEdge requires named fields",
        ));
    };

    let mut source_labels = None;
    let mut source = None;
    let mut predicate = None;
    let mut target = None;
    let mut target_labels = None;

    for field in &fields.named {
        for attr in field
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("graph"))
        {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("source_labels") {
                    set_graph_edge_field(
                        &mut source_labels,
                        field,
                        "source_labels",
                    )?;
                    return Ok(());
                }

                if meta.path.is_ident("source") {
                    set_graph_edge_field(&mut source, field, "source")?;
                    return Ok(());
                }

                if meta.path.is_ident("predicate") {
                    set_graph_edge_field(&mut predicate, field, "predicate")?;
                    return Ok(());
                }

                if meta.path.is_ident("target") {
                    set_graph_edge_field(&mut target, field, "target")?;
                    return Ok(());
                }

                if meta.path.is_ident("target_labels") {
                    set_graph_edge_field(
                        &mut target_labels,
                        field,
                        "target_labels",
                    )?;
                    return Ok(());
                }

                Err(meta.error("unsupported graph field attribute"))
            })?;
        }
    }

    Ok(GraphEdgeFields {
        source_labels: required_graph_edge_field(
            source_labels,
            fields,
            "source_labels",
        )?,
        source: required_graph_edge_field(source, fields, "source")?,
        predicate: required_graph_edge_field(predicate, fields, "predicate")?,
        target: required_graph_edge_field(target, fields, "target")?,
        target_labels: required_graph_edge_field(
            target_labels,
            fields,
            "target_labels",
        )?,
    })
}

fn set_graph_edge_field(
    target: &mut Option<syn::Ident>,
    field: &syn::Field,
    role: &str,
) -> syn::Result<()> {
    if target.is_some() {
        return Err(syn::Error::new_spanned(
            field,
            format!("GraphEdge requires exactly one #[graph({role})] field"),
        ));
    }

    *target = field.ident.clone();
    Ok(())
}

fn required_graph_edge_field(
    field: Option<syn::Ident>,
    fields: &syn::FieldsNamed,
    role: &str,
) -> syn::Result<syn::Ident> {
    field.ok_or_else(|| {
        syn::Error::new_spanned(
            fields,
            format!("GraphEdge requires exactly one #[graph({role})] field"),
        )
    })
}

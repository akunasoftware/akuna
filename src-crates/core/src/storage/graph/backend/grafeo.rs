use std::{collections::HashMap, path::PathBuf};

use crate::storage::graph::{
    GraphDbContext, GraphEdge, GraphError, GraphNode, GraphNodeSearchQuery,
    GraphNodeSearchResult, GraphTarget, GraphWriteOperation, search_text,
};
use grafeo::GrafeoDB;
use serde_json::Map;

const ENGINE_NAME: &str = "grafeo";
const NODE_ID_PROPERTY: &str = "_id";
const SEARCH_TEXT_PROPERTY: &str = "_search_text";
const SEARCH_EMBEDDING_PROPERTY: &str = "_search_embedding";

/// Grafeo-backed graph storage context.
pub(crate) struct GrafeoDbContext {
    session: grafeo::Session,
    /// Keep database handle alive for session-backed persistence.
    db: GrafeoDB,
}

impl GrafeoDbContext {
    /// Creates a Grafeo-backed graph database context rooted at the given path.
    pub fn new(persist_at: PathBuf) -> Result<Self, GraphError> {
        let db = GrafeoDB::open(&persist_at).map_err(|source| {
            GraphError::DbInit {
                engine: ENGINE_NAME,
                source: Box::new(source),
            }
        })?;

        Ok(Self {
            session: db.session(),
            db,
        })
    }

    /// Creates an in-memory Grafeo-backed graph database context.
    pub fn new_in_memory() -> Self {
        let db = GrafeoDB::new_in_memory();

        Self {
            session: db.session(),
            db,
        }
    }
}

impl GraphDbContext for GrafeoDbContext {
    fn put_node(
        &self,
        node: &GraphNode,
        search_embedding: &[f32],
    ) -> Result<(), GraphError> {
        let labels = node.labels.iter().map(String::as_str).collect::<Vec<_>>();
        let id = node.id.clone();
        self.ensure_search_indexes(&labels, search_embedding.len())?;

        let mut properties = match node.metadata.as_ref() {
            Some(metadata) => {
                let serde_json::Value::Object(properties) =
                    serde_json::to_value(metadata).map_err(|source| {
                        GraphError::Serialization {
                            source: Box::new(source),
                        }
                    })?
                else {
                    return Err(GraphError::InvalidNodeItem);
                };

                properties
            }
            None => Map::new(),
        };
        if properties.keys().any(|key| is_reserved_property(key)) {
            return Err(GraphError::InvalidNodeItem);
        }

        properties.insert(
            NODE_ID_PROPERTY.to_string(),
            serde_json::Value::String(id.clone()),
        );
        properties.insert(
            "name".to_string(),
            serde_json::Value::String(node.name.clone()),
        );
        properties.insert(
            "description".to_string(),
            node.description
                .clone()
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null),
        );
        properties.insert(
            SEARCH_TEXT_PROPERTY.to_string(),
            serde_json::Value::String(search_text(node)),
        );
        let assignments = properties
            .keys()
            .enumerate()
            .map(|(index, key)| {
                let safe_key = sanitize_property_key(key);
                format!("node.{safe_key} = $p{index}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mut params = HashMap::from([(
            "id".to_string(),
            grafeo::Value::from(id.clone()),
        )]);
        for (index, (_, value)) in properties.into_iter().enumerate() {
            params
                .insert(format!("p{}", index), convert_json_to_grafeo(value)?);
        }
        let embedding_param = format!("p{}", params.len());
        params.insert(
            embedding_param.clone(),
            grafeo::Value::from(search_embedding),
        );
        let assignments = format!(
            "{assignments}, node.{SEARCH_EMBEDDING_PROPERTY} = ${embedding_param}"
        );
        let query = format!(
            "MERGE (node:{} {{_id: $id}}) SET {}",
            compose_gql_labels(&labels),
            assignments,
        );

        self.session
            .execute_with_params(&query, params)
            .map_err(|source| GraphError::WriteFailed {
                engine: ENGINE_NAME,
                operation: GraphWriteOperation::Put,
                target: GraphTarget::Node {
                    labels: labels
                        .iter()
                        .map(|label| (*label).to_string())
                        .collect(),
                    id: id.clone(),
                },
                source: Box::new(source),
            })?;
        Ok(())
    }

    fn get_node(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Option<GraphNode>, GraphError> {
        let id: &str = id;

        let query = format!(
            "MATCH (node:{}) WHERE node._id = $id RETURN node",
            compose_gql_labels(labels),
        );
        let params =
            HashMap::from([("id".to_string(), grafeo::Value::from(id))]);
        let result = self.session.execute_with_params(&query, params).map_err(
            |source| GraphError::QueryExecution {
                engine: ENGINE_NAME,
                source: Box::new(source),
            },
        )?;

        let Some(value) = result.rows().first().and_then(|row| row.first())
        else {
            return Ok(None);
        };

        let grafeo::Value::Map(node_values) = value else {
            return Err(GraphError::InvalidNodeItem);
        };
        let mut values = node_values
            .iter()
            .filter(|(key, _)| !is_reserved_property(key.as_str()))
            .map(|(key, value)| {
                convert_grafeo_to_json(value)
                    .map(|value| (key.to_string(), value))
            })
            .collect::<Result<serde_json::Map<_, _>, _>>()?;
        let name = values
            .remove("name")
            .and_then(|value| value.as_str().map(ToString::to_string))
            .ok_or(GraphError::InvalidNodeItem)?;
        let description = values
            .remove("description")
            .and_then(|value| value.as_str().map(ToString::to_string));
        let metadata = if values.is_empty() {
            None
        } else {
            Some(
                serde_json::from_value(serde_json::Value::Object(values))
                    .map_err(|source| GraphError::Deserialization {
                        source: Box::new(source),
                    })?,
            )
        };

        Ok(Some(GraphNode {
            id: id.to_string(),
            labels: labels.iter().map(|label| (*label).to_string()).collect(),
            name,
            description,
            metadata,
        }))
    }

    fn delete_node(&self, labels: &[&str], id: &str) -> Result<(), GraphError> {
        let id: &str = id;

        let query = format!(
            "MATCH (node:{}) WHERE node._id = $id DELETE node RETURN node",
            compose_gql_labels(labels),
        );
        let params =
            HashMap::from([("id".to_string(), grafeo::Value::from(id))]);

        let result = self.session.execute_with_params(&query, params).map_err(
            |source| GraphError::WriteFailed {
                engine: ENGINE_NAME,
                operation: GraphWriteOperation::Delete,
                target: GraphTarget::Node {
                    labels: labels
                        .iter()
                        .map(|label| (*label).to_string())
                        .collect(),
                    id: id.to_string(),
                },
                source: Box::new(source),
            },
        )?;

        if result.is_empty() {
            return Err(GraphError::NotFound {
                target: GraphTarget::Node {
                    labels: labels
                        .iter()
                        .map(|label| (*label).to_string())
                        .collect(),
                    id: id.to_string(),
                },
            });
        }

        Ok(())
    }

    fn search_nodes(
        &self,
        query: &GraphNodeSearchQuery,
        query_embedding: &[f32],
    ) -> Result<Vec<GraphNodeSearchResult>, GraphError> {
        let labels = match query.label.as_deref() {
            Some(label) => vec![label.to_string()],
            None => match self.db.schema() {
                grafeo::admin::SchemaInfo::Lpg(schema) => {
                    schema.labels.into_iter().map(|label| label.name).collect()
                }
                _ => Vec::new(),
            },
        };

        let mut scores = HashMap::new();
        for label in labels {
            self.ensure_search_indexes(
                &[label.as_str()],
                query_embedding.len(),
            )?;
            let search_results = self
                .db
                .hybrid_search(
                    &label,
                    SEARCH_TEXT_PROPERTY,
                    SEARCH_EMBEDDING_PROPERTY,
                    &query.query,
                    Some(query_embedding),
                    query.limit,
                    None,
                )
                .map_err(|source| GraphError::QueryExecution {
                    engine: ENGINE_NAME,
                    source: Box::new(source),
                })?;

            for (node_id, score) in search_results {
                scores
                    .entry(node_id)
                    .and_modify(|current| {
                        if score > *current {
                            *current = score;
                        }
                    })
                    .or_insert(score);
            }
        }

        let mut results = Vec::new();
        for (node_id, score) in scores {
            // Skip nodes that vanished between search and fetch (concurrent
            // delete). Callers get partial results consistent with current
            // state instead of a stale error.
            let Some(node) = self.db.get_node(node_id) else {
                continue;
            };
            let node = {
                let labels = node
                    .labels
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                let mut values = node
                    .properties
                    .iter()
                    .filter(|(key, _)| !is_reserved_property(key.as_ref()))
                    .map(|(key, value)| {
                        convert_grafeo_to_json(value)
                            .map(|value| (key.to_string(), value))
                    })
                    .collect::<Result<serde_json::Map<_, _>, _>>()?;
                let id = node
                    .get_property(NODE_ID_PROPERTY)
                    .and_then(|value| match value {
                        grafeo::Value::String(value) => Some(value.to_string()),
                        _ => None,
                    })
                    .ok_or(GraphError::InvalidNodeItem)?;
                let name = values
                    .remove("name")
                    .and_then(|value| value.as_str().map(ToString::to_string))
                    .ok_or(GraphError::InvalidNodeItem)?;
                let description = values
                    .remove("description")
                    .and_then(|value| value.as_str().map(ToString::to_string));
                values.remove(NODE_ID_PROPERTY);
                let metadata = if values.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Object(values))
                };

                GraphNode {
                    id,
                    labels,
                    name,
                    description,
                    metadata,
                }
            };
            results.push(GraphNodeSearchResult { node, score });
        }
        results.sort_by(|left, right| right.score.total_cmp(&left.score));
        results.truncate(query.limit);

        Ok(results)
    }

    fn put_edge(&self, edge: &GraphEdge) -> Result<(), GraphError> {
        let predicate = validate_relationship_type(edge.predicate.as_str())?;
        let source_labels = edge
            .source_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let target_labels = edge
            .target_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let params = HashMap::from([
            (
                "source_id".to_string(),
                grafeo::Value::from(edge.source.as_str()),
            ),
            (
                "target_id".to_string(),
                grafeo::Value::from(edge.target.as_str()),
            ),
        ]);
        let exists_query = format!(
            "MATCH (source:{})-[edge:{}]->(target:{}) WHERE source._id = $source_id AND target._id = $target_id RETURN edge",
            compose_gql_labels(&source_labels),
            predicate,
            compose_gql_labels(&target_labels),
        );
        let exists = !self
            .session
            .execute_with_params(&exists_query, params.clone())
            .map_err(|source| GraphError::QueryExecution {
                engine: ENGINE_NAME,
                source: Box::new(source),
            })?
            .rows()
            .is_empty();

        if exists {
            return Ok(());
        }

        let query = format!(
            "MATCH (source:{}), (target:{}) WHERE source._id = $source_id AND target._id = $target_id INSERT (source)-[:{}]->(target)",
            compose_gql_labels(&source_labels),
            compose_gql_labels(&target_labels),
            predicate,
        );

        self.session
            .execute_with_params(&query, params)
            .map_err(|source| GraphError::WriteFailed {
                engine: ENGINE_NAME,
                operation: GraphWriteOperation::Put,
                target: GraphTarget::Edge {
                    predicate: edge.predicate.clone(),
                    source_id: edge.source.clone(),
                    target_id: edge.target.clone(),
                },
                source: Box::new(source),
            })?;

        Ok(())
    }

    fn delete_edge(&self, edge: &GraphEdge) -> Result<(), GraphError> {
        let predicate = validate_relationship_type(edge.predicate.as_str())?;
        let source_labels = edge
            .source_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let target_labels = edge
            .target_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let query = format!(
            "MATCH (source:{})-[edge:{}]->(target:{}) WHERE source._id = $source_id AND target._id = $target_id DELETE edge RETURN edge",
            compose_gql_labels(&source_labels),
            predicate,
            compose_gql_labels(&target_labels),
        );
        let params = HashMap::from([
            (
                "source_id".to_string(),
                grafeo::Value::from(edge.source.as_str()),
            ),
            (
                "target_id".to_string(),
                grafeo::Value::from(edge.target.as_str()),
            ),
        ]);

        let target = GraphTarget::Edge {
            predicate: edge.predicate.clone(),
            source_id: edge.source.clone(),
            target_id: edge.target.clone(),
        };

        let result = self.session.execute_with_params(&query, params).map_err(
            |source| GraphError::WriteFailed {
                engine: ENGINE_NAME,
                operation: GraphWriteOperation::Delete,
                target: target.clone(),
                source: Box::new(source),
            },
        )?;

        if result.is_empty() {
            return Err(GraphError::NotFound { target });
        }

        Ok(())
    }
}

impl GrafeoDbContext {
    /// Ensures search indexes exist for each label.
    fn ensure_search_indexes(
        &self,
        labels: &[&str],
        dimensions: usize,
    ) -> Result<(), GraphError> {
        for label in labels {
            self.db
                .create_text_index(label, SEARCH_TEXT_PROPERTY)
                .map_err(|source| GraphError::QueryExecution {
                    engine: ENGINE_NAME,
                    source: Box::new(source),
                })?;
            self.db
                .create_vector_index(
                    label,
                    SEARCH_EMBEDDING_PROPERTY,
                    Some(dimensions),
                    Some("cosine"),
                    None,
                    None,
                    None,
                )
                .map_err(|source| GraphError::QueryExecution {
                    engine: ENGINE_NAME,
                    source: Box::new(source),
                })?;
        }

        Ok(())
    }
}

/// Joins graph labels into a cypher-compatible label expression.
fn compose_gql_labels(labels: &[&str]) -> String {
    labels
        .iter()
        .map(|label| sanitize_property_key(label))
        .collect::<Vec<_>>()
        .join(":")
}

/// Sanitizes a string for safe interpolation into a Cypher property key or label.
fn sanitize_property_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Validates a Cypher relationship type before query interpolation.
fn validate_relationship_type(predicate: &str) -> Result<&str, GraphError> {
    let mut chars = predicate.chars();
    let Some(first) = chars.next() else {
        return Err(GraphError::InvalidEdgePredicate {
            predicate: predicate.to_owned(),
        });
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(GraphError::InvalidEdgePredicate {
            predicate: predicate.to_owned(),
        });
    }
    if chars.any(|character| {
        !(character.is_ascii_alphanumeric() || character == '_')
    }) {
        return Err(GraphError::InvalidEdgePredicate {
            predicate: predicate.to_owned(),
        });
    }

    Ok(predicate)
}

fn is_reserved_property(key: &str) -> bool {
    key.starts_with('_')
}

fn convert_json_to_grafeo(
    value: serde_json::Value,
) -> Result<grafeo::Value, GraphError> {
    Ok(match value {
        serde_json::Value::Null => grafeo::Value::Null,
        serde_json::Value::Bool(value) => grafeo::Value::from(value),
        serde_json::Value::Number(value) => value
            .as_i64()
            .map(grafeo::Value::from)
            .or_else(|| value.as_f64().map(grafeo::Value::from))
            .ok_or(GraphError::InvalidNodeItem)?,
        serde_json::Value::String(value) => grafeo::Value::from(value),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            return Err(GraphError::InvalidNodeItem);
        }
    })
}

fn convert_grafeo_to_json(
    value: &grafeo::Value,
) -> Result<serde_json::Value, GraphError> {
    Ok(match value {
        grafeo::Value::Null => serde_json::Value::Null,
        grafeo::Value::Bool(value) => serde_json::Value::Bool(*value),
        grafeo::Value::Int64(value) => {
            serde_json::Value::Number((*value).into())
        }
        grafeo::Value::Float64(value) => serde_json::Number::from_f64(*value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        grafeo::Value::String(value) => {
            serde_json::Value::String(value.to_string())
        }
        grafeo::Value::Map(values) => serde_json::Value::Object(
            values
                .iter()
                .map(|(key, value)| {
                    convert_grafeo_to_json(value)
                        .map(|value| (key.to_string(), value))
                })
                .collect::<Result<Map<_, _>, _>>()?,
        ),
        _ => return Err(GraphError::InvalidNodeItem),
    })
}

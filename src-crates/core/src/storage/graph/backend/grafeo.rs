use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use crate::storage::graph::{
    GraphDbContext, GraphEdge, GraphError, GraphNode, GraphTarget,
    GraphWriteOperation,
};
use grafeo::GrafeoDB;
use serde_json::Map;

const ENGINE_NAME: &str = "grafeo";
const NODE_ID_PROPERTY: &str = "_id";
const METADATA_KEYS_PROPERTY: &str = "_metadata_keys";

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
    fn put_node(&self, node: &GraphNode) -> Result<(), GraphError> {
        let labels = node.labels.iter().map(String::as_str).collect::<Vec<_>>();
        validate_labels(&labels)?;
        let id = node.id.clone();

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
        for key in properties.keys() {
            validate_graph_identifier(key)?;
        }
        let mut metadata_keys = properties.keys().cloned().collect::<Vec<_>>();
        metadata_keys.sort();

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
            METADATA_KEYS_PROPERTY.to_string(),
            serde_json::Value::String(metadata_keys.join(",")),
        );
        let property_keys = properties.keys().cloned().collect::<HashSet<_>>();
        let mut assignments = properties
            .keys()
            .enumerate()
            .map(|(index, key)| format!("node.{key} = $p{index}"))
            .collect::<Vec<_>>();
        let mut params = HashMap::from([(
            "id".to_string(),
            grafeo::Value::from(id.clone()),
        )]);
        for (index, (_, value)) in properties.into_iter().enumerate() {
            params
                .insert(format!("p{}", index), convert_json_to_grafeo(value)?);
        }
        if let Some(node_id) = self.find_node_id(&labels, &id)
            && let Some(node) = self.db.get_node(node_id)
        {
            let mut stale_keys = node
                .properties
                .iter()
                .filter_map(|(key, _)| {
                    let key = key.as_ref();
                    (!is_internal_property(key) && !property_keys.contains(key))
                        .then(|| key.to_string())
                })
                .collect::<Vec<_>>();
            stale_keys.sort();
            for key in stale_keys {
                validate_graph_identifier(&key)?;
                let index = params.len() - 1;
                assignments.push(format!("node.{key} = $p{index}"));
                params.insert(format!("p{index}"), grafeo::Value::Null);
            }
        }
        let query = format!(
            "MERGE (node:{} {{_id: $id}}) SET {}",
            compose_gql_labels(&labels),
            assignments.join(", "),
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
        validate_labels(labels)?;
        let Some(node_id) = self.find_node_id(labels, id) else {
            return Ok(None);
        };

        self.graph_node(node_id)
    }

    fn delete_node(&self, labels: &[&str], id: &str) -> Result<(), GraphError> {
        validate_labels(labels)?;

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

    fn put_edge(&self, edge: &GraphEdge) -> Result<(), GraphError> {
        let predicate = validate_relationship_type(edge.predicate.as_str())?;
        let source_labels = edge
            .source_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        validate_labels(&source_labels)?;
        let target_labels = edge
            .target_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        validate_labels(&target_labels)?;
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

        let target = GraphTarget::Edge {
            predicate: edge.predicate.clone(),
            source_id: edge.source.clone(),
            target_id: edge.target.clone(),
        };
        let query = format!(
            "MATCH (source:{}), (target:{}) WHERE source._id = $source_id AND target._id = $target_id INSERT (source)-[:{}]->(target)",
            compose_gql_labels(&source_labels),
            compose_gql_labels(&target_labels),
            predicate,
        );

        self.session
            .execute_with_params(&query, params.clone())
            .map_err(|source| GraphError::WriteFailed {
                engine: ENGINE_NAME,
                operation: GraphWriteOperation::Put,
                target: target.clone(),
                source: Box::new(source),
            })?;
        let inserted = !self
            .session
            .execute_with_params(&exists_query, params)
            .map_err(|source| GraphError::QueryExecution {
                engine: ENGINE_NAME,
                source: Box::new(source),
            })?
            .rows()
            .is_empty();
        if !inserted {
            return Err(GraphError::NotFound { target });
        }

        Ok(())
    }

    fn delete_edge(&self, edge: &GraphEdge) -> Result<(), GraphError> {
        let predicate = validate_relationship_type(edge.predicate.as_str())?;
        let source_labels = edge
            .source_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        validate_labels(&source_labels)?;
        let target_labels = edge
            .target_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        validate_labels(&target_labels)?;
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

    fn neighbors(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Vec<(GraphEdge, GraphNode)>, GraphError> {
        validate_labels(labels)?;
        let Some(node_id) = self.find_node_id(labels, id) else {
            return Ok(Vec::new());
        };

        let mut edge_ids = HashSet::new();
        let edges = self
            .session
            .get_neighbors_outgoing(node_id)
            .into_iter()
            .chain(self.session.get_neighbors_incoming(node_id));
        let mut neighbors = Vec::new();

        for (_, edge_id) in edges {
            if !edge_ids.insert(edge_id) {
                continue;
            }
            let Some(edge) = self.session.get_edge(edge_id) else {
                continue;
            };
            let Some(source) = self.graph_node(edge.src)? else {
                continue;
            };
            let Some(target) = self.graph_node(edge.dst)? else {
                continue;
            };

            let graph_edge = GraphEdge {
                source_labels: source.labels.clone(),
                source: source.id.clone(),
                predicate: edge.edge_type.to_string(),
                target: target.id.clone(),
                target_labels: target.labels.clone(),
            };
            let neighbor = if edge.src == node_id { target } else { source };
            neighbors.push((graph_edge, neighbor));
        }
        neighbors.sort_by(
            |(left_edge, left_node), (right_edge, right_node)| {
                left_edge
                    .source_labels
                    .cmp(&right_edge.source_labels)
                    .then_with(|| left_edge.source.cmp(&right_edge.source))
                    .then_with(|| {
                        left_edge.predicate.cmp(&right_edge.predicate)
                    })
                    .then_with(|| left_edge.target.cmp(&right_edge.target))
                    .then_with(|| {
                        left_edge.target_labels.cmp(&right_edge.target_labels)
                    })
                    .then_with(|| left_node.labels.cmp(&right_node.labels))
                    .then_with(|| left_node.id.cmp(&right_node.id))
            },
        );

        Ok(neighbors)
    }
}

impl GrafeoDbContext {
    /// Finds a stored node's internal ID.
    fn find_node_id(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Option<grafeo::NodeId> {
        let id = grafeo::Value::from(id);
        self.db
            .find_nodes_by_property(NODE_ID_PROPERTY, &id)
            .into_iter()
            .find(|node_id| {
                self.db.get_node(*node_id).is_some_and(|node| {
                    node.labels.len() == labels.len()
                        && labels.iter().all(|label| node.has_label(label))
                })
            })
    }

    /// Converts a stored graph node.
    fn graph_node(
        &self,
        node_id: grafeo::NodeId,
    ) -> Result<Option<GraphNode>, GraphError> {
        let Some(node) = self.db.get_node(node_id) else {
            return Ok(None);
        };

        let labels = node
            .labels
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let metadata_keys =
            node.get_property(METADATA_KEYS_PROPERTY).and_then(|value| {
                match value {
                    grafeo::Value::String(value) => Some(
                        value
                            .split(',')
                            .filter(|key| !key.is_empty())
                            .collect::<HashSet<_>>(),
                    ),
                    _ => None,
                }
            });
        let mut values = node
            .properties
            .iter()
            .filter(|(key, _)| !is_internal_property(key.as_ref()))
            .filter(|(key, value)| {
                matches!(key.as_ref(), "name" | "description")
                    || metadata_keys
                        .as_ref()
                        .is_some_and(|keys| keys.contains(key.as_ref()))
                    || (metadata_keys.is_none()
                        && !matches!(value, grafeo::Value::Null))
            })
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
        let metadata = if values.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(values))
        };

        Ok(Some(GraphNode {
            id,
            labels,
            name,
            description,
            metadata,
        }))
    }
}

/// Joins graph labels into a cypher-compatible label expression.
fn compose_gql_labels(labels: &[&str]) -> String {
    labels.join(":")
}

/// Validates graph labels used in query interpolation.
fn validate_labels(labels: &[&str]) -> Result<(), GraphError> {
    if labels.is_empty() {
        return Err(GraphError::InvalidNodeItem);
    }
    for label in labels {
        validate_graph_identifier(label)?;
    }

    Ok(())
}

/// Validates a graph identifier used in query interpolation.
fn validate_graph_identifier(identifier: &str) -> Result<&str, GraphError> {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        return Err(GraphError::InvalidNodeItem);
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(GraphError::InvalidNodeItem);
    }
    if chars.any(|character| {
        !(character.is_ascii_alphanumeric() || character == '_')
    }) {
        return Err(GraphError::InvalidNodeItem);
    }

    Ok(identifier)
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
    is_internal_property(key) || matches!(key, "name" | "description")
}

fn is_internal_property(key: &str) -> bool {
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

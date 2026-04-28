use std::{collections::HashMap, fs};

use grafeo::GrafeoDB;
use serde_json::Map;

use crate::{
    GraphError, GraphTarget, GraphWriteOperation,
    dirs::{AppDirType, get_app_dir},
    graph::{
        primitives::{GraphDbContext, GraphEdge, GraphNode},
        storage::GraphStorage,
    },
};

const ENGINE_NAME: &str = "grafeo";
const NODE_ID_PROPERTY: &str = "_id";

/// Grafeo-backed graph storage context.
pub struct GrafeoDbContext {
    session: grafeo::Session,
    /// Storage mode backing this graph context.
    storage: GraphStorage,
    /// Keep database handle alive for session-backed persistence.
    _db: GrafeoDB,
}

impl GrafeoDbContext {
    /// Creates a Grafeo-backed graph database context.
    pub fn new(name: String) -> Result<Self, GraphError> {
        let persist_at = get_app_dir(AppDirType::Data).join(name);
        let db = GrafeoDB::open(&persist_at).map_err(|source| {
            GraphError::DbInit {
                engine: ENGINE_NAME,
                source: Box::new(source),
            }
        })?;

        Ok(Self {
            session: db.session(),
            storage: GraphStorage::Persistent(persist_at),
            _db: db,
        })
    }

    /// Creates an in-memory Grafeo-backed graph database context.
    pub fn new_in_memory() -> Self {
        let db = GrafeoDB::new_in_memory();

        Self {
            session: db.session(),
            storage: GraphStorage::InMemory,
            _db: db,
        }
    }

    #[cfg(test)]
    pub(super) fn node_count_by_id(
        &self,
        labels: &[&str],
        id: impl AsRef<str>,
    ) -> Result<usize, GraphError> {
        let params = HashMap::from([(
            "id".to_string(),
            grafeo::Value::from(id.as_ref()),
        )]);
        let result = self
            .session
            .execute_with_params(
                &format!(
                    "MATCH (node:{}) WHERE node._id = $id RETURN node",
                    compose_gql_labels(labels),
                ),
                params,
            )
            .map_err(|source| GraphError::QueryExecution {
                engine: ENGINE_NAME,
                source: Box::new(source),
            })?;

        Ok(result.rows().len())
    }
}

impl GraphDbContext for GrafeoDbContext {
    fn storage(&self) -> &GraphStorage {
        &self.storage
    }

    fn put_node<T>(&self, node: &T) -> Result<(), GraphError>
    where
        T: GraphNode + ?Sized,
    {
        let labels = node.labels();
        let id = node.id().to_string();
        let mut properties = match node.metadata() {
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

        properties.insert(
            NODE_ID_PROPERTY.to_string(),
            serde_json::Value::String(id.clone()),
        );
        properties.insert(
            "name".to_string(),
            serde_json::Value::String(node.name().to_string()),
        );
        properties.insert(
            "description".to_string(),
            node.description()
                .map(|description| {
                    serde_json::Value::String(description.to_string())
                })
                .unwrap_or(serde_json::Value::Null),
        );
        let assignments = properties
            .keys()
            .enumerate()
            .map(|(index, key)| format!("node.{} = $p{}", key, index))
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

    fn get_node<T>(
        &self,
        labels: &[&str],
        id: impl AsRef<str>,
    ) -> Result<Option<T>, GraphError>
    where
        T: GraphNode,
    {
        let id = id.as_ref();

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

        let serde_json::Value::Object(mut values) =
            convert_grafeo_to_json(value)?
        else {
            return Err(GraphError::InvalidNodeItem);
        };
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
            Some(
                serde_json::from_value(serde_json::Value::Object(values))
                    .map_err(|source| GraphError::Deserialization {
                        source: Box::new(source),
                    })?,
            )
        };

        Ok(Some(T::from_graph_parts(
            id.to_string(),
            labels.iter().map(|label| (*label).to_string()).collect(),
            name,
            description,
            metadata,
        )))
    }

    fn delete_node<T>(
        &self,
        labels: &[&str],
        id: impl AsRef<str>,
    ) -> Result<(), GraphError>
    where
        T: GraphNode,
    {
        let id = id.as_ref();

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

    fn put_edge<T>(&self, edge: &T) -> Result<(), GraphError>
    where
        T: GraphEdge + ?Sized,
    {
        let predicate = edge.predicate();
        let source_labels = edge.source_labels();
        let target_labels = edge.target_labels();
        let params = HashMap::from([
            ("source_id".to_string(), grafeo::Value::from(edge.source())),
            ("target_id".to_string(), grafeo::Value::from(edge.target())),
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
                    predicate: edge.predicate().to_string(),
                    source_id: edge.source().to_string(),
                    target_id: edge.target().to_string(),
                },
                source: Box::new(source),
            })?;

        Ok(())
    }

    fn delete_edge<T>(&self, edge: &T) -> Result<(), GraphError>
    where
        T: GraphEdge + ?Sized,
    {
        let predicate = edge.predicate();
        let source_labels = edge.source_labels();
        let target_labels = edge.target_labels();
        let query = format!(
            "MATCH (source:{})-[edge:{}]->(target:{}) WHERE source._id = $source_id AND target._id = $target_id DELETE edge RETURN edge",
            compose_gql_labels(&source_labels),
            predicate,
            compose_gql_labels(&target_labels),
        );
        let params = HashMap::from([
            ("source_id".to_string(), grafeo::Value::from(edge.source())),
            ("target_id".to_string(), grafeo::Value::from(edge.target())),
        ]);

        let target = GraphTarget::Edge {
            predicate: edge.predicate().to_string(),
            source_id: edge.source().to_string(),
            target_id: edge.target().to_string(),
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

    fn destroy(self) -> Result<(), GraphError> {
        self._db
            .close()
            .map_err(|source| GraphError::GraphDestroy {
                engine: ENGINE_NAME,
                source: Box::new(source),
            })?;

        let GraphStorage::Persistent(storage_path) = self.storage else {
            return Ok(());
        };

        if !storage_path.exists() {
            return Ok(());
        }

        fs::remove_dir_all(storage_path).map_err(|source| {
            GraphError::GraphDestroy {
                engine: ENGINE_NAME,
                source: Box::new(source),
            }
        })
    }
}

fn compose_gql_labels(labels: &[&str]) -> String {
    labels.join(":")
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
                .filter(|(key, _)| !key.as_str().starts_with('_'))
                .map(|(key, value)| {
                    convert_grafeo_to_json(value)
                        .map(|value| (key.to_string(), value))
                })
                .collect::<Result<Map<_, _>, _>>()?,
        ),
        _ => return Err(GraphError::InvalidNodeItem),
    })
}

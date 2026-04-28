use std::path::PathBuf;

/// Grafeo graph storage backend.
pub mod grafeo;

/// Backend-neutral graph storage mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphStorage {
    /// Data is kept in memory only.
    InMemory,
    /// Data is persisted at the given path.
    Persistent(PathBuf),
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::SystemTime};

    use serde::{Deserialize, Serialize};
    use tokio::task::JoinSet;

    use crate::{
        GraphError, GraphTarget,
        graph::{
            primitives::{GraphDbContext, GraphEdge, GraphNode},
            storage::{GraphStorage, grafeo::GrafeoDbContext},
        },
    };

    #[derive(Clone, Debug, GraphNode, PartialEq)]
    struct TestNode {
        #[graph(id)]
        id: String,
        #[graph(labels)]
        labels: Vec<String>,
        #[graph(name)]
        name: String,
        #[graph(description)]
        description: Option<String>,
        #[graph(metadata)]
        metadata: Option<TestMetadata>,
    }

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    struct TestMetadata {
        val: String,
    }

    #[derive(Clone, Debug, GraphEdge, PartialEq, Eq)]
    struct TestEdge {
        #[graph(source_labels)]
        source_labels: Vec<String>,
        #[graph(source)]
        source: String,
        #[graph(predicate)]
        predicate: String,
        #[graph(target)]
        target: String,
        #[graph(target_labels)]
        target_labels: Vec<String>,
    }

    const EXISTING_COUNT: usize = 20;
    const TOTAL_COUNT: usize = 30;
    const DELETE_EVERY: usize = 5;
    const UPSERT_EVERY: usize = 3;

    macro_rules! graph_db_impls {
        ($tests:ident) => {
            $tests!(
                grafeo,
                GrafeoDbContext::new_in_memory,
                GrafeoDbContext::new
            );
        };
    }

    macro_rules! graph_db_tests {
        ($name:ident, $new_in_memory:path, $new_persistent:path) => {
            mod $name {
                use super::*;

                /// Typed graph storage supports node and edge CRUD.
                #[tokio::test]
                async fn crud_lifecycle() {
                    run_crud_lifecycle($new_in_memory());

                    let db_name = db_name();
                    let graph_db = $new_persistent(db_name)
                        .expect("Failed to initialize persisted graph db");

                    run_crud_lifecycle(graph_db);
                }

                /// Node ids are unique within labels.
                #[tokio::test]
                async fn ids_are_label_scoped() {
                    let graph_db = $new_in_memory();
                    let first = TestNode {
                        id: "same".to_string(),
                        labels: vec!["Concept".to_string(), "Old".to_string()],
                        name: "old".to_string(),
                        description: None,
                        metadata: Some(TestMetadata {
                            val: "old".to_string(),
                        }),
                    };
                    let second = TestNode {
                        id: first.id.clone(),
                        labels: vec!["Concept".to_string(), "New".to_string()],
                        name: "new".to_string(),
                        description: None,
                        metadata: Some(TestMetadata {
                            val: "new".to_string(),
                        }),
                    };

                    graph_db
                        .put_node(&first)
                        .expect("Failed to put first node");
                    assert_eq!(
                        graph_db
                            .node_count_by_id(&["Concept", "Old"], &first.id)
                            .expect("Failed to count first node"),
                        1,
                    );

                    graph_db.put_node(&second).expect("Failed to upsert node");
                    assert_eq!(
                        graph_db
                            .node_count_by_id(&["Concept", "Old"], &second.id)
                            .expect("Failed to count original label node"),
                        1,
                    );
                    assert_eq!(
                        graph_db
                            .node_count_by_id(&["Concept", "New"], &second.id)
                            .expect("Failed to count upserted node"),
                        1,
                    );

                    let retrieved = graph_db
                        .get_node::<TestNode>(&["Concept", "New"], &second.id)
                        .expect("Failed to read upserted node");

                    assert_eq!(retrieved, Some(second));
                }

                /// Edge endpoints are scoped by node labels.
                #[tokio::test]
                async fn edges_are_label_scoped() {
                    let graph_db = $new_in_memory();
                    let old_source = TestNode {
                        id: "source".to_string(),
                        labels: vec!["Concept".to_string(), "Old".to_string()],
                        name: "old source".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let new_source = TestNode {
                        id: old_source.id.clone(),
                        labels: vec!["Concept".to_string(), "New".to_string()],
                        name: "new source".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let old_target = TestNode {
                        id: "target".to_string(),
                        labels: vec!["Concept".to_string(), "Old".to_string()],
                        name: "old target".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let new_target = TestNode {
                        id: old_target.id.clone(),
                        labels: vec!["Concept".to_string(), "New".to_string()],
                        name: "new target".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let edge = TestEdge {
                        source_labels: new_source.labels.clone(),
                        source: new_source.id.clone(),
                        predicate: "RELATES_TO".to_string(),
                        target: new_target.id.clone(),
                        target_labels: new_target.labels.clone(),
                    };
                    let old_edge = TestEdge {
                        source_labels: old_source.labels.clone(),
                        source: old_source.id.clone(),
                        predicate: edge.predicate.clone(),
                        target: old_target.id.clone(),
                        target_labels: old_target.labels.clone(),
                    };

                    graph_db
                        .put_node(&old_source)
                        .expect("Failed to put old source");
                    graph_db
                        .put_node(&new_source)
                        .expect("Failed to put new source");
                    graph_db
                        .put_node(&old_target)
                        .expect("Failed to put old target");
                    graph_db
                        .put_node(&new_target)
                        .expect("Failed to put new target");

                    graph_db
                        .put_edge(&edge)
                        .expect("Failed to put scoped edge");

                    assert!(matches!(
                        graph_db.delete_edge(&old_edge),
                        Err(GraphError::NotFound { .. })
                    ));

                    graph_db
                        .delete_edge(&edge)
                        .expect("Failed to delete scoped edge");
                }

                /// Persistent graph nodes survive context reinitialization.
                #[tokio::test]
                async fn persists_and_destroys() {
                    let db_name = db_name();
                    let item = TestNode {
                        id: "123".to_string(),
                        labels: labels_vec(),
                        name: "persisted".to_string(),
                        description: Some("persisted-val".to_string()),
                        metadata: Some(TestMetadata {
                            val: "persisted-extra".to_string(),
                        }),
                    };

                    {
                        let graph_db = $new_persistent(db_name.clone())
                            .expect("Failed to initialize persisted graph db");

                        assert!(matches!(
                            graph_db.storage(),
                            GraphStorage::Persistent(_)
                        ));

                        graph_db
                            .put_node(&item)
                            .expect("Failed to insert persisted entry");
                    }

                    let graph_db = $new_persistent(db_name.clone())
                        .expect("Failed to reinitialize persisted graph db");
                    let retrieved = graph_db
                        .get_node::<TestNode>(labels(), &item.id)
                        .expect("Failed to read persisted entry");

                    assert_eq!(retrieved, Some(item));

                    graph_db
                        .destroy()
                        .expect("Failed to clean persisted test db");

                    let graph_db = $new_persistent(db_name)
                        .expect("Failed to reinitialize destroyed graph db");
                    let retrieved = graph_db
                        .get_node::<TestNode>(labels(), "123")
                        .expect("Failed to read from destroyed graph db");

                    assert_eq!(retrieved, None);

                    graph_db.destroy().expect(
                        "Failed to clean reinitialized destroyed graph db",
                    );
                }

                /// One shared context can handle concurrent mixed workloads safely.
                #[tokio::test]
                async fn concurrency_single_ctx() {
                    let graph_db = Arc::new($new_in_memory());
                    let mut work = JoinSet::new();

                    for index in 0..EXISTING_COUNT {
                        let graph_db = Arc::clone(&graph_db);

                        work.spawn(async move {
                            let item = TestNode {
                                id: format!("item-{index}"),
                                labels: labels_vec(),
                                name: format!("item-{index}"),
                                description: Some(format!("val-{index}")),
                                metadata: Some(TestMetadata {
                                    val: format!("extra-{index}"),
                                }),
                            };

                            graph_db.put_node(&item)
                        });
                    }

                    finish(work).await;

                    let mut work = JoinSet::new();

                    for index in 0..TOTAL_COUNT {
                        let graph_db = Arc::clone(&graph_db);

                        work.spawn(async move {
                            let id = format!("item-{index}");

                            if index < EXISTING_COUNT
                                && index % DELETE_EVERY == 0
                            {
                                return graph_db
                                    .delete_node::<TestNode>(labels(), &id)
                                    .map(|_| ());
                            }

                            let item = expected_node(index)
                                .expect("item should be present");

                            graph_db.put_node(&item).map(|_| ())
                        });
                    }

                    finish(work).await;

                    for index in 0..TOTAL_COUNT {
                        let id = format!("item-{index}");
                        let item = graph_db
                            .get_node::<TestNode>(labels(), &id)
                            .expect(
                                "Failed to read concurrently inserted entry",
                            );

                        assert_eq!(item, expected_node(index));
                    }
                }

                /// Many contexts in one process can handle concurrent mixed workloads on one storage path.
                #[tokio::test]
                async fn concurrency_multi_ctx() {
                    let db_name = db_name();
                    let mut work = JoinSet::new();

                    for index in 0..EXISTING_COUNT {
                        let db_name = db_name.clone();

                        work.spawn(async move {
                            let graph_db = $new_persistent(db_name)?;
                            let item = TestNode {
                                id: format!("item-{index}"),
                                labels: labels_vec(),
                                name: format!("item-{index}"),
                                description: Some(format!("val-{index}")),
                                metadata: Some(TestMetadata {
                                    val: format!("extra-{index}"),
                                }),
                            };

                            graph_db.put_node(&item)
                        });
                    }

                    finish(work).await;

                    let mut work = JoinSet::new();

                    for index in 0..TOTAL_COUNT {
                        let db_name = db_name.clone();

                        work.spawn(async move {
                            let graph_db = $new_persistent(db_name)?;
                            let id = format!("item-{index}");

                            if index < EXISTING_COUNT
                                && index % DELETE_EVERY == 0
                            {
                                return graph_db
                                    .delete_node::<TestNode>(labels(), &id)
                                    .map(|_| ());
                            }

                            let item = expected_node(index)
                                .expect("item should be present");

                            graph_db.put_node(&item).map(|_| ())
                        });
                    }

                    finish(work).await;

                    let graph_db = $new_persistent(db_name)
                        .expect("Failed to open verification graph db");

                    for index in 0..TOTAL_COUNT {
                        let id = format!("item-{index}");
                        let item = graph_db
                            .get_node::<TestNode>(labels(), &id)
                            .expect(
                                "Failed to read concurrently inserted entry",
                            );

                        assert_eq!(item, expected_node(index));
                    }

                    graph_db.destroy().expect("Failed to clean test db");
                }
            }
        };
    }

    graph_db_impls!(graph_db_tests);

    fn run_crud_lifecycle(graph_db: impl GraphDbContext) {
        let item = TestNode {
            id: "123".to_string(),
            labels: labels_vec(),
            name: "some".to_string(),
            description: Some("some-val".to_string()),
            metadata: Some(TestMetadata {
                val: "some-extra".to_string(),
            }),
        };

        graph_db.put_node(&item).expect("Failed to put entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &item.id)
            .expect("Failed to read entry");

        assert_eq!(retrieved, Some(item.clone()));

        let updated = TestNode {
            id: item.id.clone(),
            labels: labels_vec(),
            name: "updated".to_string(),
            description: Some("updated-val".to_string()),
            metadata: Some(TestMetadata {
                val: "updated-extra".to_string(),
            }),
        };

        graph_db.put_node(&updated).expect("Failed to update entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &updated.id)
            .expect("Failed to read updated entry");

        assert_eq!(retrieved, Some(updated.clone()));

        let inserted = TestNode {
            id: "456".to_string(),
            labels: labels_vec(),
            name: "inserted".to_string(),
            description: Some("inserted-val".to_string()),
            metadata: Some(TestMetadata {
                val: "inserted-extra".to_string(),
            }),
        };
        graph_db
            .put_node(&inserted)
            .expect("Failed to put missing entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &inserted.id)
            .expect("Failed to read inserted entry");

        assert_eq!(retrieved, Some(inserted.clone()));

        let edge = TestEdge {
            source_labels: updated.labels.clone(),
            source: updated.id.clone(),
            predicate: "RELATES_TO".to_string(),
            target: inserted.id.clone(),
            target_labels: inserted.labels.clone(),
        };

        graph_db.put_edge(&edge).expect("Failed to put edge");
        graph_db.delete_edge(&edge).expect("Failed to delete edge");

        let missing_edge_delete = graph_db.delete_edge(&edge);

        assert!(matches!(
            missing_edge_delete,
            Err(GraphError::NotFound {
                target: GraphTarget::Edge { predicate, .. }
            }) if predicate == edge.predicate()
        ));

        graph_db
            .delete_node::<TestNode>(labels(), &inserted.id)
            .expect("Failed to delete inserted entry");

        graph_db
            .delete_node::<TestNode>(labels(), &updated.id)
            .expect("Failed to delete entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &updated.id)
            .expect("Failed to read deleted entry");

        assert_eq!(retrieved, None);

        let missing_delete =
            graph_db.delete_node::<TestNode>(labels(), &updated.id);

        assert!(matches!(
            missing_delete,
            Err(GraphError::NotFound {
                target: GraphTarget::Node { id, .. }
            }) if id == updated.id
        ));

        graph_db.destroy().expect("Failed to clean test db");
    }

    async fn finish(mut work: JoinSet<Result<(), GraphError>>) {
        while let Some(result) = work.join_next().await {
            result
                .expect("Concurrent task panicked")
                .expect("Concurrent task failed");
        }
    }

    fn expected_node(index: usize) -> Option<TestNode> {
        if index < EXISTING_COUNT && index.is_multiple_of(DELETE_EVERY) {
            return None;
        }

        let description = if index >= EXISTING_COUNT {
            format!("inserted-{index}")
        } else if index.is_multiple_of(UPSERT_EVERY) {
            format!("upserted-{index}")
        } else {
            format!("updated-{index}")
        };

        Some(TestNode {
            id: format!("item-{index}"),
            labels: labels_vec(),
            name: format!("item-{index}"),
            description: Some(description),
            metadata: Some(TestMetadata {
                val: format!("extra-{index}"),
            }),
        })
    }

    fn labels() -> &'static [&'static str] {
        &["Concept", "Item"]
    }

    fn labels_vec() -> Vec<String> {
        labels().iter().map(|label| (*label).to_string()).collect()
    }

    fn db_name() -> String {
        format!(
            "graphdb-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("System clock should be after Unix epoch")
                .as_nanos(),
        )
    }
}

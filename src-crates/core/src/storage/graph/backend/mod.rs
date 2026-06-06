//! Graph storage backend implementations.
//!
//! Each backend provides a concrete context that satisfies the
//! [`crate::storage::graph::GraphDbContext`] trait. Modules here are crate-private;
//! callers obtain contexts via [`crate::storage::graph::open_context`] or
//! [`crate::storage::graph::in_memory_context`].

mod grafeo;

pub(crate) use grafeo::GrafeoDbContext;

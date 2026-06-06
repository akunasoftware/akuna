# akuna

Binary crate exposing the `akuna` command-line interface for knowledge work.
Wires together file extraction, schema generation, and a local REST API server
on top of the shared `akuna-core` workspace crate.

## Usage

```text
akuna --help
akuna extract ./notes.md --metadata --content
akuna schemas generate
akuna serve
```

See the [workspace README](../../README.md) for project overview.

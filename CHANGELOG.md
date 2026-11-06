# Changelog

All notable changes to `vector-mcp` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.3.8] - 2026-11-06

### Changed
- Replaced the action-routed `vector_collection_management` and `vector_search` tools (and the local action splitter) with seven typed tools: `vector_create_collection`, `vector_add_documents`, `vector_delete_collection`, `vector_list_collections`, `vector_semantic_search`, `vector_lexical_search`, `vector_search`. Each tool now takes only its own arguments.
- Declared tool-use contracts, links, data locations and business roles; replaced the two non-agent prompts with an operator prompt and two business-role prompts; raised `scion-sdk` to `>=0.9.0` (including the chromadb extra).

## [3.3.7] - 2026-10-30

### Changed
- Require `scion-sdk>=0.6.4,<0.7.0`; `uv.lock` relocked.

## [3.3.4] - 2026-10-26

### Added
- Central spec-delivery Pages dashboard and the `vector-mcp-development` skill.

## [1.14.0] - 2026-05-22

### Added
- Initial CHANGELOG.md creation
- docs/concepts.md with CONCEPT ID registry

### Changed
- Standardized project structure per agent-packages ecosystem conventions

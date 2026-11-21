"""CONN-VECTOR-MCP-CONNECTOR-CONTENT-R001: the pack extends only the core ontology (no redeclared core term, no old IRI base)."""

from __future__ import annotations

from pathlib import Path

import pytest

rdflib = pytest.importorskip("rdflib")
from rdflib.namespace import OWL, RDF, RDFS  # noqa: E402

pytestmark = pytest.mark.spec("CONN-VECTOR-MCP-CONNECTOR-CONTENT-R001")

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "vector_mcp" / "ontology"
CONNECTOR_NS = "/ontology/connector/"
# Terms the core ontology already declares; the pack references them and never redeclares them.
CORE_OWNED = [
    "Chunk",
    "Embedding",
    "SearchHit",
    "SearchQuery",
    "VectorCollection",
    "dimensions",
    "distanceMetric",
    "embeddingModel",
    "hostedOn",
    "storedIn",
]
_KINDS = (
    RDFS.Class,
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    RDF.Property,
)


def _graphs():
    for path in sorted(ONTOLOGY_DIR.glob("*.ttl")):
        graph = rdflib.Graph()
        graph.parse(path)
        yield path, graph


def test_pack_declares_no_core_owned_term() -> None:
    clashes = []
    for path, graph in _graphs():
        for kind in _KINDS:
            for subject in graph.subjects(RDF.type, kind):
                text = str(subject)
                if CONNECTOR_NS in text and text.rsplit("#", 1)[-1] in CORE_OWNED:
                    clashes.append(f"{path.name}: {text}")
    assert not clashes, f"pack redeclares core terms: {clashes}"


def test_pack_uses_no_pre_rename_iri_base() -> None:
    for path in sorted(ONTOLOGY_DIR.glob("*.ttl")):
        assert "http://knuckles.team" not in path.read_text(), path.name

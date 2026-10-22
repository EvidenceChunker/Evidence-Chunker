"""Opt-in external tests. Known specification gaps remain real failures."""
from pathlib import Path
import hashlib
import json
import pytest

def pytest_addoption(parser):
    group = parser.getgroup("submitted-specification")
    group.addoption("--run-spec-integration", action="store_true",
                    help="Run real PDF and embedding model tests.")
    group.addoption("--run-spec-install", action="store_true",
                    help="Run network-dependent clean-venv installation test.")
    group.addoption("--spec-install-target", default="evidence-chunker",
                    help="pip install target; default checks submitted PyPI installation.")
    group.addoption("--spec-pdf", default="data/pdfs/attention.pdf")
    group.addoption("--spec-oracle", default="tests/specification/fixtures/attention.expected.json")

def pytest_configure(config):
    config.addinivalue_line("markers", "spec(id): submitted specification row ID")
    config.addinivalue_line("markers", "spec_gap: expected requirement not met by baseline code; NOT xfail")
    config.addinivalue_line("markers", "spec_integration: real PDF or embedding model required")
    config.addinivalue_line("markers", "spec_install: clean environment installation with network")

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "spec_integration" in item.keywords and not config.getoption("--run-spec-integration"):
            item.add_marker(pytest.mark.skip(reason="Opt in with --run-spec-integration; not acceptance PASS"))
        if "spec_install" in item.keywords and not config.getoption("--run-spec-install"):
            item.add_marker(pytest.mark.skip(reason="Opt in with --run-spec-install; not acceptance PASS"))

@pytest.fixture(scope="session")
def spec_pdf_bundle(request):
    from evidence_chunker import EvidenceChunker
    from evidence_chunker.parser.docling import DoclingParser
    path = Path(request.config.getoption("--spec-pdf")).resolve()
    oracle_path = Path(request.config.getoption("--spec-oracle")).resolve()
    if not path.is_file() or not oracle_path.is_file():
        pytest.fail("Required PDF/oracle missing. Prepare --spec-pdf and --spec-oracle; see fixtures/README.md.")
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    assert oracle.get("pdf_sha256") == hashlib.sha256(path.read_bytes()).hexdigest(), "PDF hash mismatch"
    assert isinstance(oracle.get("table_count"), int) and oracle["table_count"] > 0
    assert isinstance(oracle.get("page_count"), int) and oracle["page_count"] > 0
    assert len(oracle.get("tables", [])) == oracle["table_count"], "Human-reviewed table oracle incomplete"
    assert [t["index"] for t in oracle["tables"]] == list(range(oracle["table_count"]))
    for t in oracle["tables"]:
        assert isinstance(t.get("page_no"), int) and 1 <= t["page_no"] <= oracle["page_count"]
        assert t.get("cell_texts"), "Fill independently reviewed cell_texts for every table"
        if t.get("expect_eu", True):
            assert t.get("data_rows"), "Fill independently reviewed non-header data rows"
            assert isinstance(t.get("expect_split"), bool)
            assert "section_header" in t and "required_text" in t and "absent_fields" in t
    chunker = EvidenceChunker()
    doc = chunker._parse(str(path))
    parsed = DoclingParser().from_doc(doc)
    return path, oracle, chunker, doc, parsed


from pathlib import Path

import pytest

from acadmem.config import BGE_QUERY_PREFIX, Config, load_config
from acadmem.models import (
    Block,
    Chunk,
    Section,
    infer_path_metadata,
    stable_document_id,
)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("3B1/Questions/foo.md", ("3B1", "Questions")),
        ("3B1/Handouts/week5/deep/Probability.pdf", ("3B1", "Handouts")),
        ("3B1/readme.md", ("3B1", None)),
        ("loose.md", (None, None)),
        ("Some Module/Notes/a b.md", ("Some Module", "Notes")),
    ],
)
def test_infer_path_metadata_is_not_hardcoded(path, expected):
    assert infer_path_metadata(path) == expected


def test_stable_document_id_is_deterministic_and_path_specific():
    assert stable_document_id("a/b.md") == stable_document_id("a/b.md")
    assert stable_document_id("a/b.md") != stable_document_id("a/c.md")


def test_section_page_range_from_blocks():
    s = Section("d#s1", "T", 1, ("T",), None, [Block("a", page=3), Block("b", page=5), Block("c")])
    assert (s.page_start, s.page_end) == (3, 5)
    assert Section("d#s2", "T", 1, ("T",), None, [Block("x")]).page_start is None


def test_chunk_embedding_text_includes_breadcrumb_without_duplicates():
    c = Chunk(
        chunk_id="d:00000", document_id="d", source_path="m/x.md", file_type="md",
        module="m", document_type=None, modification_time=0.0, chunk_index=0,
        section_id="d#s1", parent_section_id=None, heading_path=["Doc", "Bayes"],
        text="body", page_number=None, page_start=None, page_end=None,
    )
    assert c.embedding_text("Doc") == "Doc > Bayes\n\nbody"
    assert c.heading == "Doc > Bayes"


def test_config_precedence_file_env_override(tmp_path: Path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('embedding_model = "from-file"\nmax_chunk_tokens = 100\n', encoding="utf-8")
    env = {"ACADMEM_MAX_CHUNK_TOKENS": "200", "ACADMEM_VAULT": str(tmp_path / "vault")}
    cfg = load_config(cfg_file, env=env, embedding_model="from-arg")
    assert cfg.embedding_model == "from-arg"
    assert cfg.max_chunk_tokens == 200
    assert cfg.vault_path == tmp_path / "vault"


def test_config_defaults_and_query_prefix():
    cfg = Config()
    assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.query_prefix == BGE_QUERY_PREFIX
    assert load_config(env={}, query_prefix="none").query_prefix is None


def test_unknown_config_key_rejected():
    with pytest.raises(KeyError):
        Config().with_overrides(bogus=1)

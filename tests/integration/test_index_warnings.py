import pytest

from semantic_router.index import PineconeIndex, PostgresIndex
from semantic_router.utils.logger import logger


@pytest.mark.parametrize(
    "index_cls", [PineconeIndex, PostgresIndex], ids=lambda c: c.__name__
)
def test_building_an_index_warns_about_nothing(index_cls, make_index, monkeypatch):
    # a warning on every build, for settings the caller never touched, is noise
    warned: list[str] = []
    monkeypatch.setattr(logger, "warning", warned.append)
    make_index(index_cls)
    assert warned == []


def test_postgres_default_prefix_warns_about_nothing(monkeypatch):
    # index_prefix defaults to "semantic_router_", which 0.x tables are named with; it
    # is a supported setting, not a deprecated one
    import uuid

    warned: list[str] = []
    monkeypatch.setattr(logger, "warning", warned.append)
    index = PostgresIndex(index_name=f"t{uuid.uuid4().hex[:12]}")
    assert index.index_prefix == "semantic_router_"
    assert warned == []

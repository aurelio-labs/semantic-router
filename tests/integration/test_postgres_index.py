"""PostgresIndex against the local pgvector container. The async write path must
produce the same deterministic ids as the sync one, so adding an utterance twice
(or once per path) is a no-op rather than a duplicate row."""

import asyncio
import uuid

from semantic_router.index.postgres import PostgresIndex, PostgresIndexRecord

ROUTES = ["billing", "billing", "technical"]
UTTERANCES = ["i want a refund", "charged twice", "app crash"]
VECTORS = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]


def _index() -> PostgresIndex:
    # unique table per test so the suite runs in parallel; conftest drops it afterwards
    return PostgresIndex(
        index_name=f"test_{uuid.uuid4().hex}", index_prefix="", dimensions=3
    )


def test_record_id_depends_only_on_route_and_utterance():
    a = PostgresIndexRecord(vector=[0.0], route="billing", utterance="i want a refund")
    b = PostgresIndexRecord(vector=[1.0], route="billing", utterance="i want a refund")
    c = PostgresIndexRecord(vector=[1.0], route="billing", utterance="charged twice")
    assert a.id == b.id
    assert a.id != c.id


def test_async_add_is_idempotent_and_shares_ids_with_sync_add():
    index = _index()
    # open the sync connection too, so the cleanup fixture can drop the table
    index._init_index(force_create=True)

    async def run() -> int:
        await index._init_async_index(force_create=True)
        await index.aadd(VECTORS, ROUTES, UTTERANCES)
        # the bug this pins: a second async add used to insert three more rows
        await index.aadd(VECTORS, ROUTES, UTTERANCES)
        count = await index.alen()
        # alen() leaves a transaction open, and its share lock would block the
        # cleanup fixture's DROP TABLE on the sync connection; release it here
        await index.async_conn.close()
        return count

    assert asyncio.run(run()) == 3
    # the sync path must land on the same ids, so it adds nothing either
    index.add(VECTORS, ROUTES, UTTERANCES)
    assert len(index) == 3

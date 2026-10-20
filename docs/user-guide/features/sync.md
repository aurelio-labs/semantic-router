> **You're reading the docs for Semantic Router 0.x.** Version 1.0 removes sync: the index is the single source of truth. The [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1#sync-is-gone) covers the change, and [indexes](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/components/indexes) how it works now.

When you use a remote index like `PineconeIndex` or `QdrantIndex`, there are two copies of your routes: the ones in your `SemanticRouter` object, and the ones stored in the index. They can drift. Sync strategies decide which copy wins when they do.

Getting this right matters most in distributed setups, where several processes share one index. Pick the right strategy and you avoid both wasted startup time and subtle routing bugs.

## The strategies

- **`error`** — raise if local and remote differ. Use this when drift should never happen silently.
- **`remote`** — remote is the source of truth. Overwrite local to match.
- **`local`** — local is the source of truth. Overwrite remote to match.
- **`merge-force-local`** — merge, with local winning. Remote utterances survive only if their route also exists locally; everything else is dropped. Where a route exists on both sides with different `function_schemas` or `metadata`, local's version wins and is written to remote.
- **`merge-force-remote`** — the mirror image. Remote wins; local utterances survive only if their route exists remotely.
- **`merge`** — merge both, combining utterances where a route name appears on both sides. On a conflict in `function_schemas` or `metadata`, local wins.

You can apply a strategy two ways: automatically at startup with `auto_sync`, or on demand with `SemanticRouter.sync`.

## `auto_sync` at startup

Pass `auto_sync` when you create the router, and it reconciles with the index immediately. Note this adds to initialization time, since it has to compare and possibly write.

```python
import os
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
from semantic_router.index import PineconeIndex

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
os.environ["PINECONE_API_KEY"] = "<YOUR_API_KEY>"

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

routes = [politics, chitchat]

encoder = OpenAIEncoder()

pc_index = PineconeIndex(
    region="us-east-1",
    index_name="sync-example",
)
# make sure the index exists before the router tries to sync with it
pc_index.index = pc_index._init_index(force_create=True)

sr = SemanticRouter(
    encoder=encoder, routes=routes, index=pc_index,
    auto_sync="local",
)
```

Confirm the two sides agree:

```python
sr.is_synced()
```

## Checking sync

`is_synced()` compares the routes, utterances, and metadata in your router against what's in the index. It works in two passes.

**The fast check** hashes your local router — built from the encoder type and name, plus each route's name, utterances, description, function schemas, LLM, score threshold, and metadata — and compares it to the hash stored in the index. Matching hashes mean you're in sync, and it returns `True` straight away.

**The slow check** only runs if the hashes differ. It rebuilds a `LayerConfig` from the remote index and compares it to the local one field by field. If those match, you're in sync after all. If not, something has genuinely drifted.

To fix drift on demand, call `sync` with a strategy. It does exactly what `auto_sync` would do at startup. If your local router holds the ground truth, push it to the index:

```python
sr.sync(sync_mode="local")
```

Run `sr.is_synced()` again and it should now return `True`.

## Seeing what drifted

Often you'll want to know *why* things diverged before you overwrite anything. `get_utterance_diff` gives you a readable diff:

```python
diff = sr.get_utterance_diff()
```

```python
["- politics: don't you just hate the president",
"- politics: don't you just love the president",
"- politics: isn't politics the best thing ever",
'- politics: they will save the country!',
"- politics: they're going to destroy this country!",
"- politics: why don't you tell me about your political opinions",
'+ chitchat: how\'s the weather today?',
'+ chitchat: how are things going?',
'+ chitchat: lovely weather today',
'+ chitchat: the weather is horrendous',
'+ chitchat: let\'s go to the chippy']
```

It lists every route in the remote index and compares it against your local routes. Anything that differs shows up here.

For finer control, build an `UtteranceDiff` object from the two sets of utterances:

```python
local_utterances = sr.to_config().to_utterances()
remote_utterances = sr.index.get_utterances()

diff = UtteranceDiff.from_utterances(
    local_utterances=local_utterances, remote_utterances=remote_utterances
)
```

Every `Utterance` in `diff.diff` carries a `diff_tag`:

- `'+'` — exists in remote only.
- `'-'` — exists in local only.
- `' '` — exists in both.

Pull out whichever set you want to inspect:

```python
diff.get_tag("+")  # remote only
diff.get_tag("-")  # local only
diff.get_tag(" ")  # both
```

Once you understand the drift and know which side should win, apply a strategy:

```python
sr._execute_sync_strategy(sync_mode="local")
```

Then confirm:

```python
sr.is_synced()
```

`True` means the two sides match again.

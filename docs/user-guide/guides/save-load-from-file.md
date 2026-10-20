> **You're reading the docs for Semantic Router 0.x.** Version 1.0 is out: read [this page for v1](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/save-load-from-file), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

You can save a router to a file and load it back later — useful for shipping the same routes between environments, or just for not rebuilding them every time.

JSON and YAML both work.

```python
# JSON
router.to_json("router.json")
new_router = SemanticRouter.from_json("router.json")

# YAML
router.to_yaml("router.yaml")
new_router = SemanticRouter.from_yaml("router.yaml")
```

The file holds everything needed to recreate the router. If you use a remote index, the [sync features](../features/sync) keep the loaded router and the index in step.

## Full example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb)

```bash
!pip install -qU semantic-router
```

## Build a router

```python
from semantic_router import Route

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
```

```python
import os
from getpass import getpass
from semantic_router import SemanticRouter
from semantic_router.encoders import CohereEncoder

# dashboard.cohere.ai
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass(
    "Enter Cohere API Key: "
)

encoder = CohereEncoder()

router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

Quick check that it works:

```python
router("isn't politics the best thing ever")
```

```
RouteChoice(name='politics', function_call=None, similarity_score=None)
```

```python
router("how's the weather today?")
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

## Save it

```python
router.to_json("router.json")
```

## Load it

Have a look at what got saved:

```python
import json

with open("router.json", "r") as f:
    router_json = json.load(f)

print(router_json)
```

Encoder type, encoder name, and the routes — everything a new router needs. Load it with `from_json`:

```python
router = SemanticRouter.from_json("router.json")
```

Confirm it came back intact:

```python
print(
    f"""{router.encoder.type=}
{router.encoder.name=}
{router.routes=}"""
)
```

And that it still routes:

```python
router("isn't politics the best thing ever")
```

```
RouteChoice(name='politics', function_call=None, similarity_score=None)
```

```python
router("how's the weather today?")
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

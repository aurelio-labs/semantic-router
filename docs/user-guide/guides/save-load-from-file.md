Route layers can be saved to and loaded from files. This can be useful if we want to save a route layer to a file for later use, or if we want to load a route layer from a file.

We can save and load route layers to/from YAML or JSON files. For JSON we do:

```python
# save to JSON
router.to_json("router.json")
# load from JSON
new_router = SemanticRouter.from_json("router.json")
```

For YAML we do:

```python
# save to YAML
router.to_yaml("router.yaml")
# load from YAML
new_router = SemanticRouter.from_yaml("router.yaml")
```

The saved files contain all the information needed to initialize new semantic routers. If you are using a remote index, you can use the [sync features](../features/sync) to keep the router in sync with the index.

## Full Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb)

Here we will show how to save routers to YAML or JSON files, and how to load a router from file.

We start by installing the library:

```bash
!pip install -qU semantic-router
```

## Define Route

First let's create a list of routes:

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

We define a semantic router using these routes and using the Cohere encoder.

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

## Test Route

```python
router("isn't politics the best thing ever")
```

Output:
```
RouteChoice(name='politics', function_call=None, similarity_score=None)
```

```python
router("how's the weather today?")
```

Output:
```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

## Save To JSON

To save our semantic router we call the `to_json` method:

```python
router.to_json("router.json")
```

## Loading from JSON

We can view the router file we just saved to see what information is stored.

```python
import json

with open("router.json", "r") as f:
    router_json = json.load(f)

print(router_json)
```

It tells us our encoder type, encoder name, and routes. This is everything we need to initialize a new router. To do so, we use the `from_json` method.

```python
router = SemanticRouter.from_json("router.json")
```

We can confirm that our router has been initialized with the expected attributes by viewing the `SemanticRouter` object:

```python
print(
    f"""{router.encoder.type=}
{router.encoder.name=}
{router.routes=}"""
)
```

---

## Test Route Again

```python
router("isn't politics the best thing ever")
```

Output:
```
RouteChoice(name='politics', function_call=None, similarity_score=None)
```

```python
router("how's the weather today?")
```

Output:
```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
``` 
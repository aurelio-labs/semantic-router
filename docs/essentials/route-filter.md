We can filter the routes that the `SemanticRouter` considers when making a classification. This can be useful if we want to restrict the scope of possible routes based on some context.

For example, we may have a router with several routes, `politics`, `weather`, `chitchat`, etc. We may want to restrict the scope of the classification to only consider the `chitchat` route. We can do this by passing a `route_filter` argument to our `SemanticRouter` calls like so:

```python
sr("don't you love politics?", route_filter=["chitchat"])
```

In this case, the `SemanticRouter` will only consider the `chitchat` route for the classification.

## Full Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/09-route-filter.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/00-introduction.ipynb)

We start by installing the library:

```python
!pip install -qU semantic-router
```

We start by defining a dictionary mapping routes to example phrases that should trigger those routes.

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
```

Let's define another for good measure:

```python
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

Now we initialize our embedding model:

```python
import os
from getpass import getpass
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass(
    "Enter Cohere API Key: "
)
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
#     "Enter OpenAI API Key: "
# )

encoder = CohereEncoder()
# encoder = OpenAIEncoder()
```

Now we define the `SemanticRouter`. When called, the router will consume text (a query) and output the category (`Route`) it belongs to — to initialize a `SemanticRouter` we need our `encoder` model and a list of `routes`.

```python
from semantic_router.routers import SemanticRouter

sr = SemanticRouter(encoder=encoder, routes=routes)
```

Now we can test it:

```python
sr("don't you love politics?")
```

```
RouteChoice(name='politics', function_call=None, similarity_score=None)
```

```python
sr("how's the weather today?")
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

Both are classified accurately, what if we send a query that is unrelated to our existing `Route` objects?

```python
sr("I'm interested in learning about llama 2")
```

```
RouteChoice(name=None, function_call=None, similarity_score=None)
```

In this case, we return `None` because no matches were identified.

## Demonstrating the Filter Feature

Now, let's demonstrate the filter feature. We can specify a subset of routes to consider when making a classification. This can be useful if we want to restrict the scope of possible routes based on some context.

For example, let's say we only want to consider the "chitchat" route for a particular query:

```python
sr("don't you love politics?", route_filter=["chitchat"])
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

Even though the query might be more related to the "politics" route, it will be classified as "chitchat" because we've restricted the routes to consider.

Similarly, we can restrict it to the "politics" route:

```python
sr("how's the weather today?", route_filter=["politics"])
```

```
RouteChoice(name=None, function_call=None, similarity_score=None)
```

In this case, it will return `None` because the query doesn't match the "politics" route well enough to pass the threshold. 
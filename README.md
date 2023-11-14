[![Aurelio AI](https://pbs.twimg.com/profile_banners/1671498317455581184/1696285195/1500x500)](https://aurelio.ai)

# Semantic Router

Semantic Router is a superfast decision layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic of semantic vector space to make those decisions — _routing_ our requests using _semantic_ meaning.

## Quickstart

To get started with _semantic-router_ we install it like so:

```
pip install -qU semantic-router
```

We begin by defining a set of `Decision` objects. These are the decision paths that the semantic router can decide to use, let's try two simple decisions for now — one for talk on _politics_ and another for _chitchat_:

```python
from semantic_router.schema import Decision

# we could use this as a guide for our chatbot to avoid political conversations
politics = Decision(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president"
        "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!"
    ]
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Decision(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy"
    ]
)

# we place both of our decisions together into single list
decisions = [politics, chitchat]
```

We have our decisions ready, now we initialize an embedding / encoder model. We currently support a `CohereEncoder` and `OpenAIEncoder` — more encoders will be added soon. To initialize them we do:

```python
import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# for Cohere
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"
encoder = CohereEncoder()

# or for OpenAI
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
encoder = OpenAIEncoder()
```

With our `decisions` and `encoder` defined we now create a `DecisionLayer`. The decision layer handles our semantic decision making.

```python
from semantic_router.layer import DecisionLayer

dl = DecisionLayer(encoder=encoder, decisions=decisions)
```

We can now use our decision layer to make super fast decisions based on user queries. Let's try with two queries that should trigger our decisions:

```python
dl("don't you love politics?")
```

```
[Out]: 'politics'
```

Correct decision, let's try another:

```python
dl("how's the weather today?")
```

```
[Out]: 'chitchat'
```

We get both decisions correct! Now lets try sending an unrelated query:

```python
dl("I'm interested in learning about llama 2")
```

```
[Out]: 
```

In this case, no decision could be made as we had no matches — so our decision layer returned `None`!

## 📚 Resources

|     |     |
| --- | --- |
| 🏃 [Walkthrough](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/walkthrough.ipynb) | Quickstart Python notebook |

> **You're reading the docs for Semantic Router 0.x.** Threshold optimization is now calibration in version 1.0: read [the v1 page](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/features/calibration), or the [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) to upgrade.

A route matches only if its similarity score beats its `score_threshold`. Score above it and the route is chosen. Fall below and either another route wins, or nothing does.

That one number decides a lot, so it's worth getting right — but tuning it by hand is slow and fiddly. Instead, hand the router a few *(utterance, target route)* examples and let `fit` find the thresholds for you. It usually takes seconds and the improvement can be dramatic. `evaluate` tells you how well you're doing before and after.

## Full example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/06-threshold-optimization.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/06-threshold-optimization)

```python
!pip install -qU "semantic-router>=0.1.5"
```

## Set up the router

A `SemanticRouter` needs `routes` and an `encoder`. (If you use dynamic routes you'll also want an `llm`, or the OpenAI default.)

Four routes to start: *politics*, *chitchat*, *mathematics*, and *biology*.

```python
from semantic_router import Route

# steer a chatbot away from politics
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

# switch to a more conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "Did you watch the game last night?",
        "what's your favorite type of music?",
        "Have you read any good books lately?",
        "nice weather we're having",
        "Do you have any plans for the weekend?",
    ],
)

# hand off to an agent with math tools
mathematics = Route(
    name="mathematics",
    utterances=[
        "can you explain the concept of a derivative?",
        "What is the formula for the area of a triangle?",
        "how do you solve a system of linear equations?",
        "What is the concept of a prime number?",
        "Can you explain the Pythagorean theorem?",
    ],
)

# hand off to an agent with biology knowledge
biology = Route(
    name="biology",
    utterances=[
        "what is the process of osmosis?",
        "can you explain the structure of a cell?",
        "What is the role of RNA?",
        "What is genetic mutation?",
        "Can you explain the process of photosynthesis?",
    ],
)

routes = [politics, chitchat, mathematics, biology]
```

We'll use the local `HuggingFaceEncoder`. `CohereEncoder`, `FastEmbedEncoder`, `OpenAIEncoder`, and `AzureOpenAIEncoder` all work too.

```python
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder()
```

```python
from semantic_router import SemanticRouter

sr = SemanticRouter(encoder=encoder, routes=routes)
```

Out of the box it does reasonably well:

```python
for utterance in [
    "don't you love politics?",
    "how's the weather today?",
    "What's DNA?",
    "I'm interested in learning about llama 2",
]:
    print(f"{utterance} -> {sr(utterance).name}")
```

```
don't you love politics? -> politics
how's the weather today? -> chitchat
What's DNA? -> biology
I'm interested in learning about llama 2 -> None
```

## Measure it

`evaluate` takes a list of utterances and their target routes and returns an accuracy:

```python
test_data = [
    ("don't you love politics?", "politics"),
    ("how's the weather today?", "chitchat"),
    ("What's DNA?", "biology"),
    ("I'm interested in learning about llama 2", None),
]

X, y = zip(*test_data)

accuracy = sr.evaluate(X=X, y=y)
print(f"Accuracy: {accuracy*100:.2f}%")
```

```
Generating embeddings: 100%|██████████| 1/1 [00:00<00:00, 76.91it/s]
Accuracy: 100.00%
```

Perfect — on four examples. That's not a real test. Let's try a bigger, harder set.

*Tip: an LLM is a quick way to generate test examples for your own routes. The more realistic they are, the more your accuracy number will reflect real-world performance.*

```python
test_data = [
    # politics
    ("What's your opinion on the current government?", "politics"),
    ("Who do you think will win the next election?", "politics"),
    ("What are your thoughts on the new policy?", "politics"),
    ("How do you feel about the political situation?", "politics"),
    ("Do you agree with the president's actions?", "politics"),
    ("What's your stance on the political debate?", "politics"),
    ("How do you see the future of our country?", "politics"),
    ("What do you think about the opposition party?", "politics"),
    ("Do you believe the government is doing enough?", "politics"),
    ("What's your opinion on the political scandal?", "politics"),
    ("Do you think the new law will make a difference?", "politics"),
    ("What are your thoughts on the political reform?", "politics"),
    ("Do you agree with the government's foreign policy?", "politics"),
    # chitchat
    ("What's the weather like?", "chitchat"),
    ("It's a beautiful day today.", "chitchat"),
    ("How's your day going?", "chitchat"),
    ("It's raining cats and dogs.", "chitchat"),
    ("Let's grab a coffee.", "chitchat"),
    ("What's up?", "chitchat"),
    ("It's a bit chilly today.", "chitchat"),
    ("How's it going?", "chitchat"),
    ("Nice weather we're having.", "chitchat"),
    ("It's a bit windy today.", "chitchat"),
    ("Let's go for a walk.", "chitchat"),
    ("How's your week been?", "chitchat"),
    ("It's quite sunny today.", "chitchat"),
    ("How are you feeling?", "chitchat"),
    ("It's a bit cloudy today.", "chitchat"),
    # mathematics
    ("What is the Pythagorean theorem?", "mathematics"),
    ("Can you solve this quadratic equation?", "mathematics"),
    ("What is the derivative of x squared?", "mathematics"),
    ("Explain the concept of integration.", "mathematics"),
    ("What is the area of a circle?", "mathematics"),
    ("How do you calculate the volume of a sphere?", "mathematics"),
    ("What is the difference between a vector and a scalar?", "mathematics"),
    ("Explain the concept of a matrix.", "mathematics"),
    ("What is the Fibonacci sequence?", "mathematics"),
    ("How do you calculate permutations?", "mathematics"),
    ("What is the concept of probability?", "mathematics"),
    ("Explain the binomial theorem.", "mathematics"),
    ("What is the difference between discrete and continuous data?", "mathematics"),
    ("What is a complex number?", "mathematics"),
    ("Explain the concept of limits.", "mathematics"),
    # biology
    ("What is photosynthesis?", "biology"),
    ("Explain the process of cell division.", "biology"),
    ("What is the function of mitochondria?", "biology"),
    ("What is DNA?", "biology"),
    ("What is the difference between prokaryotic and eukaryotic cells?", "biology"),
    ("What is an ecosystem?", "biology"),
    ("Explain the theory of evolution.", "biology"),
    ("What is a species?", "biology"),
    ("What is the role of enzymes?", "biology"),
    ("What is the circulatory system?", "biology"),
    ("Explain the process of respiration.", "biology"),
    ("What is a gene?", "biology"),
    ("What is the function of the nervous system?", "biology"),
    ("What is homeostasis?", "biology"),
    ("What is the difference between a virus and a bacteria?", "biology"),
    ("What is the role of the immune system?", "biology"),
    # some None examples, so thresholds don't collapse to zero
    ("What is the capital of France?", None),
    ("how many people live in the US?", None),
    ("when is the best time to visit Bali?", None),
    ("how do I learn a language", None),
    ("tell me an interesting fact", None),
    ("what is the best programming language?", None),
    ("I'm interested in learning about llama 2", None),
]
```

```python
X, y = zip(*test_data)

accuracy = sr.evaluate(X=X, y=y)
print(f"Accuracy: {accuracy*100:.2f}%")
```

```
Generating embeddings: 100%|██████████| 1/1 [00:00<00:00, 9.23it/s]
Accuracy: 34.85%
```

Ouch. The good news: this is easy to fix.

## Optimize the thresholds

Optimization finds the best `score_threshold` for each route. First, look at the defaults:

```python
route_thresholds = sr.get_thresholds()
print("Default route thresholds:", route_thresholds)
```

```
Default route thresholds: {'politics': 0.5, 'chitchat': 0.5, 'mathematics': 0.5, 'biology': 0.5}
```

Every route sits at 0.5. Now call `fit` with your training utterances `X` and labels `y`:

```python
sr.fit(X=X, y=y)
```

```
Generating embeddings: 100%|██████████| 1/1 [00:00<00:00, 9.21it/s]
Training: 100%|██████████| 500/500 [00:01<00:00, 419.45it/s, acc=0.89]
```

The thresholds have moved a lot:

```python
route_thresholds = sr.get_thresholds()
print("Updated route thresholds:", route_thresholds)
```

```
Updated route thresholds: {'politics': 0.05050505050505051, 'chitchat': 0.32323232323232326, 'mathematics': 0.18181818181818182, 'biology': 0.21212121212121213}
```

Don't read too much into the absolute values. The right thresholds depend heavily on the encoder — OpenAI's `text-embedding-ada-002`, for instance, tends to land in the `0.5` to `0.8` range with this library. That's exactly why fitting beats guessing.

The result:

```python
accuracy = sr.evaluate(X=X, y=y)
print(f"Accuracy: {accuracy*100:.2f}%")
```

```
Generating embeddings: 100%|██████████| 1/1 [00:00<00:00, 8.89it/s]
Accuracy: 89.39%
```

From 35% to 89%, in about a second.

To push further, the next lever is your routes themselves: add more utterances, look at *which* examples still fail, and reshape the routes around them. That's more hands-on than `fit`, but it's how you squeeze out the last few points.

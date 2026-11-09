[![Semantic Router](https://i.ibb.co.com/g423grt/semantic-router-banner.png)](https://aurelio.ai)

<p>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/semantic-router?logo=python&logoColor=gold" />
<a href="https://github.com/aurelio-labs/semantic-router/graphs/contributors"><img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/aurelio-labs/semantic-router" />
<a href="https://github.com/aurelio-labs/semantic-router/commits/main"><img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/aurelio-labs/semantic-router" />
<img alt="" src="https://img.shields.io/github/repo-size/aurelio-labs/semantic-router" />
<a href="https://github.com/aurelio-labs/semantic-router/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/aurelio-labs/semantic-router" />
<a href="https://github.com/aurelio-labs/semantic-router/pulls"><img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/aurelio-labs/semantic-router" />
<a href="https://github.com/aurelio-labs/semantic-router/blob/main/LICENSE"><img alt="Github License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

Semantic Router is a superfast decision-making layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, it decides in milliseconds: it compares the _semantic_ meaning of a request against your routes in vector space, or asks a small decision model in a single fast call. Every decision comes with a confidence you can calibrate into a real probability.

#### [Read the Docs](https://docs.aurelio.ai/docs/semantic-router/v1/get-started/introduction)

---

## Versions

Semantic Router has two release lines. The **1.x line** (`main` branch) rebuilds the routing layer around calibrated confidence, decision models and composable routers. The **0.x line** (`v0` branch) finishes at 0.3.0 and receives bug fixes only.

```
pip install -qU "semantic-router>=1,<2"   # 1.x
pip install -qU "semantic-router<1"       # 0.x
```

Upgrading from 0.x? The [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1) covers every change, with before and after code. The 0.x docs stay on the [versions page](https://docs.aurelio.ai/docs/semantic-router/v1/get-started/versions).

---

## Quickstart

This quickstart runs entirely on your machine, with a small local embedding model and no API keys:

```
pip install -qU "semantic-router[fastembed]"
```

We begin by defining a set of `Route` objects. These are the decisions the router can make. Let's try two simple routes for now, one for talk on _politics_ and another for _chitchat_:

```python
from semantic_router import Route

# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
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

# we place both of our decisions together into single list
routes = [politics, chitchat]
```

Next we pick an encoder, the model that turns text into vectors. We use a local FastEmbed model here; OpenAI, Cohere, Hugging Face and [many more](https://github.com/aurelio-labs/semantic-router/tree/main/semantic_router/encoders) work the same way.

```python
from semantic_router.encoders import FastEmbedEncoder

encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
```

With our `routes` and `encoder` defined, we create an `IndexedRouter`. It embeds every utterance once and matches each query against them.

```python
from semantic_router import IndexedRouter

router = IndexedRouter(encoder=encoder, routes=routes)
```

Now we can make super fast decisions based on user queries:

```python
router("don't you love politics?").name
```

```
[Out]: 'politics'
```

Correct decision, let's try another:

```python
router("how's the weather today?").name
```

```
[Out]: 'chitchat'
```

We get both decisions correct! By default the best route always wins, so to let the router say "none of these" we calibrate it on a few labelled queries it hasn't seen (`None` means no route should match) and set a confidence threshold:

```python
router.fit(
    X=[
        "who should I vote for in the election?", "what do you think of the prime minister?", "is the government doing a good job?",
        "tell me your opinion on the new tax law", "are you left wing or right wing?", "the opposition party is a joke",
        "nice day out, isn't it?", "it's freezing outside", "how was your weekend?",
        "fancy grabbing some fish and chips?", "what a lovely sunny afternoon", "how have you been?",
        "how do I reset my password?", "what's the capital of australia?", "write me a python function to sort a list",
        "what is the boiling point of water?", "recommend a good sci-fi book", "how many legs does a spider have?",
    ],
    y=["politics"] * 6 + ["chitchat"] * 6 + [None] * 6,
)
router.confidence_threshold = 0.5  # now a probability: act only when it's more likely right than wrong

router("I'm interested in learning about llama 2").name
```

```
[Out]:
```

No route cleared the bar, so the router returned `None`!

## Beyond embeddings

- **`DecisionRouter`** asks a decision model (TypeSafe's Jev, nace.ai's Drex, Laya or Von) which route applies, and can answer several yes/no, choice and score questions about one message in a single call. New to decision models? Read [Understanding Jev](https://jamesbriggs.io/understanding-jev) or watch [the video](https://youtu.be/P-t4F_UlxkI).
- **`Chain`** answers easy queries from a fast router and falls through to a slower one only when it isn't sure. **`Merge`** averages several routers.
- **Hybrid search**: give `IndexedRouter` a `sparse_encoder` to weigh exact keywords alongside meaning.
- **Production indexes**: keep your routes in [Pinecone](https://github.com/aurelio-labs/semantic-router/blob/main/docs/indexes/pinecone.ipynb), [Qdrant](https://github.com/aurelio-labs/semantic-router/blob/main/docs/indexes/qdrant.ipynb) or Postgres.
- **Multi-modal**: route on [images](https://github.com/aurelio-labs/semantic-router/blob/main/docs/07-multi-modal.ipynb) as well as text.

---

## 📚 Resources

### Docs

| Notebook | Description |
| -------- | ----------- |
| [Introduction](https://github.com/aurelio-labs/semantic-router/blob/main/docs/00-introduction.ipynb) | Introduction to Semantic Router and the `IndexedRouter` |
| [Save/Load Routers](https://github.com/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb) | Save any router to YAML or JSON and load it back, calibration included |
| [LangChain Integration](https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb) | How to integrate Semantic Router with LangChain Agents |
| [Local Execution](https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb) | Fully local routing with FastEmbed, no API keys |
| [Calibration](https://github.com/aurelio-labs/semantic-router/blob/main/docs/06-calibration.ipynb) | Turn raw scores into probabilities with `fit()`, and set thresholds that mean what they say |
| [Multi-Modal Routes](https://github.com/aurelio-labs/semantic-router/blob/main/docs/07-multi-modal.ipynb) | Using multi-modal routes to identify Shrek vs. not-Shrek pictures |
| [Healthcare Administrative Routing](https://github.com/aurelio-labs/semantic-router/blob/main/docs/examples/healthcare-routing/healthcare-administrative-routing.ipynb) | Fully local routing, calibration, and evaluation across seven synthetic healthcare administrative workflows |

### Online Course

[![Semantic Router Course](https://github.com/aurelio-labs/assets/blob/main/images/aurelio-1080p-header-dark-semantic-router.jpg)](https://www.aurelio.ai/course/semantic-router)

### Community

- Dimitrios Manias, Ali Chouman, Abdallah Shami, [Semantic Routing for Enhanced Performance of LLM-Assisted Intent-Based 5G Core Network Management and Orchestration](https://arxiv.org/abs/2404.15869), IEEE GlobeCom 2024
- Julian Horsey, [Semantic Router superfast decision layer for LLMs and AI agents](https://www.geeky-gadgets.com/semantic-router-superfast-decision-layer-for-llms-and-ai-agents/), Geeky Gadgets
- azhar, [Beyond Basic Chatbots: How Semantic Router is Changing the Game](https://medium.com/ai-insights-cobet/beyond-basic-chatbots-how-semantic-router-is-changing-the-game-783dd959a32d), AI Insights @ Medium
- Daniel Avila, [Semantic Router: Enhancing Control in LLM Conversations](https://blog.codegpt.co/semantic-router-enhancing-control-in-llm-conversations-68ce905c8d33), CodeGPT @ Medium
- Yogendra Sisodia, [Stop Chat-GPT From Going Rogue In Production With Semantic Router](https://medium.com/@scholarly360/stop-chat-gpt-from-going-rogue-in-production-with-semantic-router-937a4768ae19), Medium
- Aniket Hingane, [LLM Apps: Why you Must Know Semantic Router in 2024: Part 1](https://medium.com/@learn-simplified/llm-apps-why-you-must-know-semantic-router-in-2024-part-1-bfbda81374c5), Medium
- Adrien Sales, [🔀 Semantic Router w. ollama/gemma2 : real life 10ms hotline challenge 🤯](https://dev.to/adriens/semantic-router-w-ollamagemma2-real-life-10ms-hotline-challenge-1i3f)
- Adrien Sales, [Kaggle Notebook 🔀 Semantic Router: `ollama`/ `gemma2:9b` hotline](https://www.kaggle.com/code/adriensales/semantic-router-ollama-gemma2-hotline/notebook)

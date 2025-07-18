[![Semantic Router](https://i.ibb.co.com/g423grt/semantic-router-banner.png)](https://aurelio.ai)

<p>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/semantic-router?logo=python&logoColor=gold" />
<a href="https://github.com/aurelio-labs/semantic-router/graphs/contributors"><img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/aurelio-labs/semantic-router" />
<a href="https://github.com/aurelio-labs/semantic-router/commits/main"><img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/aurelio-labs/semantic-router" />
<img alt="" src="https://img.shields.io/github/repo-size/aurelio-labs/semantic-router" />
<a href="https://github.com/aurelio-labs/semantic-router/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/aurelio-labs/semantic-router" />
<a href="https://github.com/aurelio-labs/semantic-router/pulls"><img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/aurelio-labs/semantic-router" />
<img src="https://codecov.io/gh/aurelio-labs/semantic-router/graph/badge.svg?token=H8OOMV2TUF" />
<a href="https://github.com/aurelio-labs/semantic-router/blob/main/LICENSE"><img alt="Github License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

Semantic Router is a superfast decision-making layer for your LLMs and agents. Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic of semantic vector space to make those decisions — _routing_ our requests using _semantic_ meaning.

#### [Read the Docs](https://docs.aurelio.ai/semantic-router/get-started/introduction)

---

## Quickstart

To get started with _semantic-router_ we install it like so:

```
pip install -qU semantic-router
```

❗️ _If wanting to use a fully local version of semantic router you can use `HuggingFaceEncoder` and `LlamaCppLLM` (`pip install -qU "semantic-router[local]"`, see [here](https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb)). To use the `HybridRouteLayer` you must `pip install -qU "semantic-router[hybrid]"`._

We begin by defining a set of `Route` objects. These are the decision paths that the semantic router can decide to use, let's try two simple routes for now — one for talk on _politics_ and another for _chitchat_:

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

We have our routes ready, now we initialize an embedding / encoder model. We currently support a `CohereEncoder` and `OpenAIEncoder` — more encoders will be added soon. To initialize them we do:

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

With our `routes` and `encoder` defined we now create a `RouteLayer`. The route layer handles our semantic decision making.

```python
from semantic_router.routers import SemanticRouter

rl = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

We can now use our route layer to make super fast decisions based on user queries. Let's try with two queries that should trigger our route decisions:

```python
rl("don't you love politics?").name
```

```
[Out]: 'politics'
```

Correct decision, let's try another:

```python
rl("how's the weather today?").name
```

```
[Out]: 'chitchat'
```

We get both decisions correct! Now lets try sending an unrelated query:

```python
rl("I'm interested in learning about llama 2").name
```

```
[Out]:
```

In this case, no decision could be made as we had no matches — so our route layer returned `None`!

## Integrations

The _encoders_ of semantic router include easy-to-use integrations with [Cohere](https://github.com/aurelio-labs/semantic-router/blob/main/semantic_router/encoders/cohere.py), [OpenAI](https://github.com/aurelio-labs/semantic-router/blob/main/docs/encoders/openai-embed-3.ipynb), [Hugging Face](https://github.com/aurelio-labs/semantic-router/blob/main/docs/encoders/huggingface.ipynb), [FastEmbed](https://github.com/aurelio-labs/semantic-router/blob/main/docs/encoders/fastembed.ipynb), and [more](https://github.com/aurelio-labs/semantic-router/tree/main/semantic_router/encoders) — we even support [multi-modality](https://github.com/aurelio-labs/semantic-router/blob/main/docs/07-multi-modal.ipynb)!.

Our utterance vector space also integrates with [Pinecone](https://github.com/aurelio-labs/semantic-router/blob/main/docs/indexes/pinecone.ipynb) and [Qdrant](https://github.com/aurelio-labs/semantic-router/blob/main/docs/indexes/qdrant.ipynb)!

---

## 📚 Resources

### Docs

| Notebook | Description |
| -------- | ----------- |
| [Introduction](https://github.com/aurelio-labs/semantic-router/blob/main/docs/00-introduction.ipynb) | Introduction to Semantic Router and static routes |
| [Dynamic Routes](https://github.com/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb) | Dynamic routes for parameter generation and functionc calls |
| [Save/Load Layers](https://github.com/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb) | How to save and load `RouteLayer` from file |
| [LangChain Integration](https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb) | How to integrate Semantic Router with LangChain Agents |
| [Local Execution](https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb) | Fully local Semantic Router with dynamic routes — *local models such as Mistral 7B outperform GPT-3.5 in most tests* |
| [Route Optimization](https://github.com/aurelio-labs/semantic-router/blob/main/docs/06-threshold-optimization.ipynb) | How to train route layer thresholds to optimize performance |
| [Multi-Modal Routes](https://github.com/aurelio-labs/semantic-router/blob/main/docs/07-multi-modal.ipynb) | Using multi-modal routes to identify Shrek vs. not-Shrek pictures |

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
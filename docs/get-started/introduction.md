---
title: Introduction
description: Semantic Router is a superfast decision layer for LLMs and agents.
---

Semantic Router is a superfast decision layer for LLMs and agents. Most tools make routing decisions by asking an LLM and waiting for it to answer. That's slow. Semantic Router decides in milliseconds: it compares the *meaning* of an input against your routes in vector space, or asks a small decision model that answers in a single fast call.

The payoff is simple. Decisions land in milliseconds, not seconds. You skip an LLM call, so you cut cost. And because routes are explicit, you stay in control of what happens next.

## How it works

You define routes: the decisions your app can make. Each one has a few example phrases and, if you like, a plain description. A router scores every route for an incoming request and hands back the best one, with a confidence.

There are a few kinds of router, and they all behave the same way:

- **`IndexedRouter`** embeds your example utterances once, up front. At runtime it matches each request to the closest route by meaning. Give it a sparse encoder as well and it becomes a hybrid router that also weighs exact keywords.
- **`DecisionRouter`** asks a decision model which route applies: TypeSafe's Jev, or any model that speaks the same API. It needs no index and no embeddings, because it reads your route descriptions.
- **`Chain`** and **`Merge`** are routers made of routers. A Chain tries a fast router first and only falls through to a slower one when the fast one isn't sure. A Merge asks several routers at once and averages their answers.

## Confidence you can act on

Every router returns a `confidence` with its choice. Call `fit()` with a few dozen labelled example queries and the router learns to turn its raw scores into probabilities. After that, a confidence of 0.85 means the route is right about 85% of the time. So you can set a threshold from a policy ("only act when you're 90% sure") instead of guessing at similarity scores.

## What you can build

- **Intent routing** for chatbots and agents: send each message to the right tool, prompt or team.
- **Guardrails**: catch off-topic, unsafe or out-of-scope requests before they reach an LLM.
- **Multi-modal routing**: route on images as well as text.
- **Production scale**: keep your routes in Pinecone, Qdrant or Postgres, and share one index between many routers.

It works with the encoders you already use: OpenAI, Cohere, Hugging Face, FastEmbed, and more. With a local encoder like FastEmbed, routing runs entirely on your machine.

## Start here

New to Semantic Router? The [quickstart](quickstart) gets you routing in a few minutes, with no API keys.

Upgrading from 0.x? Version 1 is a breaking release. The [migration guide](../user-guide/guides/migration-to-v1) covers every change, with before and after code. The 0.x docs stay online: see [versions](versions).

## Resources

- [GitHub repository](https://github.com/aurelio-labs/semantic-router)
- [Online course](https://www.aurelio.ai/course/semantic-router)

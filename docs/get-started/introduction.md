Semantic Router is a superfast decision layer for LLMs and agents. Most tools make routing decisions by asking an LLM and waiting for it to answer. That's slow. Semantic Router skips the wait — it compares the *meaning* of an input against your routes in vector space and picks a match in milliseconds.

The payoff is simple. Decisions land in milliseconds, not seconds. You skip an LLM call, so you cut cost. And because routes are explicit, you stay in control of what happens next.

## How it works

You define routes, each one a handful of example phrases. Semantic Router embeds those examples once, up front. At runtime it embeds the incoming request and matches it to the closest route by meaning. No keyword matching, and no LLM in the hot path.

From there you can:

- **Trigger functions.** Dynamic routes extract parameters and call your code.
- **Go multi-modal.** Route on images, not just text.
- **Scale it.** Persist routes in Pinecone, Qdrant, or Postgres.
- **Run it anywhere.** Cloud APIs, fully local, or a mix of both.

It works with the encoders you already use — OpenAI, Cohere, Hugging Face, FastEmbed, and more.

## Running local or in the cloud

You choose how much runs on your machine:

- **Cloud.** Embeddings from OpenAI, Cohere, or another API.
- **Hybrid.** Local embeddings, API-based LLMs.
- **Fully local.** Everything on your hardware, with models like Llama and Mistral. No external calls.

## Start here

New to Semantic Router? The [quickstart](quickstart) gets you routing in a few minutes.

Upgrading from a 0.0.x release? v0.1 introduced breaking changes — the [migration guide](../user-guide/guides/migration-to-v0-1) walks you through them.

## Resources

- [GitHub repository](https://github.com/aurelio-labs/semantic-router)
- [Online course](https://www.aurelio.ai/course/semantic-router)

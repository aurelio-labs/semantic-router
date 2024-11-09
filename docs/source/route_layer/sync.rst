Synchronizing with Remote Instances
===================================

Semantic router supports several *remote instances* that store our routes and
utterances, such as Pinecone or Qdrant, supported via the `PineconeIndex` and
`QdrantIndex` respectively.

Using these remote instances is optional, but it allows us to scale our
semantic router to a larger number of routes and utterances. However, we must
decide how to synchronize between our local metadata and the remote instance —
particularly when reinitializing a local instance that should connect to an
existing remote instance.

Semantic router supports several synchronization strategies that can be passed
to the `sync` parameter of the various `BaseIndex` objects. Those strategies
are:

* `error`: Raise an error if local and remote are not synchronized.
* `remote`: Take remote as the source of truth and update local to align.
* `local`: Take local as the source of truth and update remote to align.
* `merge-force-remote`: Merge both local and remote taking only remote routes
  utterances when a route with same route name is present both locally and
  remotely.
* `merge-force-local`: Merge both local and remote taking only local routes
  utterances when a route with same route name is present both locally and
  remotely.
* `merge`: Merge both local and remote, merging also local and remote utterances
  when a route with same route name is present both locally and remotely.

You can try this yourself by running the following:

.. code-block:: python

    from semantic_router import Route
    from semantic_router.encoders import OpenAIEncoder
    from semantic_router.index.pinecone import PineconeIndex
    from semantic_router.layer import RouteLayer


    politics = Route(
        name="politics",
        utterances=[
            "isn't politics the best thing ever",
            "why don't you tell me about your political opinions",
            "don't you just love the president",
        ],
    )

    chitchat = Route(
        name="chitchat",
        utterances=[
            "how's the weather today?",
            "how are things going?",
        ],
    )

    routes = [politics, chitchat]

    encoder = OpenAIEncoder(openai_api_key=openai_api_key)

    pc_index = PineconeIndex(
        api_key=pinecone_api_key,
        region="us-east-1",
        index_name="sync-example",
        sync="local",  # here we specify the synchronization strategy
    )

    rl = RouteLayer(encoder=encoder, routes=routes, index=pc_index)

When initializing the `PineconeIndex` object, we can specify the `sync` parameter.

Checking for Synchronization
----------------------------

To verify whether the local and remote instances are synchronized, you can use
the `RouteLayer.is_synced` method. This method checks if the routes, utterances,
and associated metadata in the local instance match those stored in the remote
index.

The `is_synced` method works in two steps. The first is our *fast* sync check.
The fast check creates a hash of our local route layer which is constructed
from:

- `encoder_type` and `encoder_name`
- `route` names
- `route` utterances
- `route` description
- `route` function schemas (if any)
- `route` llm (if any)
- `route` score threshold
- `route` metadata (if any)

The fast check then compares this hash to the hash of the remote index. If
the hashes match, we know that the local and remote instances are synchronized
and we can return `True`. If the hashes do not match, we need to perform a
*slow* sync check.

The slow sync check works by creating a `LayerConfig` object from the remote
index and then comparing this to our local `LayerConfig` object. If the two
objects match, we know that the local and remote instances are synchronized and
we can return `True`. If the two objects do not match, we need to perform a
diff.

The diff works by creating a list of all the routes in the remote index and
then comparing these to the routes in our local instance. Any differences
between the remote and local routes are shown in the diff.
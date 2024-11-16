Synchronizing the Route Layer with Indexes
===========================================

The `RouteLayer` class is the main class in the semantic router package. It
contains the routes and allows us to interact with the underlying index. Both
the `RouteLayer` and the various index classes support synchronization
strategies that allow us to synchronize the routes and utterances in the layer
with the underlying index.

This functionality becomes increasingly important when using the semantic
router in a distributed environment. For example, when using one of the *remote
instances*, such as `PineconeIndex` or `QdrantIndex`. Deciding the correct
synchronization strategy for these remote indexes will save application time
and reduce the risk of errors.

Semantic router supports several synchronization strategies. Those strategies
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

There are two ways to specify the synchronization strategy. The first is to
specify the strategy when initializing the `RouteLayer` object via the
`auto_sync` parameter. The second is to trigger synchronization directly via
the `RouteLayer.sync` method.

---

Using the `auto_sync` parameter
-------------------------------

The `auto_sync` parameter is used to specify the synchronization strategy when
initializing the `RouteLayer` object. Depending on the chosen strategy, the
`RouteLayer` object will automatically synchronize with the defined index. As
this happens on initialization, this will often increase the initialization
time of the `RouteLayer` object.

Let's see an example of `auto_sync` in action.

.. code-block:: python

    from semantic_router import Route

    # we could use this as a guide for our chatbot to avoid political conversations
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

    encoder = OpenAIEncoder(openai_api_key=openai_api_key)

    pc_index = PineconeIndex(
        api_key=pinecone_api_key,
        region="us-east-1",
        index_name="sync-example",
    )
    # before initializing the RouteLayer with auto_sync we should initialize
    # the index
    pc_index.index = pc_index._init_index(force_create=True)

    # now we can initialize the RouteLayer with local auto_sync
    rl = RouteLayer(
        encoder=encoder, routes=routes, index=pc_index,
        auto_sync="local"
    )

Now we can run `rl.is_synced()` to confirm that our local and remote instances
are synchronized.

.. code-block:: python

    rl.is_synced()

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
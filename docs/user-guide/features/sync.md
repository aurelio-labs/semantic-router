The `SemanticRouter` class is the main class in the semantic router package. It contains the routes and allows us to interact with the underlying index. Both the `SemanticRouter` and the various index classes support synchronization strategies that allow us to synchronize the routes and utterances in the layer with the underlying index.

This functionality becomes increasingly important when using the semantic router in a distributed environment. For example, when using one of the *remote instances*, such as `PineconeIndex` or `QdrantIndex`. Deciding the correct synchronization strategy for these remote indexes will save application time and reduce the risk of errors.

Semantic router supports several synchronization strategies. Those strategies are:

* `error`: Raise an error if local and remote are not synchronized.

* `remote`: Take remote as the source of truth and update local to align.

* `local`: Take local as the source of truth and update remote to align.

* `merge-force-local`: Merge both local and remote keeping local as the priority. Remote utterances are only merged into local *if* a matching route for the utterance is found in local, all other route-utterances are dropped. Where a route exists in both local and remote, but each contains different `function_schema` or `metadata` information, the local version takes priority and local `function_schemas` and `metadata` is propagated to all remote utterances belonging to the given route.

* `merge-force-remote`: Merge both local and remote keeping remote as the priority. Local utterances are only merged into remote *if* a matching route for the utterance is found in the remote, all other route-utterances are dropped. Where a route exists in both local and remote, but each contains different `function_schema` or `metadata` information, the remote version takes priority and remote `function_schemas` and `metadata` are propagated to all local routes.

* `merge`: Merge both local and remote, merging also local and remote utterances when a route with same route name is present both locally and remotely. If a route exists in both local and remote but contains different `function_schemas` or `metadata` information, the local version takes priority and local `function_schemas` and `metadata` are propagated to all remote routes.

There are two ways to specify the synchronization strategy. The first is to specify the strategy when initializing the `SemanticRouter` object via the `auto_sync` parameter. The second is to trigger synchronization directly via the `SemanticRouter.sync` method.

---

## Using the `auto_sync` parameter

The `auto_sync` parameter is used to specify the synchronization strategy when initializing the `SemanticRouter` object. Depending on the chosen strategy, the `SemanticRouter` object will automatically synchronize with the defined index. As this happens on initialization, this will often increase the initialization time of the `SemanticRouter` object.

Let's see an example of `auto_sync` in action.

```python
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
from semantic_router.indexes import PineconeIndex

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
# before initializing the SemanticRouter with auto_sync we should initialize
# the index
pc_index.index = pc_index._init_index(force_create=True)

# now we can initialize the SemanticRouter with local auto_sync
sr = SemanticRouter(
    encoder=encoder, routes=routes, index=pc_index,
    auto_sync="local"
)
```

Now we can run `sr.is_synced()` to confirm that our local and remote instances are synchronized.

```python
sr.is_synced()
```

## Checking for Synchronization

To verify whether the local and remote instances are synchronized, you can use the `SemanticRouter.is_synced` method. This method checks if the routes, utterances, and associated metadata in the local instance match those stored in the remote index.

The `is_synced` method works in two steps. The first is our *fast* sync check. The fast check creates a hash of our local route layer which is constructed from:

- `encoder_type` and `encoder_name`
- `route` names
- `route` utterances
- `route` description
- `route` function schemas (if any)
- `route` llm (if any)
- `route` score threshold
- `route` metadata (if any)

The fast check then compares this hash to the hash of the remote index. If the hashes match, we know that the local and remote instances are synchronized and we can return `True`. If the hashes do not match, we need to perform a *slow* sync check.

The slow sync check works by creating a `LayerConfig` object from the remote index and then comparing this to our local `LayerConfig` object. If the two objects match, we know that the local and remote instances are synchronized and we can return `True`. If the two objects do not match, we must investigate and decide how to synchronize the two instances.

To quickly sync the local and remote instances we can use the `SemanticRouter.sync` method. This method is equivalent to the `auto_sync` strategy specified when initializing the `SemanticRouter` object. So, if we assume our local `SemanticRouter` object contains the ground truth routes, we would use the `local` strategy to copy our local routes to the remote instance.

```python
sr.sync(sync_mode="local")
```

After running the above code, we can check whether the local and remote instances are synchronized by rerunning `sr.is_synced()`, which should now return `True`.

## Investigating Synchronization Differences

We may often need to further investigate and understand *why* our local and remote instances have become desynchronized. The first step in further investigation and resolution of synchronization differences is to see the differences. We can get a readable diff using the `SemanticRouter.get_utterance_diff` method.

```python
diff = sr.get_utterance_diff()
```

```python
["- politics: don't you just hate the president",
"- politics: don't you just love the president",
"- politics: isn't politics the best thing ever",
'- politics: they will save the country!',
"- politics: they're going to destroy this country!",
"- politics: why don't you tell me about your political opinions",
'+ chitchat: how\'s the weather today?',
'+ chitchat: how are things going?',
'+ chitchat: lovely weather today',
'+ chitchat: the weather is horrendous',
'+ chitchat: let\'s go to the chippy']
```

The diff works by creating a list of all the routes in the remote index and then comparing these to the routes in our local instance. Any differences between the remote and local routes are shown in the above diff.

Now, to resolve these differences we will need to initialize an `UtteranceDiff` object. This object will contain the differences between the remote and local utterances. We can then use this object to decide how to synchronize the two instances. To initialize the `UtteranceDiff` object we need to get our local and remote utterances.

```python
local_utterances = sr.to_config().to_utterances()
remote_utterances = sr.index.get_utterances()
```

We create an utterance diff object like so:

```python
diff = UtteranceDiff.from_utterances(
    local_utterances=local_utterances, remote_utterances=remote_utterances
)
```

`UtteranceDiff` objects include all diff information inside the `diff` attribute (which is a list of `Utterance` objects). Each of our `Utterance` objects inside `UtteranceDiff.diff` now contain a populated `diff_tag` attribute, where:

- `diff_tag='+'` indicates the utterance exists in the remote instance *only*.
- `diff_tag='-'` indicates the utterance exists in the local instance *only*.
- `diff_tag=' '` indicates the utterance exists in both the local and remote instances.

After initializing an `UtteranceDiff` object we can get all utterances with each diff tag like so:

```python
# all utterances that exist only in remote
diff.get_tag("+")

# all utterances that exist only in local
diff.get_tag("-")

# all utterances that exist in both local and remote
diff.get_tag(" ")
```

These can be investigated if needed. Once we're happy with our understanding of the issues we can resolve them by executing a synchronization by running the `SemanticRouter._execute_sync_strategy` method:

```python
sr._execute_sync_strategy(sync_mode="local")
```

Once complete, we can confirm that our local and remote instances are synchronized by running `sr.is_synced()`:

```python
sr.is_synced()
```

If the above returns `True` we are now synchronized!

```
                  .=                
                 :%%*               
                -%%%%#              
               =%%%%%%#.            
              +%%%%%%%+             
             *%%%%%%%=              
           .#%%%%%%%-               
          .#%%%%%%%: -%:            
         :%%%%%%%#. =%%%=           
        -%%%%%%%#  *%%%%%+          
       =%%%%%%%*  -%%%%%%%*         
      .-------:    -%%%%%%%#        
:*****************+ :%%%%%%%#.      
-%%%%%%%%%%%%%%%%%%%* .#%%%%%%%:     
=%%%%%%%%%%%%%%%%%%%%%#..#%%%%%%%-    
+%%%%%%%%%%%%%%%%%%%%%%%#. *%%%%%%%=   
                         +%%%%%%%+  
                          =#######+ 
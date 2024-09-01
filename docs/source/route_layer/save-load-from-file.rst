Route Layers from File
======================

Route layers can be saved to and loaded from files. This can be useful if
we want to save a route layer to a file for later use, or if we want to
load a route layer from a file.

We can save and load route layers to/from YAML or JSON files. For JSON we
do:

.. code:: python

    # save to JSON
    rl.to_json("layer.json")
    # load from JSON
    new_rl = RouteLayer.from_json("layer.json")

For YAML we do:

.. code:: python

    # save to YAML
    rl.to_yaml("layer.yaml")
    # load from YAML
    new_rl = RouteLayer.from_yaml("layer.yaml")

The saved files contain all the information needed to initialize new
route layers. If we are using a remote index, we can use the
`sync features`_ to keep the route layer in sync with the index.

.. _sync features: sync.html

Full Example
---------------

|Open In Colab| |Open nbviewer|

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb
.. |Open nbviewer| image:: https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg
   :target: https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/01-save-load-from-file.ipynb

Here we will show how to save routers to YAML or JSON files, and how to
load a route layer from file.

We start by installing the library:

.. code:: none

    !pip install -qU semantic-router

Define Route
------------

First let's create a list of routes:

.. code:: python

    from semantic_router import Route
    
    politics = Route(
        name="politics",
        utterances=[
            "isn't politics the best thing ever",
            "why don't you tell me about your political opinions",
            "don't you just love the president" "don't you just hate the president",
            "they're going to destroy this country!",
            "they will save the country!",
        ],
    )
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


We define a route layer using these routes and using the Cohere encoder.

.. code:: ipython3

    import os
    from getpass import getpass
    from semantic_router import RouteLayer
    from semantic_router.encoders import CohereEncoder
    
    # dashboard.cohere.ai
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass(
        "Enter Cohere API Key: "
    )
    
    encoder = CohereEncoder()
    
    rl = RouteLayer(encoder=encoder, routes=routes)


.. parsed-literal::

    [32m2024-05-07 15:03:35 INFO semantic_router.utils.logger local[0m


Test Route
----------

.. code:: ipython3

    rl("isn't politics the best thing ever")




.. parsed-literal::

    RouteChoice(name='politics', function_call=None, similarity_score=None)



.. code:: ipython3

    rl("how's the weather today?")




.. parsed-literal::

    RouteChoice(name='chitchat', function_call=None, similarity_score=None)



Save To JSON
------------

To save our route layer we call the ``to_json`` method:

.. code:: ipython3

    rl.to_json("layer.json")


.. parsed-literal::

    [32m2024-05-07 15:03:37 INFO semantic_router.utils.logger Saving route config to layer.json[0m


Loading from JSON
-----------------

We can view the router file we just saved to see what information is
stored.

.. code:: ipython3

    import json
    
    with open("layer.json", "r") as f:
        layer_json = json.load(f)
    
    print(layer_json)


.. parsed-literal::

    {'encoder_type': 'cohere', 'encoder_name': 'embed-english-v3.0', 'routes': [{'name': 'politics', 'utterances': ["isn't politics the best thing ever", "why don't you tell me about your political opinions", "don't you just love the presidentdon't you just hate the president", "they're going to destroy this country!", 'they will save the country!'], 'description': None, 'function_schemas': None, 'llm': None, 'score_threshold': 0.3}, {'name': 'chitchat', 'utterances': ["how's the weather today?", 'how are things going?', 'lovely weather today', 'the weather is horrendous', "let's go to the chippy"], 'description': None, 'function_schemas': None, 'llm': None, 'score_threshold': 0.3}]}


It tells us our encoder type, encoder name, and routes. This is
everything we need to initialize a new router. To do so, we use the
``from_json`` method.

.. code:: ipython3

    rl = RouteLayer.from_json("layer.json")


.. parsed-literal::

    [32m2024-05-07 15:03:37 INFO semantic_router.utils.logger Loading route config from layer.json[0m
    [32m2024-05-07 15:03:37 INFO semantic_router.utils.logger local[0m


We can confirm that our layer has been initialized with the expected
attributes by viewing the ``RouteLayer`` object:

.. code:: ipython3

    print(
        f"""{rl.encoder.type=}
    {rl.encoder.name=}
    {rl.routes=}"""
    )


.. parsed-literal::

    rl.encoder.type='cohere'
    rl.encoder.name='embed-english-v3.0'
    rl.routes=[Route(name='politics', utterances=["isn't politics the best thing ever", "why don't you tell me about your political opinions", "don't you just love the presidentdon't you just hate the president", "they're going to destroy this country!", 'they will save the country!'], description=None, function_schemas=None, llm=None, score_threshold=0.3), Route(name='chitchat', utterances=["how's the weather today?", 'how are things going?', 'lovely weather today', 'the weather is horrendous', "let's go to the chippy"], description=None, function_schemas=None, llm=None, score_threshold=0.3)]


--------------

Test Route Again
----------------

.. code:: ipython3

    rl("isn't politics the best thing ever")




.. parsed-literal::

    RouteChoice(name='politics', function_call=None, similarity_score=None)



.. code:: ipython3

    rl("how's the weather today?")




.. parsed-literal::

    RouteChoice(name='chitchat', function_call=None, similarity_score=None)



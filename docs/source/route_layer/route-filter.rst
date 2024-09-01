Semantic Router Filter
======================

We can filter the routes that the `RouteLayer` considers when making a
classification. This can be useful if we want to restrict the scope of
possible routes based on some context.

For example, we may have a route layer with several routes, `politics`, 
`weather`, `chitchat`, etc. We may want to restrict the scope of the 
classification to only consider the `chitchat` route. We can do this by
passing a `route_filter` argument to our `RouteLayer` calls like so:

.. code:: python

    rl("don't you love politics?", route_filter=["chitchat"])

In this case, the `RouteLayer` will only consider the `chitchat` route
for the classification.


Full Example
------------

|Open In Colab| |Open nbviewer|

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/09-route-filter.ipynb
.. |Open nbviewer| image:: https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg
   :target: https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/00-introduction.ipynb

We start by installing the library:

.. code:: ipython3

    !pip install -qU semantic-router

We start by defining a dictionary mapping routes to example phrases that
should trigger those routes.

.. code:: ipython3

    from semantic_router import Route
    
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


Let's define another for good measure:

.. code:: ipython3

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

Now we initialize our embedding model:

.. code:: ipython3

    import os
    from getpass import getpass
    from semantic_router.encoders import CohereEncoder, OpenAIEncoder
    
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass(
        "Enter Cohere API Key: "
    )
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    #     "Enter OpenAI API Key: "
    # )
    
    encoder = CohereEncoder()
    # encoder = OpenAIEncoder()

Now we define the ``RouteLayer``. When called, the route layer will
consume text (a query) and output the category (``Route``) it belongs to
— to initialize a ``RouteLayer`` we need our ``encoder`` model and a
list of ``routes``.

.. code:: ipython3

    from semantic_router.layer import RouteLayer
    
    rl = RouteLayer(encoder=encoder, routes=routes)


.. parsed-literal::

    [32m2024-05-07 16:02:43 INFO semantic_router.utils.logger local[0m


Now we can test it:

.. code:: ipython3

    rl("don't you love politics?")




.. parsed-literal::

    RouteChoice(name='politics', function_call=None, similarity_score=None)



.. code:: ipython3

    rl("how's the weather today?")




.. parsed-literal::

    RouteChoice(name='chitchat', function_call=None, similarity_score=None)



Both are classified accurately, what if we send a query that is
unrelated to our existing ``Route`` objects?

.. code:: ipython3

    rl("I'm interested in learning about llama 2")




.. parsed-literal::

    RouteChoice(name=None, function_call=None, similarity_score=None)



In this case, we return ``None`` because no matches were identified.

Demonstrating the Filter Feature
--------------------------------

Now, let's demonstrate the filter feature. We can specify a subset of
routes to consider when making a classification. This can be useful if
we want to restrict the scope of possible routes based on some context.

For example, let's say we only want to consider the “chitchat” route for
a particular query:

.. code:: ipython3

    rl("don't you love politics?", route_filter=["chitchat"])




.. parsed-literal::

    RouteChoice(name='chitchat', function_call=None, similarity_score=None)



Even though the query might be more related to the “politics” route, it
will be classified as “chitchat” because we’ve restricted the routes to
consider.

Similarly, we can restrict it to the “politics” route:

.. code:: ipython3

    rl("how's the weather today?", route_filter=["politics"])




.. parsed-literal::

    RouteChoice(name=None, function_call=None, similarity_score=None)



In this case, it will return None because the query doesn’t match the
“politics” route well enough to pass the threshold.

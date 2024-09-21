Route Layer
===========

The `RouteLayer` is the main class of the semantic router. It is responsible
for making decisions about which route to take based on an input utterance.
A `RouteLayer` consists of an `encoder`, an `index`, and a list of `routes`.
Route layers that include dynamic routes (i.e. routes that can generate dynamic
decision outputs) also include an `llm`.

.. toctree::
   :hidden:
   :maxdepth: 2
   :glob:

   route_layer/*

To use a `RouteLayer` we first need some `routes`. We can initialize them like
so:

.. code-block:: python

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

We initialize an encoder — there are many options available here, from local
to API-based. For now we'll use the `OpenAIEncoder`.

.. code-block:: python

    import os
    from semantic_router.encoders import OpenAIEncoder

    os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

    encoder = OpenAIEncoder()

Now we define the `RouteLayer`. When called, the route layer will consume text
(a query) and output the category (`Route`) it belongs to — to initialize a
`RouteLayer` we need our `encoder` model and a list of `routes`.

.. code-block:: python

    from semantic_router.layer import RouteLayer

    rl = RouteLayer(encoder=encoder, routes=routes)

Now we can call the `RouteLayer` with an input query:

.. code-block:: python

    rl("don't you love politics?")

.. code-block:: none

    [Out]: RouteChoice(name='politics', function_call=None, similarity_score=None)

The output is a `RouteChoice` object, which contains the name of the route,
the function call (if any), and the similarity score that triggered the route
choice.

We can try another query:

.. code-block:: python

    rl("how's the weather today?")

.. code-block:: none

    [Out]: RouteChoice(name='chitchat', function_call=None, similarity_score=None)

Both are classified accurately, what if we send a query that is unrelated to
our existing Route objects?

.. code-block:: python

    rl("I'm interested in learning about llama 3")

.. code-block:: none

    [Out]: RouteChoice(name=None, function_call=None, similarity_score=None)

In this case, the `RouteLayer` is unable to find a route that matches the
input query and so returns a `RouteChoice` with `name=None`.

We can also retrieve multiple routes with their associated score using
`retrieve_multiple_routes`:

.. code-block:: python

    rl.retrieve_multiple_routes("Hi! How are you doing in politics??")

.. code-block:: none

    [Out]: [RouteChoice(name='politics', function_call=None, similarity_score=0.859),
            RouteChoice(name='chitchat', function_call=None, similarity_score=0.835)]

If `retrieve_multiple_routes` is called with a query that does not match any
routes, it will return an empty list:

.. code-block:: python

    rl.retrieve_multiple_routes("I'm interested in learning about llama 3")

.. code-block:: none

    [Out]: []

You can find an introductory notebook for the [route layer here](https://github.com/aurelio-labs/semantic-router/blob/main/docs/00-introduction.ipynb).

Route Threshold Optimization
============================

Route score thresholds are what defines whether a route should be
chosen. If the score we identify for any given route is higher than the
``Route.score_threshold`` it passes, otherwise it does not and *either*
another route is chosen, or we return *no* route.

Given that this one ``score_threshold`` parameter can define the choice
of a route, it's important to get it right — but it's incredibly
inefficient to do so manually. Instead, we can use the ``fit`` and
``evaluate`` methods of our ``RouteLayer``. All we must do is pass a
smaller number of *(utterance, target route)* examples to our methods,
and with ``fit`` we will often see dramatically improved performance
within seconds.


Full Example
------------

|Open In Colab| |Open nbviewer|

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/06-threshold-optimization.ipynb
.. |Open nbviewer| image:: https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg
   :target: https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/06-threshold-optimization


.. code:: ipython3

    !pip install -qU "semantic-router[local]"

Define RouteLayer
-----------------

As usual we will define our ``RouteLayer``. The ``RouteLayer`` requires
just ``routes`` and an ``encoder``. If using dynamic routes you must
also define an ``llm`` (or use the OpenAI default).

We will start by defining four routes; *politics*, *chitchat*,
*mathematics*, and *biology*.

.. code:: ipython3

    from semantic_router import Route
    
    # we could use this as a guide for our chatbot to avoid political conversations
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
    
    # this could be used as an indicator to our chatbot to switch to a more
    # conversational prompt
    chitchat = Route(
        name="chitchat",
        utterances=[
            "Did you watch the game last night?",
            "what's your favorite type of music?",
            "Have you read any good books lately?",
            "nice weather we're having",
            "Do you have any plans for the weekend?",
        ],
    )
    
    # we can use this to switch to an agent with more math tools, prompting, and LLMs
    mathematics = Route(
        name="mathematics",
        utterances=[
            "can you explain the concept of a derivative?",
            "What is the formula for the area of a triangle?",
            "how do you solve a system of linear equations?",
            "What is the concept of a prime number?",
            "Can you explain the Pythagorean theorem?",
        ],
    )
    
    # we can use this to switch to an agent with more biology knowledge
    biology = Route(
        name="biology",
        utterances=[
            "what is the process of osmosis?",
            "can you explain the structure of a cell?",
            "What is the role of RNA?",
            "What is genetic mutation?",
            "Can you explain the process of photosynthesis?",
        ],
    )
    
    # we place all of our decisions together into single list
    routes = [politics, chitchat, mathematics, biology]


.. parsed-literal::

    c:\Users\Siraj\Documents\Personal\Work\Aurelio\Virtual Environments\semantic_router_3\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


For our encoder we will use the local ``HuggingFaceEncoder``. Other
popular encoders include ``CohereEncoder``, ``FastEmbedEncoder``,
``OpenAIEncoder``, and ``AzureOpenAIEncoder``.

.. code:: ipython3

    from semantic_router.encoders import HuggingFaceEncoder
    
    encoder = HuggingFaceEncoder()

Now we initialize our ``RouteLayer``.

.. code:: ipython3

    from semantic_router.layer import RouteLayer
    
    rl = RouteLayer(encoder=encoder, routes=routes)


.. parsed-literal::

    [32m2024-05-07 15:53:24 INFO semantic_router.utils.logger local[0m


By default, we should get reasonable performance:

.. code:: ipython3

    for utterance in [
        "don't you love politics?",
        "how's the weather today?",
        "What's DNA?",
        "I'm interested in learning about llama 2",
    ]:
        print(f"{utterance} -> {rl(utterance).name}")


.. parsed-literal::

    don't you love politics? -> politics
    how's the weather today? -> chitchat
    What's DNA? -> biology
    I'm interested in learning about llama 2 -> None


We can evaluate the performance of our route layer using the
``evaluate`` method. All we need is to pass a list of utterances and
target route labels:

.. code:: ipython3

    test_data = [
        ("don't you love politics?", "politics"),
        ("how's the weather today?", "chitchat"),
        ("What's DNA?", "biology"),
        ("I'm interested in learning about llama 2", None),
    ]
    
    # unpack the test data
    X, y = zip(*test_data)
    
    # evaluate using the default thresholds
    accuracy = rl.evaluate(X=X, y=y)
    print(f"Accuracy: {accuracy*100:.2f}%")


.. parsed-literal::

    Generating embeddings: 100%|██████████| 1/1 [00:00<00:00, 76.91it/s]

.. parsed-literal::

    Accuracy: 100.00%


On this small subset we get perfect accuracy — but what if we try we a
larger, more robust dataset?

*Hint: try using GPT-4 or another LLM to generate some examples for your
own use-cases. The more accurate examples you provide, the better you
can expect the routes to perform on your actual use-case.*

.. code:: ipython3

    test_data = [
        # politics
        ("What's your opinion on the current government?", "politics"),
        ("Who do you think will win the next election?", "politics"),
        ("What are your thoughts on the new policy?", "politics"),
        ("How do you feel about the political situation?", "politics"),
        ("Do you agree with the president's actions?", "politics"),
        ("What's your stance on the political debate?", "politics"),
        ("How do you see the future of our country?", "politics"),
        ("What do you think about the opposition party?", "politics"),
        ("Do you believe the government is doing enough?", "politics"),
        ("What's your opinion on the political scandal?", "politics"),
        ("Do you think the new law will make a difference?", "politics"),
        ("What are your thoughts on the political reform?", "politics"),
        ("Do you agree with the government's foreign policy?", "politics"),
        # chitchat
        ("What's the weather like?", "chitchat"),
        ("It's a beautiful day today.", "chitchat"),
        ("How's your day going?", "chitchat"),
        ("It's raining cats and dogs.", "chitchat"),
        ("Let's grab a coffee.", "chitchat"),
        ("What's up?", "chitchat"),
        ("It's a bit chilly today.", "chitchat"),
        ("How's it going?", "chitchat"),
        ("Nice weather we're having.", "chitchat"),
        ("It's a bit windy today.", "chitchat"),
        ("Let's go for a walk.", "chitchat"),
        ("How's your week been?", "chitchat"),
        ("It's quite sunny today.", "chitchat"),
        ("How are you feeling?", "chitchat"),
        ("It's a bit cloudy today.", "chitchat"),
        # mathematics
        ("What is the Pythagorean theorem?", "mathematics"),
        ("Can you solve this quadratic equation?", "mathematics"),
        ("What is the derivative of x squared?", "mathematics"),
        ("Explain the concept of integration.", "mathematics"),
        ("What is the area of a circle?", "mathematics"),
        ("How do you calculate the volume of a sphere?", "mathematics"),
        ("What is the difference between a vector and a scalar?", "mathematics"),
        ("Explain the concept of a matrix.", "mathematics"),
        ("What is the Fibonacci sequence?", "mathematics"),
        ("How do you calculate permutations?", "mathematics"),
        ("What is the concept of probability?", "mathematics"),
        ("Explain the binomial theorem.", "mathematics"),
        ("What is the difference between discrete and continuous data?", "mathematics"),
        ("What is a complex number?", "mathematics"),
        ("Explain the concept of limits.", "mathematics"),
        # biology
        ("What is photosynthesis?", "biology"),
        ("Explain the process of cell division.", "biology"),
        ("What is the function of mitochondria?", "biology"),
        ("What is DNA?", "biology"),
        ("What is the difference between prokaryotic and eukaryotic cells?", "biology"),
        ("What is an ecosystem?", "biology"),
        ("Explain the theory of evolution.", "biology"),
        ("What is a species?", "biology"),
        ("What is the role of enzymes?", "biology"),
        ("What is the circulatory system?", "biology"),
        ("Explain the process of respiration.", "biology"),
        ("What is a gene?", "biology"),
        ("What is the function of the nervous system?", "biology"),
        ("What is homeostasis?", "biology"),
        ("What is the difference between a virus and a bacteria?", "biology"),
        ("What is the role of the immune system?", "biology"),
        # add some None routes to prevent excessively small thresholds
        ("What is the capital of France?", None),
        ("how many people live in the US?", None),
        ("when is the best time to visit Bali?", None),
        ("how do I learn a language", None),
        ("tell me an interesting fact", None),
        ("what is the best programming language?", None),
        ("I'm interested in learning about llama 2", None),
    ]

.. code:: ipython3

    # unpack the test data
    X, y = zip(*test_data)
    
    # evaluate using the default thresholds
    accuracy = rl.evaluate(X=X, y=y)
    print(f"Accuracy: {accuracy*100:.2f}%")


.. parsed-literal::

    Generating embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.23it/s]

.. parsed-literal::

    Accuracy: 34.85%


Ouch, that's not so good! Fortunately, we can easily improve our
performance here.

Route Layer Optimization
------------------------

Our optimization works by finding the best route thresholds for each
``Route`` in our ``RouteLayer``. We can see the current, default
thresholds by calling the ``get_thresholds`` method:

.. code:: ipython3

    route_thresholds = rl.get_thresholds()
    print("Default route thresholds:", route_thresholds)


.. parsed-literal::

    Default route thresholds: {'politics': 0.5, 'chitchat': 0.5, 'mathematics': 0.5, 'biology': 0.5}


These are all preset route threshold values. Fortunately, it’s very easy
to optimize these — we simply call the ``fit`` method and provide our
training utterances ``X``, and target route labels ``y``:

.. code:: ipython3

    # Call the fit method
    rl.fit(X=X, y=y)


.. parsed-literal::

    Generating embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.21it/s]
    Training: 100%|██████████| 500/500 [00:01<00:00, 419.45it/s, acc=0.89]


Let’s see what our new thresholds look like:

.. code:: ipython3

    route_thresholds = rl.get_thresholds()
    print("Updated route thresholds:", route_thresholds)


.. parsed-literal::

    Updated route thresholds: {'politics': 0.05050505050505051, 'chitchat': 0.32323232323232326, 'mathematics': 0.18181818181818182, 'biology': 0.21212121212121213}


These are vastly different thresholds to what we were seeing before —
it’s worth noting that *optimal* values for different encoders can vary
greatly. For example, OpenAI’s Ada 002 model, when used with our
encoders will tend to output much larger numbers in the ``0.5`` to
``0.8`` range.

After training we have a final performance of:

.. code:: ipython3

    accuracy = rl.evaluate(X=X, y=y)
    print(f"Accuracy: {accuracy*100:.2f}%")


.. parsed-literal::

    Generating embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.89it/s]


.. parsed-literal::

    Accuracy: 89.39%


That is *much* better. If we wanted to optimize this further we can
focus on adding more utterances to our existing routes, analyzing
*where* exactly our failures are, and modifying our routes around those.
This extended optimzation process is much more manual, but with it we
can continue optimizing routes to get even better performance.

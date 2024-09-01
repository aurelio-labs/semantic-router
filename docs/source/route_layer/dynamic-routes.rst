Dynamic Routes
==============

In semantic-router there are two types of routes that can be chosen.
Both routes belong to the ``Route`` object, the only difference between
them is that *static* routes return a ``Route.name`` when chosen,
whereas *dynamic* routes use an LLM call to produce parameter input
values.

For example, a *static* route will tell us if a query is talking about
mathematics by returning the route name (which could be ``"math"`` for
example). A *dynamic* route does the same thing, but it also extracts
key information from the input utterance to be used in a function
associated with that route.

For example we could provide a dynamic route with associated utterances:

.. code:: python

   "what is x to the power of y?"
   "what is 9 to the power of 4?"
   "calculate the result of base x and exponent y"
   "calculate the result of base 10 and exponent 3"
   "return x to the power of y"

and we could also provide the route with a schema outlining key features
of the function:

.. code:: python

   def power(base: float, exponent: float) -> float:
       """Raise base to the power of exponent.

       Args:
           base (float): The base number.
           exponent (float): The exponent to which the base is raised.

       Returns:
           float: The result of base raised to the power of exponent.
       """
       return base ** exponent

Then, if the users input utterance is “What is 2 to the power of 3?”,
the route will be triggered, as the input utterance is semantically
similar to the route utterances. Furthermore, the route utilizes an LLM
to identify that ``base=2`` and ``expoenent=3``. These values are
returned in such a way that they can be used in the above ``power``
function. That is, the dynamic router automates the process of calling
relevant functions from natural language inputs.

As with static routes, we must create a dynamic route before adding it
to our route layer. To make a route dynamic, we need to provide the
``function_schemas`` as a list. Each function schema provides
instructions on what a function is, so that an LLM can decide how to use
it correctly.

.. code:: ipython3

    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    
    def get_time(timezone: str) -> str:
        """Finds the current time in a specific timezone.
    
        :param timezone: The timezone to find the current time in, should
            be a valid timezone from the IANA Time Zone Database like
            "America/New_York" or "Europe/London". Do NOT put the place
            name itself like "rome", or "new york", you must provide
            the IANA format.
        :type timezone: str
        :return: The current time in the specified timezone."""
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%H:%M")

.. code:: ipython3

    get_time("America/New_York")



.. parsed-literal::

    '17:57'



To get the function schema we can use the ``get_schemas_openai`` function.

.. code:: ipython3

    from semantic_router.llms.openai import get_schemas_openai
    
    schemas = get_schemas_openai([get_time])
    schemas




.. parsed-literal::

    [{'type': 'function',
      'function': {'name': 'get_time',
       'description': 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
       'parameters': {'type': 'object',
        'properties': {'timezone': {'type': 'string',
          'description': 'The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.'}},
        'required': ['timezone']}}}]



We use this to define our dynamic route:

.. code:: ipython3

    time_route = Route(
        name="get_time",
        utterances=[
            "what is the time in new york city?",
            "what is the time in london?",
            "I live in Rome, what time is it?",
        ],
        function_schemas=schemas,
    )

Then add the new route to a route layer.

Full Example
------------

|Open In Colab| |Open nbviewer|

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb
.. |Open nbviewer| image:: https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg
   :target: https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb


Installing the Library
----------------------

.. code:: ipython3

    !pip install tzdata
    !pip install -qU semantic-router


Initializing Routes and RouteLayer
----------------------------------

Dynamic routes are treated in the same way as static routes, let's begin
by initializing a ``RouteLayer`` consisting of static routes.

**⚠️ Note: We have a fully local version of dynamic routes available
at**
`docs/05-local-execution.ipynb <https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb>`__\ **\ .
The local 05 version tends to outperform the OpenAI version we demo in
this notebook, so we'd recommend trying**
`05 <https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb>`__\ **\ !**

.. code:: ipython3

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


.. parsed-literal::

    c:\Users\Siraj\Documents\Personal\Work\Aurelio\Virtual Environments\semantic_router_3\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


We initialize our ``RouteLayer`` with our ``encoder`` and ``routes``. We
can use popular encoder APIs like ``CohereEncoder`` and
``OpenAIEncoder``, or local alternatives like ``FastEmbedEncoder``.

.. code:: ipython3

    import os
    from semantic_router import RouteLayer
    from semantic_router.encoders import OpenAIEncoder
    
    # platform.openai.com
    os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
    
    encoder = OpenAIEncoder()
    
    rl = RouteLayer(encoder=encoder, routes=routes)


We run the solely static routes layer:

.. code:: ipython3

    rl("how's the weather today?")




.. parsed-literal::

    RouteChoice(name='chitchat', function_call=None, similarity_score=None)



Creating a Dynamic Route
------------------------

As with static routes, we must create a dynamic route before adding it
to our route layer. To make a route dynamic, we need to provide the
``function_schemas`` as a list. Each function schema provides
instructions on what a function is, so that an LLM can decide how to use
it correctly.

.. code:: ipython3

    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    
    def get_time(timezone: str) -> str:
        """Finds the current time in a specific timezone.
    
        :param timezone: The timezone to find the current time in, should
            be a valid timezone from the IANA Time Zone Database like
            "America/New_York" or "Europe/London". Do NOT put the place
            name itself like "rome", or "new york", you must provide
            the IANA format.
        :type timezone: str
        :return: The current time in the specified timezone."""
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%H:%M")

.. code:: ipython3

    get_time("America/New_York")




.. parsed-literal::

    '17:57'



To get the function schema we can use the ``get_schema`` function from
the ``function_call`` module.

.. code:: ipython3

    from semantic_router.llms.openai import get_schemas_openai
    
    schemas = get_schemas_openai([get_time])
    schemas




.. parsed-literal::

    [{'type': 'function',
      'function': {'name': 'get_time',
       'description': 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
       'parameters': {'type': 'object',
        'properties': {'timezone': {'type': 'string',
          'description': 'The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.'}},
        'required': ['timezone']}}}]



We use this to define our dynamic route:

.. code:: ipython3

    time_route = Route(
        name="get_time",
        utterances=[
            "what is the time in new york city?",
            "what is the time in london?",
            "I live in Rome, what time is it?",
        ],
        function_schemas=schemas,
    )

.. code:: ipython3

    time_route.llm

Add the new route to our ``layer``:

.. code:: ipython3

    rl.add(time_route)


.. parsed-literal::

    [32m2024-05-08 01:57:56 INFO semantic_router.utils.logger Adding `get_time` route[0m


.. code:: ipython3

    time_route.llm

Now we can ask our layer a time related question to trigger our new
dynamic route.

.. code:: ipython3

    response = rl("what is the time in new york city?")
    response


.. parsed-literal::

    [33m2024-05-08 01:57:57 WARNING semantic_router.utils.logger No LLM provided for dynamic route, will use OpenAI LLM default. Ensure API key is set in OPENAI_API_KEY environment variable.[0m
    [32m2024-05-08 01:57:58 INFO semantic_router.utils.logger Function inputs: [{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}][0m




.. parsed-literal::

    RouteChoice(name='get_time', function_call=[{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}], similarity_score=None)



.. code:: ipython3

    print(response.function_call)


.. parsed-literal::

    [{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}]


.. code:: ipython3

    import json
    
    for call in response.function_call:
        if call["function_name"] == "get_time":
            args = call["arguments"]
            result = get_time(**args)
    print(result)


.. parsed-literal::

    17:57


Our dynamic route provides both the route itself *and* the input
parameters required to use the route.

Dynamic Routes with Multiple Functions
--------------------------------------

Routes can be assigned multiple functions. Then, when that particular
Route is selected by the Route Layer, a number of those functions might
be invoked due to the users utterance containing relevant information
that fits their arguments.

Let's define a Route that has multiple functions.

.. code:: ipython3

    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    
    
    # Function with one argument
    def get_time(timezone: str) -> str:
        """Finds the current time in a specific timezone.
    
        :param timezone: The timezone to find the current time in, should
            be a valid timezone from the IANA Time Zone Database like
            "America/New_York" or "Europe/London". Do NOT put the place
            name itself like "rome", or "new york", you must provide
            the IANA format.
        :type timezone: str
        :return: The current time in the specified timezone."""
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%H:%M")
    
    
    def get_time_difference(timezone1: str, timezone2: str) -> str:
        """Calculates the time difference between two timezones.
        :param timezone1: The first timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".
        :param timezone2: The second timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".
        :type timezone1: str
        :type timezone2: str
        :return: The time difference in hours between the two timezones."""
        # Get the current time in UTC
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    
        # Convert the UTC time to the specified timezones
        tz1_time = now_utc.astimezone(ZoneInfo(timezone1))
        tz2_time = now_utc.astimezone(ZoneInfo(timezone2))
    
        # Calculate the difference in offsets from UTC
        tz1_offset = tz1_time.utcoffset().total_seconds()
        tz2_offset = tz2_time.utcoffset().total_seconds()
    
        # Calculate the difference in hours
        hours_difference = (tz2_offset - tz1_offset) / 3600
    
        return f"The time difference between {timezone1} and {timezone2} is {hours_difference} hours."
    
    
    # Function with three arguments
    def convert_time(time: str, from_timezone: str, to_timezone: str) -> str:
        """Converts a specific time from one timezone to another.
        :param time: The time to convert in HH:MM format.
        :param from_timezone: The original timezone of the time, should be a valid IANA timezone.
        :param to_timezone: The target timezone for the time, should be a valid IANA timezone.
        :type time: str
        :type from_timezone: str
        :type to_timezone: str
        :return: The converted time in the target timezone.
        :raises ValueError: If the time format or timezone strings are invalid.
    
        Example:
            convert_time("12:30", "America/New_York", "Asia/Tokyo") -> "03:30"
        """
        try:
            # Use today's date to avoid historical timezone issues
            today = datetime.now().date()
            datetime_string = f"{today} {time}"
            time_obj = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M").replace(
                tzinfo=ZoneInfo(from_timezone)
            )
    
            converted_time = time_obj.astimezone(ZoneInfo(to_timezone))
    
            formatted_time = converted_time.strftime("%H:%M")
            return formatted_time
        except Exception as e:
            raise ValueError(f"Error converting time: {e}")

.. code:: ipython3

    functions = [get_time, get_time_difference, convert_time]

.. code:: ipython3

    # Generate schemas for all functions
    from semantic_router.llms.openai import get_schemas_openai
    
    schemas = get_schemas_openai(functions)
    schemas




.. parsed-literal::

    [{'type': 'function',
      'function': {'name': 'get_time',
       'description': 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
       'parameters': {'type': 'object',
        'properties': {'timezone': {'type': 'string',
          'description': 'The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.'}},
        'required': ['timezone']}}},
     {'type': 'function',
      'function': {'name': 'get_time_difference',
       'description': 'Calculates the time difference between two timezones.\n:param timezone1: The first timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".\n:param timezone2: The second timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".\n:type timezone1: str\n:type timezone2: str\n:return: The time difference in hours between the two timezones.',
       'parameters': {'type': 'object',
        'properties': {'timezone1': {'type': 'string',
          'description': 'The first timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".'},
         'timezone2': {'type': 'string',
          'description': 'The second timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".'}},
        'required': ['timezone1', 'timezone2']}}},
     {'type': 'function',
      'function': {'name': 'convert_time',
       'description': 'Converts a specific time from one timezone to another.\n:param time: The time to convert in HH:MM format.\n:param from_timezone: The original timezone of the time, should be a valid IANA timezone.\n:param to_timezone: The target timezone for the time, should be a valid IANA timezone.\n:type time: str\n:type from_timezone: str\n:type to_timezone: str\n:return: The converted time in the target timezone.\n:raises ValueError: If the time format or timezone strings are invalid.\n\nExample:\n    convert_time("12:30", "America/New_York", "Asia/Tokyo") -> "03:30"',
       'parameters': {'type': 'object',
        'properties': {'time': {'type': 'string',
          'description': 'The time to convert in HH:MM format.'},
         'from_timezone': {'type': 'string',
          'description': 'The original timezone of the time, should be a valid IANA timezone.'},
         'to_timezone': {'type': 'string',
          'description': 'The target timezone for the time, should be a valid IANA timezone.'}},
        'required': ['time', 'from_timezone', 'to_timezone']}}}]



.. code:: ipython3

    # Define the dynamic route with multiple functions
    multi_function_route = Route(
        name="timezone_management",
        utterances=[
            # Utterances for get_time function
            "what is the time in New York?",
            "current time in Berlin?",
            "tell me the time in Moscow right now",
            "can you show me the current time in Tokyo?",
            "please provide the current time in London",
            # Utterances for get_time_difference function
            "how many hours ahead is Tokyo from London?",
            "time difference between Sydney and Cairo",
            "what's the time gap between Los Angeles and New York?",
            "how much time difference is there between Paris and Sydney?",
            "calculate the time difference between Dubai and Toronto",
            # Utterances for convert_time function
            "convert 15:00 from New York time to Berlin time",
            "change 09:00 from Paris time to Moscow time",
            "adjust 20:00 from Rome time to London time",
            "convert 12:00 from Madrid time to Chicago time",
            "change 18:00 from Beijing time to Los Angeles time"
            # All three functions
            "What is the time in Seattle? What is the time difference between Mumbai and Tokyo? What is 5:53 Toronto time in Sydney time?",
        ],
        function_schemas=schemas,
    )

.. code:: ipython3

    routes = [politics, chitchat, multi_function_route]

.. code:: ipython3

    rl2 = RouteLayer(encoder=encoder, routes=routes)


.. parsed-literal::

    [32m2024-05-08 01:57:58 INFO semantic_router.utils.logger local[0m


Function to Parse Route Layer Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def parse_response(response: str):
        for call in response.function_call:
            args = call["arguments"]
            if call["function_name"] == "get_time":
                result = get_time(**args)
                print(result)
            if call["function_name"] == "get_time_difference":
                result = get_time_difference(**args)
                print(result)
            if call["function_name"] == "convert_time":
                result = convert_time(**args)
                print(result)

Checking that Politics Non-Dynamic Route Still Works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    response = rl2("What is your political leaning?")
    response




.. parsed-literal::

    RouteChoice(name='politics', function_call=None, similarity_score=None)



Checking that Chitchat Non-Dynamic Route Still Works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    response = rl2("Hello bot, how are you today?")
    response




.. parsed-literal::

    RouteChoice(name='chitchat', function_call=None, similarity_score=None)



Testing the ``multi_function_route`` - The ``get_time`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    response = rl2("what is the time in New York?")
    response


.. parsed-literal::

    [33m2024-05-08 01:58:00 WARNING semantic_router.utils.logger No LLM provided for dynamic route, will use OpenAI LLM default. Ensure API key is set in OPENAI_API_KEY environment variable.[0m
    [32m2024-05-08 01:58:01 INFO semantic_router.utils.logger Function inputs: [{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}][0m




.. parsed-literal::

    RouteChoice(name='timezone_management', function_call=[{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}], similarity_score=None)



.. code:: ipython3

    parse_response(response)


.. parsed-literal::

    17:58


Testing the ``multi_function_route`` - The ``get_time_difference`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    response = rl2("What is the time difference between Los Angeles and Istanbul?")
    response


.. parsed-literal::

    [32m2024-05-08 01:58:02 INFO semantic_router.utils.logger Function inputs: [{'function_name': 'get_time_difference', 'arguments': {'timezone1': 'America/Los_Angeles', 'timezone2': 'Europe/Istanbul'}}][0m




.. parsed-literal::

    RouteChoice(name='timezone_management', function_call=[{'function_name': 'get_time_difference', 'arguments': {'timezone1': 'America/Los_Angeles', 'timezone2': 'Europe/Istanbul'}}], similarity_score=None)



.. code:: ipython3

    parse_response(response)


.. parsed-literal::

    The time difference between America/Los_Angeles and Europe/Istanbul is 10.0 hours.


Testing the ``multi_function_route`` - The ``convert_time`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    response = rl2("What is 23:02 Dubai time in Tokyo time? Please and thank you.")
    response


.. parsed-literal::

    [32m2024-05-08 01:58:04 INFO semantic_router.utils.logger Function inputs: [{'function_name': 'convert_time', 'arguments': {'time': '23:02', 'from_timezone': 'Asia/Dubai', 'to_timezone': 'Asia/Tokyo'}}][0m




.. parsed-literal::

    RouteChoice(name='timezone_management', function_call=[{'function_name': 'convert_time', 'arguments': {'time': '23:02', 'from_timezone': 'Asia/Dubai', 'to_timezone': 'Asia/Tokyo'}}], similarity_score=None)



.. code:: ipython3

    parse_response(response)


.. parsed-literal::

    04:02


The Cool Bit - Testing ``multi_function_route`` - Multiple Functions at Once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    response = rl2(
        """
        What is the time in Prague?
        What is the time difference between Frankfurt and Beijing?
        What is 5:53 Lisbon time in Bangkok time?
    """
    )


.. parsed-literal::

    [32m2024-05-08 01:58:07 INFO semantic_router.utils.logger Function inputs: [{'function_name': 'get_time', 'arguments': {'timezone': 'Europe/Prague'}}, {'function_name': 'get_time_difference', 'arguments': {'timezone1': 'Europe/Berlin', 'timezone2': 'Asia/Shanghai'}}, {'function_name': 'convert_time', 'arguments': {'time': '05:53', 'from_timezone': 'Europe/Lisbon', 'to_timezone': 'Asia/Bangkok'}}][0m


.. code:: ipython3

    response




.. parsed-literal::

    RouteChoice(name='timezone_management', function_call=[{'function_name': 'get_time', 'arguments': {'timezone': 'Europe/Prague'}}, {'function_name': 'get_time_difference', 'arguments': {'timezone1': 'Europe/Berlin', 'timezone2': 'Asia/Shanghai'}}, {'function_name': 'convert_time', 'arguments': {'time': '05:53', 'from_timezone': 'Europe/Lisbon', 'to_timezone': 'Asia/Bangkok'}}], similarity_score=None)



.. code:: ipython3

    parse_response(response)


.. parsed-literal::

    23:58
    The time difference between Europe/Berlin and Asia/Shanghai is 6.0 hours.
    11:53


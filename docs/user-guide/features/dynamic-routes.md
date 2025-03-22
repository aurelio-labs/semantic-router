In semantic-router there are two types of routes that can be chosen. Both routes belong to the `Route` object, the only difference between them is that *static* routes return a `Route.name` when chosen, whereas *dynamic* routes use an LLM call to produce parameter input values.

For example, a *static* route will tell us if a query is talking about mathematics by returning the route name (which could be `"math"` for example). A *dynamic* route does the same thing, but it also extracts key information from the input utterance to be used in a function associated with that route.

For example we could provide a dynamic route with associated utterances:

```python
"what is x to the power of y?"
"what is 9 to the power of 4?"
"calculate the result of base x and exponent y"
"calculate the result of base 10 and exponent 3"
"return x to the power of y"
```

and we could also provide the route with a schema outlining key features of the function:

```python
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent.

    Args:
        base (float): The base number.
        exponent (float): The exponent to which the base is raised.

    Returns:
        float: The result of base raised to the power of exponent.
    """
    return base ** exponent
```

Then, if the user's input utterance is "What is 2 to the power of 3?", the route will be triggered, as the input utterance is semantically similar to the route utterances. Furthermore, the route utilizes an LLM to identify that `base=2` and `exponent=3`. These values are returned in such a way that they can be used in the above `power` function. That is, the dynamic router automates the process of calling relevant functions from natural language inputs.

As with static routes, we must create a dynamic route before adding it to our router. To make a route dynamic, we need to provide the `function_schemas` as a list. Each function schema provides instructions on what a function is, so that an LLM can decide how to use it correctly.

```python
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
```

```python
get_time("America/New_York")
```

To get the function schema we can use the `get_schemas_openai` function.

```python
from semantic_router.llms.openai import get_schemas_openai

schemas = get_schemas_openai([get_time])
schemas
```

We use this to define our dynamic route:

```python
from semantic_router import Route

time_route = Route(
    name="get_time",
    utterances=[
        "what is the time in new york city?",
        "what is the time in london?",
        "I live in Rome, what time is it?",
    ],
    function_schemas=schemas,
)
```

Then add the new route to a router.

## Full Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb)

### Installing the Library

```python
!pip install tzdata
!pip install -qU semantic-router>=0.1.5
```

### Initializing Routes and SemanticRouter

Dynamic routes are treated in the same way as static routes, let's begin by initializing a `SemanticRouter` consisting of static routes.

**⚠️ Note: We have a fully local version of dynamic routes available at [docs/05-local-execution.ipynb](https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb). The local version tends to outperform the OpenAI version we demo in this document, so we'd recommend trying [05-local-execution.ipynb](https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb)!**

```python
from semantic_router import Route

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president", "don't you just hate the president",
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
```

We initialize our `SemanticRouter` with our `encoder` and `routes`. We can use popular encoder APIs like `CohereEncoder` and `OpenAIEncoder`, or local alternatives like `FastEmbedEncoder`.

```python
import os
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import OpenAIEncoder

# platform.openai.com
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

encoder = OpenAIEncoder()

sr = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

We run the router with only static routes:

```python
sr("how's the weather today?")
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

### Creating a Dynamic Route

As with static routes, we must create a dynamic route before adding it to our router. To make a route dynamic, we need to provide the `function_schemas` as a list. Each function schema provides instructions on what a function is, so that an LLM can decide how to use it correctly.

```python
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
```

```python
get_time("America/New_York")
```

```
'17:57'
```

To get the function schema we can use the `get_schemas_openai` function.

```python
from semantic_router.llms.openai import get_schemas_openai

schemas = get_schemas_openai([get_time])
schemas
```

We use this to define our dynamic route:

```python
time_route = Route(
    name="get_time",
    utterances=[
        "what is the time in new york city?",
        "what is the time in london?",
        "I live in Rome, what time is it?",
    ],
    function_schemas=schemas,
)
```

Add the new route to our router:

```python
sr.add(time_route)
```

Now we can ask our router a time related question to trigger our new dynamic route.

```python
response = sr("what is the time in new york city?")
response
```

```
RouteChoice(name='get_time', function_call=[{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}], similarity_score=None)
```

```python
print(response.function_call)
```

```
[{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}]
```

```python
import json

for call in response.function_call:
    if call["function_name"] == "get_time":
        args = call["arguments"]
        result = get_time(**args)
print(result)
```

```
17:57
```

Our dynamic route provides both the route itself *and* the input parameters required to use the route.

### Dynamic Routes with Multiple Functions

Routes can be assigned multiple functions. Then, when that particular Route is selected by the Router, a number of those functions might be invoked due to the users utterance containing relevant information that fits their arguments.

Let's define a Route that has multiple functions.

```python
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
```

```python
functions = [get_time, get_time_difference, convert_time]
```

```python
# Generate schemas for all functions
from semantic_router.llms.openai import get_schemas_openai

schemas = get_schemas_openai(functions)
```

```python
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
```

```python
routes = [politics, chitchat, multi_function_route]
```

```python
sr2 = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

#### Function to Parse Router Responses

```python
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
```

#### Testing the `multi_function_route` - Multiple Functions at Once

```python
response = sr2(
    """
    What is the time in Prague?
    What is the time difference between Frankfurt and Beijing?
    What is 5:53 Lisbon time in Bangkok time?
"""
)
```

```python
response
```

```
RouteChoice(name='timezone_management', function_call=[{'function_name': 'get_time', 'arguments': {'timezone': 'Europe/Prague'}}, {'function_name': 'get_time_difference', 'arguments': {'timezone1': 'Europe/Berlin', 'timezone2': 'Asia/Shanghai'}}, {'function_name': 'convert_time', 'arguments': {'time': '05:53', 'from_timezone': 'Europe/Lisbon', 'to_timezone': 'Asia/Bangkok'}}], similarity_score=None)
```

```python
parse_response(response)
```

```
23:58
The time difference between Europe/Berlin and Asia/Shanghai is 6.0 hours.
11:53
``` 
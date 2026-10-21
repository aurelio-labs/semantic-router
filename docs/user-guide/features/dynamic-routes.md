> **You're reading the docs for Semantic Router 0.x.** Version 1.0 removes dynamic routes. The [migration guide](https://docs.aurelio.ai/docs/semantic-router/v1/user-guide/guides/migration-to-v1#dynamic-routes-and-the-llm-classes-are-gone) shows what to do instead.

There are two kinds of route. A *static* route tells you which route matched — you get back its name, like `"math"`. A *dynamic* route does that too, but it also pulls the details out of the query and hands them to a function.

Say you have a route with utterances like these:

```python
"what is x to the power of y?"
"what is 9 to the power of 4?"
"calculate the result of base x and exponent y"
"return x to the power of y"
```

and you attach a function to it:

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

Now ask "What is 2 to the power of 3?". The route matches on meaning, same as any route. Then an LLM reads the query and works out that `base=2` and `exponent=3`, in a shape you can pass straight to `power()`. That's the whole idea: natural language in, a ready-to-call function out.

To make a route dynamic, give it `function_schemas` — a list describing each function so the LLM knows what it does and what it needs. The rest of this page walks through it.

## Full example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb)
[![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb)

```python
!pip install tzdata
!pip install -qU semantic-router>=0.1.5
```

> **Prefer to run this locally?** There's a fully local version of dynamic routes in [05-local-execution.ipynb](https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb). It tends to outperform the OpenAI version shown here, so it's worth a try.

### Start with static routes

Dynamic routes sit alongside static ones, so let's build a router with two static routes first.

```python
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

routes = [politics, chitchat]
```

Pick an encoder. Any will do — `CohereEncoder`, `OpenAIEncoder`, or a local one like `FastEmbedEncoder`.

```python
import os
from semantic_router import SemanticRouter
from semantic_router.encoders import OpenAIEncoder

# platform.openai.com
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

encoder = OpenAIEncoder()

sr = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

Check it works with static routes only:

```python
sr("how's the weather today?")
```

```
RouteChoice(name='chitchat', function_call=None, similarity_score=None)
```

### Add a dynamic route

Here's the function we want the route to call — it returns the current time in a timezone. The docstring matters: the LLM reads it to learn what the `timezone` argument should look like.

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

Generate the schema from the function with `get_schemas_openai`:

```python
from semantic_router.llms.openai import get_schemas_openai

schemas = get_schemas_openai([get_time])
schemas
```

Then define the route, passing the schema in:

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

Add it to the router:

```python
sr.add(time_route)
```

Now ask a time question:

```python
response = sr("what is the time in new york city?")
response
```

```
RouteChoice(name='get_time', function_call=[{'function_name': 'get_time', 'arguments': {'timezone': 'America/New_York'}}], similarity_score=None)
```

The route matched *and* the LLM filled in the arguments. `function_call` holds everything you need to run it:

```python
for call in response.function_call:
    if call["function_name"] == "get_time":
        result = get_time(**call["arguments"])
print(result)
```

```
17:57
```

### One route, several functions

A route can carry more than one function. When it matches, the LLM decides which functions the query needs — and it can call several at once if the query asks for several things.

Let's give one route three timezone tools:

```python
from datetime import datetime, timedelta
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


def get_time_difference(timezone1: str, timezone2: str) -> str:
    """Calculates the time difference between two timezones.
    :param timezone1: The first timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".
    :param timezone2: The second timezone, should be a valid timezone from the IANA Time Zone Database like "America/New_York" or "Europe/London".
    :type timezone1: str
    :type timezone2: str
    :return: The time difference in hours between the two timezones."""
    now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

    tz1_time = now_utc.astimezone(ZoneInfo(timezone1))
    tz2_time = now_utc.astimezone(ZoneInfo(timezone2))

    tz1_offset = tz1_time.utcoffset().total_seconds()
    tz2_offset = tz2_time.utcoffset().total_seconds()

    hours_difference = (tz2_offset - tz1_offset) / 3600

    return f"The time difference between {timezone1} and {timezone2} is {hours_difference} hours."


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
        # use today's date to avoid historical timezone issues
        today = datetime.now().date()
        datetime_string = f"{today} {time}"
        time_obj = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M").replace(
            tzinfo=ZoneInfo(from_timezone)
        )

        converted_time = time_obj.astimezone(ZoneInfo(to_timezone))
        return converted_time.strftime("%H:%M")
    except Exception as e:
        raise ValueError(f"Error converting time: {e}")
```

Generate schemas for all three and build the route. Give it utterances that cover each function, plus a couple that hit several at once:

```python
from semantic_router.llms.openai import get_schemas_openai

functions = [get_time, get_time_difference, convert_time]
schemas = get_schemas_openai(functions)

multi_function_route = Route(
    name="timezone_management",
    utterances=[
        # get_time
        "what is the time in New York?",
        "current time in Berlin?",
        "tell me the time in Moscow right now",
        "can you show me the current time in Tokyo?",
        "please provide the current time in London",
        # get_time_difference
        "how many hours ahead is Tokyo from London?",
        "time difference between Sydney and Cairo",
        "what's the time gap between Los Angeles and New York?",
        "how much time difference is there between Paris and Sydney?",
        "calculate the time difference between Dubai and Toronto",
        # convert_time
        "convert 15:00 from New York time to Berlin time",
        "change 09:00 from Paris time to Moscow time",
        "adjust 20:00 from Rome time to London time",
        "convert 12:00 from Madrid time to Chicago time",
        "change 18:00 from Beijing time to Los Angeles time",
        # all three
        "What is the time in Seattle? What is the time difference between Mumbai and Tokyo? What is 5:53 Toronto time in Sydney time?",
    ],
    function_schemas=schemas,
)

routes = [politics, chitchat, multi_function_route]
sr2 = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
```

A small dispatcher runs whatever the router asks for:

```python
def parse_response(response):
    for call in response.function_call:
        args = call["arguments"]
        if call["function_name"] == "get_time":
            print(get_time(**args))
        if call["function_name"] == "get_time_difference":
            print(get_time_difference(**args))
        if call["function_name"] == "convert_time":
            print(convert_time(**args))
```

Now ask for three things in one go:

```python
response = sr2(
    """
    What is the time in Prague?
    What is the time difference between Frankfurt and Beijing?
    What is 5:53 Lisbon time in Bangkok time?
"""
)
response
```

```
RouteChoice(name='timezone_management', function_call=[{'function_name': 'get_time', 'arguments': {'timezone': 'Europe/Prague'}}, {'function_name': 'get_time_difference', 'arguments': {'timezone1': 'Europe/Berlin', 'timezone2': 'Asia/Shanghai'}}, {'function_name': 'convert_time', 'arguments': {'time': '05:53', 'from_timezone': 'Europe/Lisbon', 'to_timezone': 'Asia/Bangkok'}}], similarity_score=None)
```

All three functions, each with the right arguments. Run them:

```python
parse_response(response)
```

```
23:58
The time difference between Europe/Berlin and Asia/Shanghai is 6.0 hours.
11:53
```

import inspect

from semantic_router.utils.function_call import FunctionSchema


def scrape_webpage(url: str, name: str = "test") -> str:
    """Provides access to web scraping. You can use this tool to scrape a webpage.
    Many webpages may return no information due to JS or adblock issues, if this
    happens, you must use a different URL.
    """
    return "hello there"


def test_function_schema():
    schema = FunctionSchema(scrape_webpage)
    assert schema.name == scrape_webpage.__name__
    assert schema.description == str(inspect.getdoc(scrape_webpage))
    assert schema.signature == str(inspect.signature(scrape_webpage))
    assert schema.output == str(inspect.signature(scrape_webpage).return_annotation)
    assert len(schema.parameters) == 2


def test_ollama_function_schema():
    schema = FunctionSchema(scrape_webpage)
    ollama_schema = schema.to_ollama()
    assert ollama_schema["type"] == "function"
    assert ollama_schema["function"]["name"] == schema.name
    assert ollama_schema["function"]["description"] == schema.description
    assert ollama_schema["function"]["parameters"]["type"] == "object"
    assert (
        ollama_schema["function"]["parameters"]["properties"]["url"]["type"] == "string"
    )
    assert (
        ollama_schema["function"]["parameters"]["properties"]["name"]["type"]
        == "string"
    )
    assert (
        ollama_schema["function"]["parameters"]["properties"]["url"]["description"]
        is None
    )
    assert (
        ollama_schema["function"]["parameters"]["properties"]["name"]["description"]
        is None
    )
    assert ollama_schema["function"]["parameters"]["required"] == ["name"]

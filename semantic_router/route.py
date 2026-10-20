from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Route(BaseModel):
    """A route: one decision a router can make, and what tells it when to make it.

    :param name: The name of the route.
    :type name: str
    :param utterances: Example queries for the route. Indexed routers embed these; a
        decision router only reads them when the route has no description.
    :type utterances: List[Any]
    :param description: What the route covers, stated literally. Decision routers
        read it.
    :type description: Optional[str]
    :param confidence_threshold: The confidence a router needs before it returns this
        route. It overrides the router's own confidence_threshold.
    :type confidence_threshold: Optional[float]
    :param metadata: Anything else to keep with the route.
    :type metadata: Dict[str, Any]
    """

    name: str
    utterances: List[Any] = []
    description: Optional[str] = None
    confidence_threshold: Optional[float] = None
    metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the route to a dictionary.

        :return: The dictionary representation of the route.
        :rtype: Dict[str, Any]
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Route":
        """Create a Route object from a dictionary.

        :param data: The dictionary to create the route from.
        :type data: Dict[str, Any]
        :return: The created route.
        :rtype: Route
        """
        return cls(**data)

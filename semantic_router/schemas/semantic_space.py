from pydantic.dataclasses import dataclass

from semantic_router.schemas.route import Route


@dataclass
class SemanticSpace:
    id: str
    routes: list[Route]
    encoder: str = ""

    def __init__(self, routes: list[Route] = []):
        self.id = ""
        self.routes = routes

    def add(self, route: Route):
        self.routes.append(route)

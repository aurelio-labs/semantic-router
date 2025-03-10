Migration Guide for v0.1.0
==========================

- `from semantic_router import RouteLayer` -> `from semantic_router.routers import SemanticRouter`
- `SemanticRouter.add(route: Route)` -> `SemanticRouter.add(routes: List[Route])`
- If expecting routes to sync between local and remote on initialization, use `SemanticRouter(..., auto_sync="local")`. Read more about `auto_sync` and :doc:`synchronization strategies <route-layer/sync>`.
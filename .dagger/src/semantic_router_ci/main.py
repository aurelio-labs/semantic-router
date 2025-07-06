import dagger
from dagger import dag, function, object_type, Container


@object_type
class SemanticRouter:
    async def build(
        self, src: dagger.Directory, extra: str | None
    ) -> Container:
        """Builds python image with uv and installs dev
        dependencies
        """
        uv_cmd = ["uv", "sync"]
        if extra:
            uv_cmd.extend(["--extra", extra])
        return await (
            dag.container()
            .from_("ghcr.io/astral-sh/uv:debian")
            .with_directory("/app", src)
            .with_workdir("/app")
            # install deps
            .with_exec(uv_cmd)
        )

    @function
    async def lint(self, src: dagger.Directory) -> str:
        """Checks if source code passes linter
        """
        return await (
            (await self.build(src=src, extra="dev"))
            # run lint checks
            .with_exec(["uv", "run", "ruff", "check", "."])
            .with_exec(["uv", "run", "ruff", "format", "."])
            .with_exec(["uv", "run", "mypy", "."])
            # return output of lint
            .stdout()
        )

    @function
    def pinecone_service(self) -> dagger.Service:
        """Build and run a pinecone-local service. Used to test
        PineconeIndex
        """
        return (
            dag.container()
            .from_("ghcr.io/pinecone-io/pinecone-local")
            .with_env_variable("PINECONE_HOST", "0.0.0.0")
            .with_env_variable("PORT", "5080")
            .with_exposed_port(5080)
            .as_service(use_entrypoint=True)
        )

    def postgres_service(self) -> dagger.Service:
        """Build and run a postgres instance with
        pgvector installed. Used to test PostgresIndex
        """
        return (
            dag.container()
            .from_("pgvector/pgvector:pg17")
            .with_env_variable("POSTGRES_USER", "postgres")
            .with_env_variable("POSTGRES_PASSWORD", "postgres")
            .with_env_variable("POSTGRES_DB", "postgres")
            .with_env_variable("POSTGRES_HOST", "localhost")
            .with_env_variable("POSTGRES_PORT", "5432")
            .with_exposed_port(5432)
            .as_service(use_entrypoint=True)
        )

    @function
    async def unit_test(self, src: dagger.Directory) -> str:
        """Runs unit tests only for semantic-router.
        """
        return await (
            # create a build with all dependencies so we cover all tests
            (await self.build(src=src, extra="all"))
            .with_service_binding("postgres", self.postgres_service())
            .with_service_binding("pinecone", self.pinecone_service())
            .with_env_variable("PINECONE_API_KEY", "pclocal")
            .with_env_variable("PINECONE_API_BASE_URL", "http://pinecone:5080")
            .with_exec(["uv", "run", "pytest", "-vv", "tests/unit"])
            .stdout()
        )
        

    @function
    async def test(self, src: dagger.Directory, scope: str = "unit") -> str:
        """Runs tests for semantic-router, scope can be 
        set to run for 'unit', 'functional', 'integration',
        or 'all'. By default scope is set to 'unit'.
        """
        # create uv command as per scope
        uv_cmd = [
            "uv", "run", "pytest", "-vv",
            (f"tests/{scope}" if scope != "all" else "tests")
        ]
        # TODO revert to full tests
        uv_cmd = ["uv", "run", "pytest", "-vv", "tests/unit/test_sync.py"]
        return await (
            # create a build with all dependencies so we cover all tests
            (await self.build(src=src, extra="all"))
            .with_service_binding("postgres", self.postgres_service())
            .with_service_binding("pinecone", self.pinecone_service())
            .with_exec(uv_cmd)
            .stdout()
        )

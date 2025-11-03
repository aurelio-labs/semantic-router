import os

import dagger
from dagger import Container, dag, function, object_type


@object_type
class SemanticRouter:
    async def build(
        self, src: dagger.Directory, extra: str | None, python_version: str = "3.11"
    ) -> Container:
        """Builds python image with uv and installs dev
        dependencies
        """
        uv_cmd = ["uv", "sync"]
        if extra:
            uv_cmd.extend(["--extra", extra])
        python_image = f"python:{python_version}-slim"
        return await (
            dag.container()
            .from_(python_image)
            .with_directory("/app", src)
            .with_workdir("/app")
            .with_exec(["apt-get", "update"])
            .with_exec(["apt-get", "install", "-y", "build-essential", "git"])
            # install deps
            .with_exec(["pip", "install", "uv"])
            .with_exec(uv_cmd)
        )

    @function
    async def lint(self, src: dagger.Directory, python_version: str = "3.11") -> str:
        """Checks if source code passes linter"""
        return await (
            (await self.build(src=src, extra="dev", python_version=python_version))
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
    async def unit_test(
        self, src: dagger.Directory, python_version: str = "3.11"
    ) -> str:
        """Runs unit tests only for semantic-router."""
        return await (
            # create a build with all dependencies so we cover all tests
            (await self.build(src=src, extra="all", python_version=python_version))
            .with_service_binding("postgres", self.postgres_service())
            .with_service_binding("pinecone", self.pinecone_service())
            .with_env_variable("PINECONE_API_KEY", "pclocal")
            .with_env_variable("PINECONE_API_BASE_URL", "http://pinecone:5080")
            .with_exec(["uv", "run", "pytest", "-vv", "tests/unit/test_sync.py"])
            .stdout()
        )

    @function
    async def test(
        self,
        src: dagger.Directory,
        scope: str = "unit",
        openai_api_key: str = "",
        cohere_api_key: str = "",
        pinecone_api_key: str = "",
        python_version: str = "3.11",
    ) -> str:
        """Runs tests for semantic-router. Scope: 'unit' (default), 'functional', 'integration', or 'all'."""
        # Map scope to pytest arguments
        if scope == "all":
            pytest_args = [
                "uv",
                "run",
                "pytest",
                "-vv",
                "--timeout=180",
                "--cov=semantic_router",
                "--cov-report=term-missing",
                "--cov-report=xml",
                "--exitfirst",
                "--maxfail=1",
                "tests",
            ]
        elif scope == "unit":
            pytest_args = [
                "uv",
                "run",
                "pytest",
                "-vv",
                "--timeout=180",
                "--exitfirst",
                "--maxfail=1",
                "tests/unit",
            ]
        elif scope == "functional":
            pytest_args = [
                "uv",
                "run",
                "pytest",
                "-vv",
                "--timeout=180",
                "-s",
                "--exitfirst",
                "--maxfail=1",
                "tests/functional",
            ]
        elif scope == "integration":
            pytest_args = [
                "uv",
                "run",
                "pytest",
                "-vv",
                "--timeout=180",
                "--exitfirst",
                "--maxfail=1",
                "tests/integration",
            ]
        else:
            pytest_args = ["uv", "run", "pytest", "-vv", "--timeout=180", "tests/unit"]

        container = await self.build(
            src=src, extra="all", python_version=python_version
        )
        if openai_api_key:
            container = container.with_env_variable("OPENAI_API_KEY", openai_api_key)
        if cohere_api_key:
            container = container.with_env_variable("COHERE_API_KEY", cohere_api_key)
        if pinecone_api_key:
            container = container.with_env_variable(
                "PINECONE_API_KEY", pinecone_api_key
            )
        # Forward optional shared index name to the test container
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
        if pinecone_index_name:
            container = container.with_env_variable(
                "PINECONE_INDEX_NAME", pinecone_index_name
            )
        container = container.with_service_binding("postgres", self.postgres_service())
        pinecone_api_base_url = os.environ.get("PINECONE_API_BASE_URL")
        # Decide cloud vs local
        if pinecone_api_base_url is None:
            # No explicit base URL provided; infer from API key
            if pinecone_api_key and pinecone_api_key != "pclocal":
                # Real key provided: prefer cloud
                pinecone_api_base_url = "https://api.pinecone.io"
            else:
                # Local mode
                pinecone_api_base_url = "http://pinecone:5080"
        # Start local emulator only if pointing to local
        if (
            pinecone_api_base_url.startswith("http://pinecone:5080")
            or "localhost" in pinecone_api_base_url
        ):
            container = container.with_service_binding(
                "pinecone", self.pinecone_service()
            )
        # Set env vars inside test container
        container = (
            container.with_env_variable("PINECONE_API_BASE_URL", pinecone_api_base_url)
            .with_env_variable(
                "POSTGRES_HOST", os.environ.get("POSTGRES_HOST", "postgres")
            )
            .with_env_variable("POSTGRES_PORT", os.environ.get("POSTGRES_PORT", "5432"))
            .with_env_variable("POSTGRES_DB", os.environ.get("POSTGRES_DB", "postgres"))
            .with_env_variable(
                "POSTGRES_USER", os.environ.get("POSTGRES_USER", "postgres")
            )
            .with_env_variable(
                "POSTGRES_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "postgres")
            )
        )
        container = container.with_exec(pytest_args)
        return await container.stdout()

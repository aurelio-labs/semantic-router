import logging
import os
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict
from typing_extensions import deprecated

from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.schema import ConfigParameter, Metric, SparseEmbedding
from semantic_router.utils.logger import logger

if TYPE_CHECKING:
    import psycopg

try:
    import psycopg

    _psycopg_installed = True
except ImportError:
    _psycopg_installed = False


class MetricPgVecOperatorMap(Enum):
    """Enum to map the metric to PostgreSQL vector operators."""

    cosine = "<=>"
    dotproduct = "<#>"  # inner product
    euclidean = "<->"  # L2 distance
    manhattan = "<+>"  # L1 distance


class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"


def parse_vector(vector_str: Union[str, Any]) -> List[float]:
    """Parses a vector from a string or other representation.

    :param vector_str: The string or object representation of a vector.
    :type vector_str: Union[str, Any]
    :return: A list of floats representing the vector.
    :rtype: List[float]
    """
    if isinstance(vector_str, str):
        vector_str = vector_str.strip('()"[]')
        return list(map(float, vector_str.split(",")))
    else:
        return vector_str


def clean_route_name(route_name: str) -> str:
    """Cleans and formats the route name by stripping spaces and replacing them with hyphens.

    :param route_name: The original route name.
    :type route_name: str
    :return: The cleaned and formatted route name.
    :rtype: str
    """
    return route_name.strip().replace(" ", "-")


class PostgresIndexRecord(BaseModel):
    """Model to represent a record in the Postgres index."""

    id: str = ""
    route: str
    utterance: str
    vector: List[float]

    def __init__(self, **data) -> None:
        """Initializes a new Postgres index record with given data.

        :param data: Field values for the record.
        :type data: dict
        """
        if not _psycopg_installed:
            raise ImportError(
                "Please install psycopg to use PostgresIndex. "
                "You can install it with: `pip install 'semantic-router[postgres]'`"
            )
        super().__init__(**data)
        clean_route = self.route.strip().replace(" ", "-")
        if len(clean_route) > 255:
            raise ValueError(
                f"The cleaned route name '{clean_route}' exceeds the 255 character limit."
            )
        route_namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, clean_route)
        hashed_uuid = uuid.uuid5(route_namespace_uuid, self.utterance)
        self.id = clean_route + "#" + str(hashed_uuid)

    def to_dict(self) -> Dict:
        """Converts the record to a dictionary.

        :return: A dictionary representation of the record.
        :rtype: Dict
        """
        return {
            "id": self.id,
            "vector": self.vector,
            "route": self.route,
            "utterance": self.utterance,
        }


class PostgresIndex(BaseIndex):
    """Postgres implementation of Index."""

    connection_string: Optional[str] = None
    index_prefix: str = "semantic_router_"
    index_name: str = "index"
    metric: Metric = Metric.COSINE
    namespace: Optional[str] = ""
    conn: Optional["psycopg.Connection"] = None
    async_conn: Optional["psycopg.AsyncConnection"] = None
    type: str = "postgres"
    index_type: IndexType = IndexType.FLAT
    init_async_index: bool = False

    def __init__(
        self,
        connection_string: Optional[str] = None,
        index_prefix: str = "semantic_router_",
        index_name: str = "index",
        metric: Metric = Metric.COSINE,
        namespace: Optional[str] = "",
        dimensions: int | None = None,
        init_async_index: bool = False,
    ):
        """Initializes the Postgres index with the specified parameters.

        :param connection_string: The connection string for the PostgreSQL database.
        :type connection_string: Optional[str]
        :param index_prefix: The prefix for the index table name.
        :type index_prefix: str
        :param index_name: The name of the index table.
        :type index_name: str
        :param dimensions: The number of dimensions for the vectors.
        :type dimensions: int
        :param metric: The metric used for vector comparisons.
        :type metric: Metric
        :param namespace: An optional namespace for the index.
        :type namespace: Optional[str]
        :param init_async_index: Whether to initialize the index asynchronously.
        :type init_async_index: bool
        """
        if not _psycopg_installed:
            raise ImportError(
                "Please install psycopg to use PostgresIndex. "
                "You can install it with: `pip install 'semantic-router[postgres]'`"
            )
        super().__init__()
        if index_prefix:
            logger.warning("`index_prefix` is deprecated and will be removed in 0.2.0")
        if connection_string or (
            connection_string := os.getenv("POSTGRES_CONNECTION_STRING")
        ):
            pass
        else:
            required_env_vars = [
                "POSTGRES_USER",
                "POSTGRES_PASSWORD",
                "POSTGRES_HOST",
                "POSTGRES_PORT",
                "POSTGRES_DB",
            ]
            missing = [var for var in required_env_vars if not os.getenv(var)]
            if missing:
                raise ValueError(
                    f"Missing required environment variables for Postgres connection: {', '.join(missing)}"
                )
            connection_string = (
                f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
                f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
            )
        self.connection_string = connection_string
        self.index = self
        self.index_prefix = index_prefix
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.namespace = namespace
        self.init_async_index = init_async_index
        self.conn = None
        self.async_conn = None

    def _init_index(self, force_create: bool = False) -> Union[Any, None]:
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        if not self.connection_string:
            raise ValueError("No `self.connection_string` attribute set")
        # Add connection and statement timeouts
        self.conn = psycopg.connect(conninfo=self.connection_string)
        if not self.has_connection():
            raise ValueError("Index has not established a connection to Postgres")

        dimensions_given = self.dimensions is not None
        if not dimensions_given:
            raise ValueError("Dimensions are required for PostgresIndex")
        table_name = self._get_table_name()
        if not self._check_embeddings_dimensions():
            raise ValueError(
                f"The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}."
            )
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE EXTENSION IF NOT EXISTS vector;
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        route VARCHAR(255),
                        utterance TEXT,
                        vector VECTOR({self.dimensions})
                    );
                    COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';
                    """
                )
                self.conn.commit()
            self._create_route_index()
            self._create_index()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise
        return self

    async def _init_async_index(self, force_create: bool = False) -> Union[Any, None]:
        logging.warning("[DEBUG] Entering _init_async_index for PostgresIndex")
        if self.async_conn is None:
            if not self.connection_string:
                raise ValueError("No `self.connection_string` attribute set")
            logging.warning(
                f"[DEBUG] Connecting async to Postgres with: {self.connection_string}"
            )
            self.async_conn = await psycopg.AsyncConnection.connect(
                self.connection_string
            )
            logging.warning(f"[DEBUG] Async connection established: {self.async_conn}")
        if self.dimensions is None and not force_create:
            logging.warning(
                "[DEBUG] No dimensions and not force_create, returning None from _init_async_index"
            )
            return None
        if self.dimensions is None:
            raise ValueError("Dimensions are required for PostgresIndex")
        table_name = self._get_table_name()
        logging.warning(f"[DEBUG] Table name for async index: {table_name}")
        if not await self._async_check_embeddings_dimensions():
            raise ValueError(
                f"The length of the vector embeddings in the existing table {table_name} "
                f"does not match the expected dimensions of {self.dimensions}."
            )
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established a connection to async Postgres")
        try:
            async with self.async_conn.cursor() as cur:
                logging.warning(f"[DEBUG] Creating extension/table for {table_name}")
                await cur.execute(
                    f"""
                    CREATE EXTENSION IF NOT EXISTS vector;
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        route VARCHAR(255),
                        utterance TEXT,
                        vector VECTOR({self.dimensions})
                    );
                    COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';
                    """
                )
                await self.async_conn.commit()
                await self._async_create_route_index()
                await self._async_create_index()
                logging.warning(
                    f"[DEBUG] Finished async index/table creation for {table_name}"
                )
        except Exception as e:
            logging.warning(f"[DEBUG] Exception in _init_async_index: {e}")
            await self.async_conn.rollback()
            raise e
        logging.warning("[DEBUG] Exiting _init_async_index for PostgresIndex")
        return self

    def _get_table_name(self) -> str:
        """
        Returns the name of the table for the index.

        :return: The table name.
        :rtype: str
        """
        return f"{self.index_prefix}{self.index_name}"

    def _get_metric_operator(self) -> str:
        """Returns the PostgreSQL operator for the specified metric.

        :return: The PostgreSQL operator.
        :rtype: str
        """
        return MetricPgVecOperatorMap[self.metric.value].value

    def _get_score_query(self, embeddings_str: str) -> str:
        """Creates the select statement required to return the embeddings distance.

        :param embeddings_str: The string representation of the embeddings.
        :type embeddings_str: str
        :return: The SQL query part for scoring.
        :rtype: str
        """
        operator = self._get_metric_operator()
        if self.metric == Metric.COSINE:
            return f"1 - (vector {operator} {embeddings_str}) AS score"
        elif self.metric == Metric.DOTPRODUCT:
            return f"(vector {operator} {embeddings_str}) * -1 AS score"
        elif self.metric == Metric.EUCLIDEAN:
            return f"vector {operator} {embeddings_str} AS score"
        elif self.metric == Metric.MANHATTAN:
            return f"vector {operator} {embeddings_str} AS score"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _get_vector_operator(self) -> str:
        if self.metric == Metric.COSINE:
            return "vector_cosine_ops"
        elif self.metric == Metric.DOTPRODUCT:
            return "vector_ip_ops"
        elif self.metric == Metric.EUCLIDEAN:
            return "vector_l2_ops"
        elif self.metric == Metric.MANHATTAN:
            return "vector_l1_ops"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _create_route_index(self) -> None:
        """Creates a index on the route column."""
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table_name}_route_idx ON {table_name} USING btree (route);"
                )
                self.conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.conn is not None:
                self.conn.rollback()
            pass
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_create_route_index(self) -> None:
        """Asynchronously creates an index on the route column."""
        table_name = self._get_table_name()

        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established a connection to async Postgres")

        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table_name}_route_idx ON {table_name} USING btree (route);"
                )
            await self.async_conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            pass
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def _create_index(self) -> None:
        """Creates an index on the vector column based on index_type."""
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        opclass = self._get_vector_operator()
        try:
            with self.conn.cursor() as cur:
                if self.index_type == IndexType.HNSW:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING hnsw (vector {opclass});
                        """
                    )
                elif self.index_type == IndexType.IVFFLAT:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 100);
                        """
                    )
                elif self.index_type == IndexType.FLAT:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 1);
                        """
                    )
                self.conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.conn is not None:
                self.conn.rollback()
            pass
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_create_index(self) -> None:
        """Asynchronously creates an index on the vector column based on index_type."""
        table_name = self._get_table_name()

        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established a connection to async Postgres")

        opclass = self._get_vector_operator()

        try:
            async with self.async_conn.cursor() as cur:
                if self.index_type == IndexType.HNSW:
                    await cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING hnsw (vector {opclass});
                        """
                    )
                elif self.index_type == IndexType.IVFFLAT:
                    await cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 100);
                        """
                    )
                elif self.index_type == IndexType.FLAT:
                    await cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_vector_idx ON {table_name} USING ivfflat (vector {opclass}) WITH (lists = 1);
                        """
                    )
            await self.async_conn.commit()
        except psycopg.errors.DuplicateTable:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            pass
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    @deprecated(
        "Use _init_index or sync methods such as `auto_sync` (read more "
        "https://docs.aurelio.ai/semantic-router/user-guide/features/sync). "
        "This method will be removed in 0.2.0"
    )
    def setup_index(self) -> None:
        """Sets up the index by creating the table and vector extension if they do not exist.

        :raises ValueError: If the existing table's vector dimensions do not match the expected dimensions.
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not self._check_embeddings_dimensions():
            raise ValueError(
                f"The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}."
            )
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR(255) PRIMARY KEY,
                    route VARCHAR(255),
                    utterance TEXT,
                    vector VECTOR({self.dimensions})
                );
                COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';
                """
            )
            self.conn.commit()
        self._create_route_index()
        self._create_index()

    def _check_embeddings_dimensions(self) -> bool:
        """Checks if the length of the vector embeddings in the table matches the expected
        dimensions, or if no table exists.

        :return: True if the dimensions match or the table does not exist, False otherwise.
        :rtype: bool
        :raises ValueError: If the vector column comment does not contain a valid integer.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='{table_name}');"
                )
                fetch_result = cur.fetchone()
                exists = fetch_result[0] if fetch_result else None
                if not exists:
                    return True
                cur.execute(
                    f"""SELECT col_description('{table_name}'::regclass, attnum) AS column_comment
                        FROM pg_attribute
                        WHERE attrelid = '{table_name}'::regclass
                        AND attname='vector'"""
                )
                result = cur.fetchone()
                dimension_comment = result[0] if result else None
                if dimension_comment:
                    try:
                        vector_length = int(dimension_comment.split()[-1])
                        return vector_length == self.dimensions
                    except ValueError:
                        raise ValueError(
                            "The 'vector' column comment does not contain a valid integer."
                        )
                else:
                    raise ValueError("No comment found for the 'vector' column.")
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_check_embeddings_dimensions(self) -> bool:
        """Asynchronously checks if the vector embedding dimensions match the expected ones.

        Returns True if dimensions match or table does not exist, False otherwise.

        :return: True if the dimensions match or the table does not exist, False otherwise.
        :rtype: bool
        :raises ValueError: If the vector column comment does not contain a valid integer.
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established a connection to async Postgres")

        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(
                    f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='{table_name}');"
                )
                fetch_result = await cur.fetchone()
                exists = fetch_result[0] if fetch_result else None

                if not exists:
                    return True

                await cur.execute(
                    f"""SELECT col_description('{table_name}'::regclass, attnum) AS column_comment
                        FROM pg_attribute
                        WHERE attrelid = '{table_name}'::regclass
                        AND attname = 'vector';"""
                )
                result = await cur.fetchone()
                dimension_comment = result[0] if result else None

                if dimension_comment:
                    try:
                        vector_length = int(dimension_comment.split()[-1])
                        return vector_length == self.dimensions
                    except ValueError:
                        raise ValueError(
                            "The 'vector' column comment does not contain a valid integer."
                        )
                else:
                    raise ValueError("No comment found for the 'vector' column.")
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        **kwargs,
    ) -> None:
        """Adds records to the index.

        :param embeddings: A list of vector embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: A list of route names corresponding to the embeddings.
        :type routes: List[str]
        :param utterances: A list of utterances corresponding to the embeddings.
        :type utterances: List[Any]
        :param function_schemas: A list of function schemas corresponding to the embeddings.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: A list of metadata corresponding to the embeddings.
        :type metadata_list: List[Dict[str, Any]]
        :raises ValueError: If the vector embeddings being added do not match the expected dimensions.
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        new_embeddings_length = len(embeddings[0])
        if new_embeddings_length != self.dimensions:
            raise ValueError(
                f"The vector embeddings being added are of length {new_embeddings_length}, which does not match the expected dimensions of {self.dimensions}."
            )
        records = [
            PostgresIndexRecord(vector=vector, route=route, utterance=utterance)
            for vector, route, utterance in zip(embeddings, routes, utterances)
        ]
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {table_name} (id, route, utterance, vector) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    [
                        (record.id, record.route, record.utterance, record.vector)
                        for record in records
                    ],
                )
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        batch_size: int = 100,
        **kwargs,
    ) -> None:
        """
        Asynchronously adds records to the index in batches.

        :param embeddings: A list of vector embeddings to add.
        :param routes: A list of route names corresponding to the embeddings.
        :param utterances: A list of utterances corresponding to the embeddings.
        :param function_schemas: (Optional) List of function schemas.
        :param metadata_list: (Optional) List of metadata dictionaries.
        :param batch_size: Number of records per batch insert.
        :raises ValueError: If the vector embeddings don't match expected dimensions.
        :raises TypeError: If connection is not an async Postgres connection.
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established an async connection to Postgres")

        table_name = self._get_table_name()
        new_embeddings_length = len(embeddings[0])
        if new_embeddings_length != self.dimensions:
            raise ValueError(
                f"The vector embeddings being added are of length {new_embeddings_length}, "
                f"which does not match the expected dimensions of {self.dimensions}."
            )

        try:
            async with self.async_conn.cursor() as cur:
                for i in range(0, len(embeddings), batch_size):
                    batch_embeddings = embeddings[i : i + batch_size]
                    batch_routes = routes[i : i + batch_size]
                    batch_utterances = utterances[i : i + batch_size]

                    values = [
                        (str(uuid.uuid4()), route, utterance, vector)
                        for route, utterance, vector in zip(
                            batch_routes, batch_utterances, batch_embeddings
                        )
                    ]

                    await cur.executemany(
                        f"INSERT INTO {table_name} (id, route, utterance, vector) "
                        f"VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                        values,
                    )

                await self.async_conn.commit()
        except Exception:
            await self.async_conn.rollback()
            raise

    def delete(self, route_name: str) -> None:
        """Deletes records with the specified route name.

        :param route_name: The name of the route to delete records for.
        :type route_name: str
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name} WHERE route = '{route_name}'")
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established an async connection to Postgres")

        table_name = self._get_table_name()

        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(
                    f"SELECT id FROM {table_name} WHERE route = %s", (route_name,)
                )
                result = await cur.fetchall()
                deleted_ids = [row[0] for row in result]

                await cur.execute(
                    f"DELETE FROM {table_name} WHERE route = %s", (route_name,)
                )

                await self.async_conn.commit()
                return deleted_ids
        except Exception:
            await self.async_conn.rollback()
            raise

    def describe(self) -> IndexConfig:
        """Describes the index by returning its type, dimensions, and total vector count.

        :return: An IndexConfig object containing the index's type, dimensions, and total vector count.
        :rtype: IndexConfig
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.Connection):
            logger.warning("Index has not established a connection to Postgres")
            return IndexConfig(
                type=self.type,
                dimensions=self.dimensions or 0,
                vectors=0,
            )
        try:
            with self.async_conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                result = cur.fetchone()
                count = result[0] if result is not None else 0
                return IndexConfig(
                    type=self.type,
                    dimensions=self.dimensions or 0,
                    vectors=count,
                )
        except Exception:
            if self.async_conn is not None:
                self.async_conn.rollback()
            raise

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return isinstance(self.conn, psycopg.Connection)

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return isinstance(self.async_conn, psycopg.AsyncConnection)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Searches the index for the query vector and returns the top_k results.

        :param vector: The query vector.
        :type vector: np.ndarray
        :param top_k: The number of top results to return.
        :type top_k: int
        :param route_filter: Optional list of routes to filter the results by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: Optional sparse vector to filter the results by.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the scores and routes of the top_k results.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                filter_query = (
                    f" AND route = ANY(ARRAY{route_filter})" if route_filter else ""
                )
                vector_str = f"'[{','.join(map(str, vector.tolist()))}]'"
                score_query = self._get_score_query(vector_str)
                operator = self._get_metric_operator()
                query = (
                    f"SELECT route, {score_query} FROM {table_name} "
                    f"WHERE true{filter_query} "
                    f"ORDER BY vector {operator} {vector_str} LIMIT {top_k}"
                )
                cur.execute(query)
                results = cur.fetchall()
                return np.array([result[1] for result in results]), [
                    result[0] for result in results
                ]
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Asynchronously search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established an async connection to Postgres")
        try:
            async with self.async_conn.cursor() as cur:
                filter_query = (
                    f" AND route = ANY(ARRAY{route_filter})" if route_filter else ""
                )
                vector_str = f"'[{','.join(map(str, vector.tolist()))}]'"
                score_query = self._get_score_query(vector_str)
                operator = self._get_metric_operator()
                query = (
                    f"SELECT route, {score_query} FROM {table_name} "
                    f"WHERE true{filter_query} "
                    f"ORDER BY vector {operator} {vector_str} LIMIT {top_k}"
                )
                await cur.execute(query)
                results = await cur.fetchall()
                return np.array([result[1] for result in results]), [
                    result[0] for result in results
                ]
        except Exception:
            await self.async_conn.rollback()
            raise

    def _get_route_ids(self, route_name: str):
        """Retrieves all vector IDs for a specific route.

        :param route_name: The name of the route to retrieve IDs for.
        :type route_name: str
        :return: A list of vector IDs.
        :rtype: List[str]
        """
        clean_route = clean_route_name(route_name)
        try:
            ids, _ = self._get_all(route_name=f"{clean_route}")
            return ids
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_get_route_ids(self, route_name: str) -> list[str]:
        """Get the IDs of the routes in the index asynchronously.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        try:
            ids, _ = await self._async_get_all(route_name=f"{clean_route}")
            return ids
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def _get_all(
        self, route_name: Optional[str] = None, include_metadata: bool = False
    ):
        """Retrieves all vector IDs and optionally metadata from the Postgres index.

        :param route_name: Optional route name to filter the results by.
        :type route_name: Optional[str]
        :param include_metadata: Whether to include metadata in the results.
        :type include_metadata: bool
        :return: A tuple containing the list of vector IDs and optionally metadata.
        :rtype: Tuple[List[str], List[Dict]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            query = "SELECT id"
            if include_metadata:
                query += ", route, utterance"
            query += f" FROM {table_name}"
            if route_name:
                query += f" WHERE route LIKE '{route_name}%'"
            all_vector_ids = []
            metadata = []
            with self.conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()
                for row in results:
                    all_vector_ids.append(row[0])
                    if include_metadata:
                        metadata.append({"sr_route": row[1], "sr_utterance": row[2]})
            return all_vector_ids, metadata
        except psycopg.errors.UndefinedTable:
            if self.conn is not None:
                self.conn.rollback()
            # Table does not exist, treat as empty
            return [], []
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_get_all(
        self, route_name: Optional[str] = None, include_metadata: bool = False
    ) -> Tuple[List[str], List[Dict]]:
        """Retrieves all vector IDs and optionally metadata from the Postgres index asynchronously.

        :param route_name: Optional route name to filter the results by.
        :type route_name: Optional[str]
        :param include_metadata: Whether to include metadata in the results.
        :type include_metadata: bool
        :return: A tuple containing the list of vector IDs and optionally metadata.
        :rtype: Tuple[List[str], List[Dict]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()

        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established a connection to async Postgres")

        try:
            query = "SELECT id"
            if include_metadata:
                query += ", route, utterance"
            query += f" FROM {table_name}"
            if route_name:
                query += f" WHERE route LIKE '{route_name}%'"

            all_vector_ids = []
            metadata = []

            async with self.async_conn.cursor() as cur:
                await cur.execute(query)
                results = await cur.fetchall()
                for row in results:
                    all_vector_ids.append(row[0])
                    if include_metadata:
                        metadata.append({"sr_route": row[1], "sr_utterance": row[2]})

            return all_vector_ids, metadata

        except psycopg.errors.UndefinedTable:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            # Table does not exist, treat as empty
            return [], []

        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def _remove_and_sync(self, routes_to_delete: dict):
        """
        Remove embeddings in a routes syncing process from the Postgres index.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        :return: List of (route, utterance) tuples that were removed.
        """
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        table_name = self._get_table_name()
        removed = []
        try:
            with self.conn.cursor() as cur:
                for route, utterances in routes_to_delete.items():
                    for utterance in utterances:
                        cur.execute(
                            f"SELECT route, utterance FROM {table_name} WHERE route = %s AND utterance = %s",
                            (route, utterance),
                        )
                        result = cur.fetchone()
                        if result:
                            removed.append(result)
                        cur.execute(
                            f"DELETE FROM {table_name} WHERE route = %s AND utterance = %s",
                            (route, utterance),
                        )
            self.conn.commit()
            return removed
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def _async_remove_and_sync(
        self, routes_to_delete: dict
    ) -> list[tuple[str, str]]:
        """Remove specified routes from index if they exist.

        This method is asynchronous.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        :return: List of (route, utterance) tuples that were removed.
        :rtype: list[tuple[str, str]]
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established an async connection to Postgres")

        table_name = self._get_table_name()
        removed = []

        try:
            async with self.async_conn.cursor() as cur:
                for route, utterances in routes_to_delete.items():
                    for utterance in utterances:
                        await cur.execute(
                            f"SELECT route, utterance FROM {table_name} WHERE route = %s AND utterance = %s",
                            (route, utterance),
                        )
                        result = await cur.fetchone()
                        if result:
                            removed.append(result)
                        await cur.execute(
                            f"DELETE FROM {table_name} WHERE route = %s AND utterance = %s",
                            (route, utterance),
                        )
            await self.async_conn.commit()
            return removed
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    def delete_all(self):
        """Deletes all records from the Postgres index.

        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name}")
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    def delete_index(self) -> None:
        """Deletes the entire table for the index.

        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                # Forcibly terminate other connections to the database (CI safety)
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                      AND pid <> pg_backend_pid();
                    """
                )
                self.conn.commit()
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.conn.commit()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise

    async def adelete_index(self) -> None:
        """Asynchronously delete the entire table for the index.

        :raises TypeError: If the async database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established an async connection to Postgres")
        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                await self.async_conn.commit()
        except Exception:
            if self.async_conn is not None:
                await self.async_conn.rollback()
            raise

    async def aget_routes(self) -> list[tuple]:
        """Asynchronously get a list of route and utterance objects currently
        stored in the index.

        :return: A list of (route_name, utterance) objects.
        :rtype: List[Tuple]
        :raises TypeError: If the database connection is not established.
        """
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established an async connection to Postgres")

        return await self._async_get_routes()

    def _write_config(self, config: ConfigParameter):
        """Write the config to the index.

        :param config: The config to write to the index.
        :type config: ConfigParameter
        """
        logger.warning("No config is written for PostgresIndex.")

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg.Connection):
            logger.warning(
                "Index has not established a connection to Postgres, returning 0"
            )
            return 0
        with self.conn.cursor() as cur:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()
                if count is None:
                    return 0
                return count[0]
            except psycopg.errors.UndefinedTable:
                logger.warning("Table does not exist, returning 0")
                return 0

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.

        :return: The total number of vectors.
        :rtype: int
        """
        table_name = self._get_table_name()
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            logger.warning(
                "Index has not established an async connection to Postgres, returning 0"
            )
            return 0
        async with self.async_conn.cursor() as cur:
            try:
                await cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = await cur.fetchone()
                if count is None:
                    return 0
                return count[0]
            except psycopg.errors.UndefinedTable:
                logger.warning("Table does not exist, returning 0")
                return 0

    def close(self):
        """Closes the psycopg connection if it exists."""
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception as e:
                logger.warning(f"Error closing Postgres connection: {e}")
            self.conn = None

    def __del__(self):
        self.close()

    def has_connection(self) -> bool:
        """Returns True if there is an active and valid psycopg connection, otherwise False."""
        if self.conn is None or self.conn.closed:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True
        except Exception:
            return False

    """Configuration for the Pydantic BaseModel."""
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

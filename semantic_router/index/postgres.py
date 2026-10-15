import os
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from semantic_router.index.base import (
    SR_ROUTE,
    SR_UTTERANCE,
    BaseIndex,
    IndexConfig,
    check_records,
)
from semantic_router.schema import Metric, SparseEmbedding
from semantic_router.utils.logger import logger

if TYPE_CHECKING:
    import psycopg

try:
    import psycopg
    from psycopg.types.json import Jsonb

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
    # whether this object has created (or found) the table, so add can skip the DDL
    _table_ready: bool = PrivateAttr(default=False)

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

    def _connect(self) -> "psycopg.Connection":
        """Returns the sync connection, opening it on first use.

        :return: The psycopg connection.
        :rtype: psycopg.Connection
        """
        if self.conn is None or self.conn.closed:
            if not self.connection_string:
                raise ValueError("No `self.connection_string` attribute set")
            self.conn = psycopg.connect(conninfo=self.connection_string)
        return self.conn

    async def _aconnect(self) -> "psycopg.AsyncConnection":
        """Returns the async connection, opening it on first use.

        :return: The psycopg async connection.
        :rtype: psycopg.AsyncConnection
        """
        if self.async_conn is None or self.async_conn.closed:
            if not self.connection_string:
                raise ValueError("No `self.connection_string` attribute set")
            self.async_conn = await psycopg.AsyncConnection.connect(
                self.connection_string
            )
        return self.async_conn

    def _create_table_query(self) -> str:
        """The statements that create the table, and add the metadata column to a
        table made before it existed.

        :return: The SQL statements.
        :rtype: str
        """
        table_name = self._get_table_name()
        return f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR(255) PRIMARY KEY,
                route VARCHAR(255),
                utterance TEXT,
                vector VECTOR({self.dimensions}),
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
            );
            ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb;
            COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';
            """

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
        self._connect()
        if self.dimensions is None:
            if force_create:
                raise ValueError("Dimensions are required for PostgresIndex")
            return None
        table_name = self._get_table_name()
        if not self._check_embeddings_dimensions():
            raise ValueError(
                f"The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}."
            )
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            self._ensure_vector_extension()
            with self.conn.cursor() as cur:
                cur.execute(self._create_table_query())
                self.conn.commit()
            self._create_route_index()
            self._create_index()
        except Exception:
            if self.conn is not None:
                self.conn.rollback()
            raise
        self._table_ready = True
        return self

    async def _init_async_index(self, force_create: bool = False) -> Union[Any, None]:
        """Async version of :meth:`_init_index`.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        await self._aconnect()
        if self.dimensions is None:
            if force_create:
                raise ValueError("Dimensions are required for PostgresIndex")
            return None
        table_name = self._get_table_name()
        if not await self._async_check_embeddings_dimensions():
            raise ValueError(
                f"The length of the vector embeddings in the existing table {table_name} "
                f"does not match the expected dimensions of {self.dimensions}."
            )
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established a connection to async Postgres")
        try:
            await self._async_ensure_vector_extension()
            async with self.async_conn.cursor() as cur:
                await cur.execute(self._create_table_query())
                await self.async_conn.commit()
                await self._async_create_route_index()
                await self._async_create_index()
        except Exception:
            await self.async_conn.rollback()
            raise
        self._table_ready = True
        return self

    def _ensure_vector_extension(self) -> None:
        """Create the pgvector extension if it is missing.

        ``CREATE EXTENSION IF NOT EXISTS`` is not atomic: two connections that
        both see the extension missing will both try to create it, and the
        loser fails on the extension catalog's unique index. Treat that as
        "already exists".
        """
        if not isinstance(self.conn, psycopg.Connection):
            raise TypeError("Index has not established a connection to Postgres")
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()
        except (psycopg.errors.UniqueViolation, psycopg.errors.DuplicateObject):
            self.conn.rollback()

    async def _async_ensure_vector_extension(self) -> None:
        """Async version of :meth:`_ensure_vector_extension`."""
        if not isinstance(self.async_conn, psycopg.AsyncConnection):
            raise TypeError("Index has not established a connection to async Postgres")
        try:
            async with self.async_conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self.async_conn.commit()
        except (psycopg.errors.UniqueViolation, psycopg.errors.DuplicateObject):
            await self.async_conn.rollback()

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

    def _rows(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        metadata_list: Optional[List[Dict[str, Any]]],
        sparse_embeddings: Optional[List[SparseEmbedding]],
    ) -> List[Tuple]:
        """Builds the rows for an upsert, one per (route, utterance) pair.

        :return: The (id, route, utterance, vector, metadata) rows.
        :rtype: List[Tuple]
        """
        if sparse_embeddings is not None:
            raise ValueError(
                "PostgresIndex stores dense vectors only, use HybridLocalIndex or "
                "PineconeIndex for sparse embeddings."
            )
        metadata_list = check_records(embeddings, routes, utterances, metadata_list)
        if embeddings and len(embeddings[0]) != self.dimensions:
            raise ValueError(
                f"The vector embeddings being added are of length {len(embeddings[0])}, "
                f"which does not match the expected dimensions of {self.dimensions}."
            )
        rows = []
        for vector, route, utterance, metadata in zip(
            embeddings, routes, utterances, metadata_list
        ):
            record = PostgresIndexRecord(
                vector=vector, route=route, utterance=utterance
            )
            rows.append(
                (
                    record.id,
                    record.route,
                    record.utterance,
                    record.vector,
                    Jsonb(metadata),
                )
            )
        return rows

    def _upsert_query(self) -> str:
        """An insert that replaces the record when its (route, utterance) id exists."""
        return (
            f"INSERT INTO {self._get_table_name()} (id, route, utterance, vector, metadata) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
            "route = EXCLUDED.route, utterance = EXCLUDED.utterance, "
            "vector = EXCLUDED.vector, metadata = EXCLUDED.metadata"
        )

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        **kwargs,
    ) -> None:
        """Upserts records into the index, one per (route, utterance) pair. The table
        is created on first use.

        :param embeddings: A list of vector embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: A list of route names corresponding to the embeddings.
        :type routes: List[str]
        :param utterances: A list of utterances corresponding to the embeddings.
        :type utterances: List[Any]
        :param metadata_list: A list of metadata corresponding to the embeddings.
        :type metadata_list: Optional[List[Dict[str, Any]]]
        :param sparse_embeddings: Not supported, PostgresIndex is dense only.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        :raises ValueError: If the vector embeddings being added do not match the expected dimensions.
        """
        if not embeddings:
            return
        self.dimensions = self.dimensions or len(embeddings[0])
        rows = self._rows(
            embeddings, routes, utterances, metadata_list, sparse_embeddings
        )
        if not self._table_ready:
            self._init_index(force_create=True)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.executemany(self._upsert_query(), rows)
                conn.commit()
        except Exception:
            conn.rollback()
            raise

    async def aadd(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        sparse_embeddings: Optional[List[SparseEmbedding]] = None,
        batch_size: int = 100,
        **kwargs,
    ) -> None:
        """Asynchronously upserts records into the index in batches, one per (route,
        utterance) pair. The table is created on first use.

        :param embeddings: A list of vector embeddings to add.
        :param routes: A list of route names corresponding to the embeddings.
        :param utterances: A list of utterances corresponding to the embeddings.
        :param metadata_list: (Optional) List of metadata dictionaries.
        :param sparse_embeddings: Not supported, PostgresIndex is dense only.
        :param batch_size: Number of records per batch insert.
        :raises ValueError: If the vector embeddings don't match expected dimensions.
        """
        if not embeddings:
            return
        self.dimensions = self.dimensions or len(embeddings[0])
        # same deterministic ids as the sync path, so adding twice never duplicates
        rows = self._rows(
            embeddings, routes, utterances, metadata_list, sparse_embeddings
        )
        if not self._table_ready:
            await self._init_async_index(force_create=True)
        conn = await self._aconnect()
        try:
            async with conn.cursor() as cur:
                for i in range(0, len(rows), batch_size):
                    await cur.executemany(
                        self._upsert_query(), rows[i : i + batch_size]
                    )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    def delete(self, route_name: str) -> None:
        """Deletes records with the specified route name.

        :param route_name: The name of the route to delete records for.
        :type route_name: str
        """
        table_name = self._get_table_name()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name} WHERE route = %s", (route_name,))
                conn.commit()
        except psycopg.errors.UndefinedTable:
            # no table yet, so there is nothing to delete
            conn.rollback()
        except Exception:
            conn.rollback()
            raise

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        table_name = self._get_table_name()
        conn = await self._aconnect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {table_name} WHERE route = %s RETURNING id",
                    (route_name,),
                )
                deleted_ids = [row[0] for row in await cur.fetchall()]
            await conn.commit()
            return deleted_ids
        except psycopg.errors.UndefinedTable:
            # no table yet, so there is nothing to delete
            await conn.rollback()
            return []
        except Exception:
            await conn.rollback()
            raise

    def describe(self) -> IndexConfig:
        """Describes the index by returning its type, dimensions, and total vector count.

        :return: An IndexConfig object containing the index's type, dimensions, and total vector count.
        :rtype: IndexConfig
        """
        return IndexConfig(
            type=self.type,
            dimensions=self.dimensions or 0,
            vectors=len(self),
        )

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

    def _search_query(
        self, vector: np.ndarray, top_k: int, route_filter: Optional[List[str]]
    ) -> Tuple[str, Tuple]:
        """Builds the nearest-neighbour query and its parameters.

        :return: The SQL query and its parameters.
        :rtype: Tuple[str, Tuple]
        """
        vector_str = f"'[{','.join(map(str, vector.tolist()))}]'"
        score_query = self._get_score_query(vector_str)
        operator = self._get_metric_operator()
        filter_query = " AND route = ANY(%s)" if route_filter else ""
        query = (
            f"SELECT route, {score_query} FROM {self._get_table_name()} "
            f"WHERE true{filter_query} "
            f"ORDER BY vector {operator} {vector_str} LIMIT {int(top_k)}"
        )
        return query, (list(route_filter),) if route_filter else ()

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
        :param sparse_vector: Not used, PostgresIndex is dense only.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the scores and routes of the top_k results.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        conn = self._connect()
        query, params = self._search_query(vector, top_k, route_filter)
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return np.array([result[1] for result in results]), [
            result[0] for result in results
        ]

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
        :param sparse_vector: Not used, PostgresIndex is dense only.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        conn = await self._aconnect()
        query, params = self._search_query(vector, top_k, route_filter)
        try:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                results = await cur.fetchall()
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        return np.array([result[1] for result in results]), [
            result[0] for result in results
        ]

    def _select_all_query(self, include_metadata: bool) -> str:
        columns = "id, route, utterance, metadata" if include_metadata else "id"
        return f"SELECT {columns} FROM {self._get_table_name()}"

    @staticmethod
    def _records(rows: list, include_metadata: bool) -> Tuple[List[str], List[Dict]]:
        """Splits selected rows into ids and flat record metadata, with the
        ``sr_route`` and ``sr_utterance`` keys next to the stored metadata.
        """
        ids = [row[0] for row in rows]
        if not include_metadata:
            return ids, []
        metadata = [
            {**(row[3] or {}), SR_ROUTE: row[1], SR_UTTERANCE: row[2]} for row in rows
        ]
        return ids, metadata

    def _get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> Tuple[List[str], List[Dict]]:
        """Retrieves all vector IDs and optionally metadata from the Postgres index.

        :param prefix: Unused for Postgres.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the results.
        :type include_metadata: bool
        :return: A tuple containing the list of vector IDs and optionally metadata.
        :rtype: Tuple[List[str], List[Dict]]
        """
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(self._select_all_query(include_metadata))
                rows = cur.fetchall()
            conn.commit()
        except psycopg.errors.UndefinedTable:
            # Table does not exist, treat as empty
            conn.rollback()
            return [], []
        except Exception:
            conn.rollback()
            raise
        return self._records(rows, include_metadata)

    async def _async_get_all(
        self, prefix: Optional[str] = None, include_metadata: bool = False
    ) -> Tuple[List[str], List[Dict]]:
        """Retrieves all vector IDs and optionally metadata from the Postgres index asynchronously.

        :param prefix: Unused for Postgres.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the results.
        :type include_metadata: bool
        :return: A tuple containing the list of vector IDs and optionally metadata.
        :rtype: Tuple[List[str], List[Dict]]
        """
        conn = await self._aconnect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(self._select_all_query(include_metadata))
                rows = await cur.fetchall()
            await conn.commit()
        except psycopg.errors.UndefinedTable:
            # Table does not exist, treat as empty
            await conn.rollback()
            return [], []
        except Exception:
            await conn.rollback()
            raise
        return self._records(rows, include_metadata)

    def delete_all(self):
        """Deletes all records from the Postgres index."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._get_table_name()}")
                conn.commit()
        except psycopg.errors.UndefinedTable:
            conn.rollback()
        except Exception:
            conn.rollback()
            raise

    def delete_index(self) -> None:
        """Deletes the entire table for the index."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self._get_table_name()}")
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        self._table_ready = False

    async def adelete_index(self) -> None:
        """Asynchronously delete the entire table for the index."""
        conn = await self._aconnect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {self._get_table_name()}")
                await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        self._table_ready = False

    def __len__(self):
        """Returns the total number of vectors in the index, 0 if the table does not
        exist yet.

        :return: The total number of vectors.
        """
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._get_table_name()}")
                count = cur.fetchone()
            conn.commit()
        except psycopg.errors.UndefinedTable:
            conn.rollback()
            return 0
        return count[0] if count is not None else 0

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.

        :return: The total number of vectors.
        :rtype: int
        """
        conn = await self._aconnect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT COUNT(*) FROM {self._get_table_name()}")
                count = await cur.fetchone()
            await conn.commit()
        except psycopg.errors.UndefinedTable:
            await conn.rollback()
            return 0
        return count[0] if count is not None else 0

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

import os
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg2
from pydantic import BaseModel

from semantic_router.index.base import BaseIndex
from semantic_router.schema import Metric
from semantic_router.utils.logger import logger


class MetricPgVecOperatorMap(Enum):
    """
    Enum to map the metric to PostgreSQL vector operators.
    """

    cosine = "<=>"
    dotproduct = "<#>"  # inner product
    euclidean = "<->"  # L2 distance
    manhattan = "<+>"  # L1 distance


def parse_vector(vector_str: Union[str, Any]) -> List[float]:
    """
    Parses a vector from a string or other representation.

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
    """
    Cleans and formats the route name by stripping spaces and replacing them with hyphens.

    :param route_name: The original route name.
    :type route_name: str
    :return: The cleaned and formatted route name.
    :rtype: str
    """
    return route_name.strip().replace(" ", "-")


class PostgresIndexRecord(BaseModel):
    """
    Model to represent a record in the Postgres index.
    """

    id: str = ""
    route: str
    utterance: str
    vector: List[float]

    def __init__(self, **data) -> None:
        """
        Initializes a new Postgres index record with given data.

        :param data: Field values for the record.
        :type data: dict
        """
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
        """
        Converts the record to a dictionary.

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
    """
    Postgres implementation of Index.
    """

    connection_string: Optional[str] = None
    index_prefix: str = "semantic_router_"
    index_name: str = "index"
    dimensions: int = 1536
    metric: Metric = Metric.COSINE
    namespace: Optional[str] = ""
    conn: Optional[psycopg2.extensions.connection] = None
    type: str = "postgres"

    def __init__(
        self,
        connection_string: Optional[str] = None,
        index_prefix: str = "semantic_router_",
        index_name: str = "index",
        dimensions: int = 1536,
        metric: Metric = Metric.COSINE,
        namespace: Optional[str] = "",
    ):
        """
        Initializes the Postgres index with the specified parameters.

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
        """
        super().__init__()
        if connection_string:
            self.connection_string = connection_string
        else:
            connection_string = os.environ.get("POSTGRES_CONNECTION_STRING")
            if not connection_string:
                raise ValueError("No connection string provided")
            self.connection_string = connection_string
        self.index_prefix = index_prefix
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.namespace = namespace
        self.conn = psycopg2.connect(dsn=self.connection_string)
        self.setup_index()

    def _get_table_name(self) -> str:
        """
        Returns the name of the table for the index.

        :return: The table name.
        :rtype: str
        """
        return f"{self.index_prefix}{self.index_name}"

    def _get_metric_operator(self) -> str:
        """
        Returns the PostgreSQL operator for the specified metric.

        :return: The PostgreSQL operator.
        :rtype: str
        """
        return MetricPgVecOperatorMap[self.metric.value].value

    def _get_score_query(self, embeddings_str: str) -> str:
        """
        Creates the select statement required to return the embeddings distance.

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

    def setup_index(self) -> None:
        """
        Sets up the index by creating the table and vector extension if they do not exist.

        :raises ValueError: If the existing table's vector dimensions do not match the expected dimensions.
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not self._check_embeddings_dimensions():
            raise ValueError(
                f"The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}."
            )
        if not isinstance(self.conn, psycopg2.extensions.connection):
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

    def _check_embeddings_dimensions(self) -> bool:
        """
        Checks if the length of the vector embeddings in the table matches the expected dimensions, or if no table exists.

        :return: True if the dimensions match or the table does not exist, False otherwise.
        :rtype: bool
        :raises ValueError: If the vector column comment does not contain a valid integer.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
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

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
    ) -> None:
        """
        Adds vectors to the index.

        :param embeddings: A list of vector embeddings to add.
        :type embeddings: List[List[float]]
        :param routes: A list of route names corresponding to the embeddings.
        :type routes: List[str]
        :param utterances: A list of utterances corresponding to the embeddings.
        :type utterances: List[Any]
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
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {table_name} (id, route, utterance, vector) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",  # if matching hash exists do nothing.
                [
                    (record.id, record.route, record.utterance, record.vector)
                    for record in records
                ],
            )
            self.conn.commit()

    def delete(self, route_name: str) -> None:
        """
        Deletes records with the specified route name.

        :param route_name: The name of the route to delete records for.
        :type route_name: str
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE route = '{route_name}'")
            self.conn.commit()

    def describe(self) -> Dict:
        """
        Describes the index by returning its type, dimensions, and total vector count.

        :return: A dictionary containing the index's type, dimensions, and total vector count.
        :rtype: Dict
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()
            if count is None:
                count = 0
            else:
                count = count[0]  # Extract the actual count from the tuple
            return {
                "type": self.type,
                "dimensions": self.dimensions,
                "total_vector_count": count,
            }

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Searches the index for the query vector and returns the top_k results.

        :param vector: The query vector.
        :type vector: np.ndarray
        :param top_k: The number of top results to return.
        :type top_k: int
        :param route_filter: Optional list of routes to filter the results by.
        :type route_filter: Optional[List[str]]
        :return: A tuple containing the scores and routes of the top_k results.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            filter_query = f" AND route = ANY({route_filter})" if route_filter else ""
            # Create the string representation of vector
            vector_str = f"'[{','.join(map(str, vector.tolist()))}]'"
            score_query = self._get_score_query(vector_str)
            operator = self._get_metric_operator()
            cur.execute(
                f"SELECT route, {score_query} FROM {table_name} WHERE true{filter_query} ORDER BY vector {operator} {vector_str} LIMIT {top_k}"
            )
            results = cur.fetchall()
            return np.array([result[1] for result in results]), [
                result[0] for result in results
            ]

    def _get_route_ids(self, route_name: str):
        """
        Retrieves all vector IDs for a specific route.

        :param route_name: The name of the route to retrieve IDs for.
        :type route_name: str
        :return: A list of vector IDs.
        :rtype: List[str]
        """
        clean_route = clean_route_name(route_name)
        ids, _ = self._get_all(route_name=f"{clean_route}")
        return ids

    def _get_all(
        self, route_name: Optional[str] = None, include_metadata: bool = False
    ):
        """
        Retrieves all vector IDs and optionally metadata from the Postgres index.

        :param route_name: Optional route name to filter the results by.
        :type route_name: Optional[str]
        :param include_metadata: Whether to include metadata in the results.
        :type include_metadata: bool
        :return: A tuple containing the list of vector IDs and optionally metadata.
        :rtype: Tuple[List[str], List[Dict]]
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")

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

    def get_routes(self) -> List[Tuple]:
        """
        Gets a list of route and utterance objects currently stored in the index.

        :return: A list of (route_name, utterance) tuples.
        :rtype: List[Tuple]
        """
        # Get all records with metadata
        _, metadata = self._get_all(include_metadata=True)
        # Create a list of (route_name, utterance) tuples
        route_tuples = [(x["sr_route"], x["sr_utterance"]) for x in metadata]
        return route_tuples

    def delete_all(self):
        """
        Deletes all records from the Postgres index.

        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name}")
            self.conn.commit()

    def delete_index(self) -> None:
        """
        Deletes the entire table for the index.

        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.commit()

    def aget_routes(self):
        logger.error("Sync remove is not implemented for PostgresIndex.")

    def __len__(self):
        """
        Returns the total number of vectors in the index.

        :return: The total number of vectors.
        :rtype: int
        :raises TypeError: If the database connection is not established.
        """
        table_name = self._get_table_name()
        if not isinstance(self.conn, psycopg2.extensions.connection):
            raise TypeError("Index has not established a connection to Postgres")
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()
            if count is None:
                return 0
            return count[0]

    class Config:
        """
        Configuration for the Pydantic BaseModel.
        """

        arbitrary_types_allowed = True

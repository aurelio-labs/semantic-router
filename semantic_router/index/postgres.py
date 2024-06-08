from semantic_router.index.base import BaseIndex
import psycopg
from psycopg.connection import Connection
from pydantic import BaseModel
from typing import Any, List, Optional, Tuple, Dict, Union
from enum import Enum
from semantic_router.schema import Metric
import numpy as np
import os
import uuid


class MetricPgVecOperatorMap(Enum):
    cosine = "<=>"
    dotproduct = "<#>"  # inner product
    euclidean = "<->"  # L2 distance
    manhattan = "<+>"  # L1 distance


def parse_vector(vector_str: Union[str, Any]) -> List[float]:
    if isinstance(vector_str, str):
        vector_str = str(vector_str)
        vector_str = vector_str.strip('()"[]')
        return list(map(float, vector_str.split(",")))
    else:
        return vector_str


class PostgresIndexRecord(BaseModel):
    id: str = ""
    route: str
    utterance: str
    vector: List[float]

    def __init__(self, **data) -> None:
        super().__init__(**data)
        clean_route = self.route.strip().replace(" ", "-")
        route_namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, clean_route)
        hashed_uuid = uuid.uuid5(route_namespace_uuid, self.utterance)
        self.id = str(hashed_uuid)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "vector": self.vector,
            "route": self.route,
            "utterance": self.utterance,
        }


class PostgresIndex(BaseIndex):
    """
    Postgres implementation of Index
    """
    connection_string: Optional[str] = None,
    index_prefix: str = "semantic_router_"
    index_name: str = "index"
    dimensions: int = 1536
    metric: Metric = Metric.COSINE
    namespace: Optional[str] = ""
    conn: Optional[Connection] = None
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
        super().__init__()
        if connection_string:
            self.connection_string = connection_string
        else:
            connection_string = os.environ["POSTGRES_CONNECTION_STRING"]
            if not connection_string:
                raise ValueError("No connection string provided")
            else:
                self.connection_string = str(connection_string)
        self.index_prefix = index_prefix
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.namespace = namespace
        self.conn = psycopg.connect(conninfo=self.connection_string)
        self.setup_index()

    def _get_table_name(self) -> str:
        return f"{self.index_prefix}{self.index_name}"

    def _get_metric_operator(self) -> str:
        return MetricPgVecOperatorMap[self.metric.value].value

    def _get_score_query(self, embeddings_str: str) -> str:
        """
        Creates the select statement required to return the embeddings distance.
        """
        opperator = self._get_metric_operator()
        if self.metric == Metric.COSINE:
            return f"1 - (vector {opperator} {embeddings_str}) AS score"
        elif self.metric == Metric.DOTPRODUCT:
            return f"(vector {opperator} {embeddings_str}) * -1 AS score"
        elif self.metric == Metric.EUCLIDEAN:
            return f"vector {opperator} {embeddings_str} AS score"
        elif self.metric == Metric.MANHATTAN:
            return f"vector {opperator} {embeddings_str} AS score"
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def setup_index(self) -> None:
        table_name = self._get_table_name()
        if not self._check_embeddings_dimensions():
            raise ValueError(
                f"The length of the vector embeddings in the existing table {table_name} does not match the expected dimensions of {self.dimensions}."
            )
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id uuid PRIMARY KEY,
                    route TEXT,
                    utterance TEXT,
                    vector VECTOR({self.dimensions})
                );
                COMMENT ON COLUMN {table_name}.vector IS '{self.dimensions}';
            """
            )
            self.conn.commit()

    def _check_embeddings_dimensions(self) -> bool:
        """
        True where the length of the vector embeddings in the table matches the expected dimensions, or no table yet exists.
        """
        table_name = self._get_table_name()
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='{table_name}');"
            )
            fetch_result = cur.fetchone()
            exists = fetch_result[0] if fetch_result is not None else None
            if not exists:
                return True
            cur.execute(
                f"""SELECT col_description('{table_name}':: regclass, attnum) AS column_comment
                        FROM pg_attribute
                        WHERE attrelid = '{table_name}':: regclass
                        AND attname='vector'"""
            )
            result = cur.fetchone()
            dimension_comment = result[0] if result else None
            if dimension_comment:
                try:
                    vector_length = int(dimension_comment.split()[-1])
                    print(vector_length)
                    return vector_length == self.dimensions
                except ValueError:
                    raise ValueError(
                        "The 'vector' column comment does not contain a valid integer."
                    )
            else:
                raise ValueError("No comment found for the 'vector' column.")

    def add(
        self, embeddings: List[List[float]], routes: List[str], utterances: List[Any]
    ) -> None:
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
        table_name = self._get_table_name()
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE route = '{route_name}'")
            self.conn.commit()

    def describe(self) -> Dict:
        table_name = self._get_table_name()
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
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
        Search the index for the query and return top_k results.
        """
        table_name = self._get_table_name()
        with self.conn.cursor() as cur:
            filter_query = f" AND route = ANY({route_filter})" if route_filter else ""
            # create the string representation of vector
            vector_str = f"'[{','.join(map(str, vector.tolist()))}]'"
            score_query = self._get_score_query(vector_str)
            opperator = self._get_metric_operator()
            cur.execute(
                f"SELECT route, {score_query} FROM {table_name} WHERE true{filter_query} ORDER BY vector {opperator} {vector_str} LIMIT {top_k}"
            )
            results = cur.fetchall()
            print(results)
            return np.array([result[1] for result in results]), [
                result[0] for result in results
            ]

    def delete_index(self) -> None:
        table_name = self._get_table_name()
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.commit()

    class Config:
        arbitrary_types_allowed = True

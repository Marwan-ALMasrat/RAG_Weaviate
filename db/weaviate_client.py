"""
db/weaviate_client.py — Weaviate Cloud connection and collection management.

Responsibilities:
  - Connect to Weaviate Cloud Sandbox
  - Create / verify the Pages and Chunks collections
  - Provide a clean teardown

Design note:
  Use `get_client()` singleton throughout the app — connection is reused.
  Call `ensure_collections()` once at startup; it's idempotent.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import weaviate
from weaviate.classes.config import Configure, DataType, Property

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_client() -> weaviate.WeaviateClient:
    """
    Return a connected Weaviate client (singleton).

    Connects to Weaviate Cloud using WEAVIATE_URL + WEAVIATE_API_KEY from .env.
    """
    if not config.WEAVIATE_URL or not config.WEAVIATE_API_KEY:
        raise ValueError(
            "WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env\n"
            "Get them from: https://console.weaviate.cloud"
        )

    logger.info("Connecting to Weaviate: %s", config.WEAVIATE_URL)

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(
            config.WEAVIATE_API_KEY
        ),
        skip_init_checks=True
    )

    logger.info("Weaviate connection established")
    return client


def close_client() -> None:
    """Close the shared client (call at app shutdown)."""
    try:
        client = get_client.__wrapped__()  # type: ignore[attr-defined]
        client.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Collection setup
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collections(client: weaviate.WeaviateClient) -> None:
    existing = set(client.collections.list_all().keys())

    if config.WEAVIATE_PAGES_COLLECTION not in existing:
        _create_pages_collection(client)
    else:
        logger.info("Collection '%s' already exists", config.WEAVIATE_PAGES_COLLECTION)

    if config.WEAVIATE_CHUNKS_COLLECTION not in existing:
        _create_chunks_collection(client)
    else:
        logger.info("Collection '%s' already exists", config.WEAVIATE_CHUNKS_COLLECTION)


def _create_pages_collection(client: weaviate.WeaviateClient) -> None:
    logger.info("Creating collection: %s", config.WEAVIATE_PAGES_COLLECTION)

    client.collections.create(
        name=config.WEAVIATE_PAGES_COLLECTION,
        properties=[
            Property(name="document_title", data_type=DataType.TEXT),
            Property(name="page_image",     data_type=DataType.BLOB),
            Property(name="filename",       data_type=DataType.TEXT),
            Property(name="page_number",    data_type=DataType.INT),
            Property(name="source_pdf",     data_type=DataType.TEXT),
        ],
        vector_config=[
            Configure.MultiVectors.self_provided(name="default")
        ],
    )

    logger.info("Collection '%s' created", config.WEAVIATE_PAGES_COLLECTION)


def _create_chunks_collection(client: weaviate.WeaviateClient) -> None:
    logger.info("Creating collection: %s", config.WEAVIATE_CHUNKS_COLLECTION)

    client.collections.create(
        name=config.WEAVIATE_CHUNKS_COLLECTION,
        properties=[
            Property(name="text",           data_type=DataType.TEXT),
            Property(name="page_number",    data_type=DataType.INT),
            Property(name="source_pdf",     data_type=DataType.TEXT),
            Property(name="chunk_index",    data_type=DataType.INT),
            Property(name="document_title", data_type=DataType.TEXT),
        ],
        vector_config=[
            Configure.Vectors.self_provided(name="default")
        ],
    )

    logger.info("Collection '%s' created", config.WEAVIATE_CHUNKS_COLLECTION)


def delete_collections(client: weaviate.WeaviateClient) -> None:
    for name in [
        config.WEAVIATE_PAGES_COLLECTION,
        config.WEAVIATE_CHUNKS_COLLECTION
    ]:
        if name in client.collections.list_all():
            client.collections.delete(name)
            logger.warning("Deleted collection: %s", name)
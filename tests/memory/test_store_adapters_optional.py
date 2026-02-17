import importlib

from seqoria_agent.memory import stores


def test_optional_store_adapter_modules_import_without_runtime_dependencies():
    for module_name in (
        "seqoria_agent.memory.stores.mongo",
        "seqoria_agent.memory.stores.postgres",
        "seqoria_agent.memory.stores.pgvector",
        "seqoria_agent.memory.stores.qdrant",
    ):
        module = importlib.import_module(module_name)
        assert module is not None


def test_store_exports_include_split_repository_and_vector_adapters():
    assert "MemoryRepositoryProtocol" in stores.__all__
    assert "InMemoryMemoryRepository" in stores.__all__
    assert "MongoMemoryRepository" in stores.__all__
    assert "PostgresMemoryRepository" in stores.__all__
    assert "PgVectorStore" in stores.__all__
    assert "QdrantVectorStore" in stores.__all__

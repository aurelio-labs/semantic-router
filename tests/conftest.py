# Monkeypatch Pinecone SDK host validation to allow 'pinecone:5080' for Dagger CI


def pytest_configure(config):
    try:
        import pinecone

        def patched_check_realistic_host(host: str) -> None:
            # Allow pinecone:5080 and http://pinecone:5080 as valid hosts for Dagger CI
            if (
                "." not in host
                and "localhost" not in host
                and not host.startswith("http://pinecone:")
                and not host.startswith("pinecone:")
            ):
                raise ValueError(
                    f"You passed '{host}' as the host but this does not appear to be valid. Call describe_index() to confirm the host of the index."
                )

        pinecone.pinecone.check_realistic_host = patched_check_realistic_host
    except ImportError:
        pass  # Pinecone not installed, nothing to patch

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests (marked `anyio`) on asyncio only."""
    return "asyncio"

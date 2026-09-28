import pytest

from agentdesk import db
from agentdesk.config import settings


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Every test gets its own seeded database and runs in offline mode."""
    settings.db_path = tmp_path / "test.db"
    settings.llm_api_key = ""
    db.init_db(reset=True)
    yield

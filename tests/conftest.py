import pytest
from starlette.testclient import TestClient

from benchtop.server import Hub, build_admin_app, build_app
from tests.test_core import toy_env


@pytest.fixture
def server_pair(tmp_path):
    toy_env(tmp_path)
    app, hub = build_app(Hub(tmp_path))
    with (
        TestClient(build_admin_app(hub), base_url="http://127.0.0.1:8787",
                   headers={"authorization": f"Bearer {hub.admin_token}"}) as control,
        TestClient(app, base_url="http://127.0.0.1:8788") as agent,
    ):
        yield control, agent, hub

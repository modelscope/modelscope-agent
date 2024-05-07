import os

import pytest
from fastapi import FastAPI
from sqlmodel import Session, select
from tool_service.tool_manager.api import \
    start_docker_container_and_store_status
from tool_service.tool_manager.connections import (create_db_and_tables,
                                                   drop_db_and_tables, engine)
from tool_service.tool_manager.models import ToolInstance, ToolRegisterInfo
from tool_service.tool_manager.sandbox import get_docker_container
from tool_service.tool_manager.utils import PortGenerator

IN_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

USE_REAL_DOCKER = os.environ.get('USE_REAL_DOCKER', 'True').lower() == 'true'

if not os.path.exists('/tmp/test-tool-node'):
    os.mkdir('/tmp/test-tool-node')

app = FastAPI()
app.containers_info = {}
app.node_port_generator = PortGenerator()


@pytest.fixture(scope='function')
def setup():
    drop_db_and_tables()
    create_db_and_tables()
    yield


@pytest.fixture
def mock_tool_info():
    return ToolRegisterInfo(
        node_name='RenewInstance_default',
        tool_name='RenewInstance',
        image='modelscope-agent/tool-node:no-modelscope',
        config={
            'name': 'RenewInstance',
            'RenewInstance': {
                'key': 'test'
            }
        },
        workspace_dir='/tmp/test-tool-node',
        tenant_id='test-tenant')


@pytest.mark.usefixtures('setup')
@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason='Need to set up the docker environment')
def test_start_docker_container_and_store_status(mock_tool_info):
    container = get_docker_container(mock_tool_info)
    if container is not None:
        container.remove(force=True)

    start_docker_container_and_store_status(mock_tool_info, app)
    container = get_docker_container(mock_tool_info)

    with Session(engine) as session:
        tool_container = session.exec(
            select(ToolInstance).where(
                ToolInstance.name == mock_tool_info.node_name)).first()
        assert tool_container.status == 'running'
        assert tool_container.container_id == container.id
        assert tool_container.name == mock_tool_info.node_name

    if container is not None:
        container.remove(force=True)

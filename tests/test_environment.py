import pytest
from modelscope_agent.environment import Environment
from modelscope_agent.schemas import Message


@pytest.fixture(scope='function')
def remote_environment():
    roles = ['agent1', 'agent2']
    env = Environment(roles=roles)
    return env


@pytest.fixture(scope='function')
def local_environment():
    roles = ['agent1', 'agent2']
    env = Environment(roles=roles, remote=False)
    return env


def test_register_roles(remote_environment):
    new_role = 'agent3'
    remote_environment.register_roles([new_role])
    assert new_role in remote_environment.roles
    assert new_role in remote_environment.messages_queue_map
    assert new_role in remote_environment.messages_list_map


def test_store_and_extract_message(remote_environment):
    remote_environment.reset_env_queues()
    role = 'agent1'
    recipient = 'agent2'
    message_content = 'Hello, World!'
    message = Message(
        content=message_content, send_to=recipient, sent_from=role)

    remote_environment.store_message_from_role(role, message)

    # Check if the message is stored correctly
    assert message in remote_environment.message_history
    assert message in remote_environment.get_message_list(recipient)

    # Extract the message by recipient role
    extracted_messages = remote_environment.extract_message_by_role(recipient)
    assert len(extracted_messages) == 1
    assert extracted_messages[0].content == message_content
    assert extracted_messages[0].sent_from == role
    assert not remote_environment.messages_queue_map[recipient].size(
    )  # Queue should be empty after extraction


def test_extract_all_history_message(remote_environment):
    remote_environment.reset_env_queues()
    role = 'agent1'
    recipient = 'agent2'
    message1 = Message(
        content='First Message', send_to=recipient, sent_from=role)
    message2 = Message(
        content='Second Message', send_to=recipient, sent_from=role)

    remote_environment.store_message_from_role(role, message1)
    remote_environment.store_message_from_role(role, message2)

    all_messages = remote_environment.extract_all_history_message()
    assert len(all_messages) == 2
    assert all_messages[0].content == 'First Message'
    assert all_messages[1].content == 'Second Message'

    # Test with a limit
    last_message = remote_environment.extract_all_history_message(limit=1)
    assert len(last_message) == 1
    assert last_message[0].content == 'Second Message'


def test_extract_all_history_message_local(local_environment):
    local_environment.reset_env_queues()
    role = 'agent1'
    recipient = 'agent2'
    message1 = Message(
        content='First Message', send_to=recipient, sent_from=role)
    message2 = Message(
        content='Second Message', send_to=recipient, sent_from=role)

    local_environment.store_message_from_role(role, message1)
    local_environment.store_message_from_role(role, message2)

    all_messages = local_environment.extract_all_history_message()
    assert len(all_messages) == 2
    assert all_messages[0].content == 'First Message'
    assert all_messages[1].content == 'Second Message'

    # Test with a limit
    last_message = local_environment.extract_all_history_message(limit=1)
    assert len(last_message) == 1
    assert last_message[0].content == 'Second Message'


def test_get_roles(remote_environment):
    agent1 = 'agent1'
    agent2 = 'agent2'
    message1 = Message(
        content='First Message', send_to=agent1, sent_from=agent2)
    message2 = Message(
        content='Second Message', send_to=agent2, sent_from=agent1)

    remote_environment.store_message_from_role(agent2, message1)
    remote_environment.store_message_from_role(agent1, message2)

    notified_roles = remote_environment.get_notified_roles()
    assert notified_roles == ['agent1', 'agent2']

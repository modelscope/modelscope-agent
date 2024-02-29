import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef

from .agent import Agent


def _create_remote(cls,
                   name,
                   max_concurrency=1,
                   *args,
                   **kwargs) -> ClientActorHandle:
    return ray.remote(
        name=name,
        max_concurrency=max_concurrency)(cls).remote(*args, **kwargs)


def _create_local(cls, *args, **kwargs):
    return cls(*args, **kwargs)


def create_component(cls,
                     name,
                     remote=False,
                     max_concurrency=1,
                     *args,
                     **kwargs):
    if remote:
        return _create_remote(cls, name, max_concurrency, *args, **kwargs)
    else:
        kwargs['remote'] = remote
        return _create_local(cls, *args, **kwargs)

from .agent import Agent


def _create_remote(cls, name, max_concurrency=1, *args, **kwargs):
    '''
    Create a remote actor by ray
    Args:
        cls: the class to be created
        name: the name of ray actor, also the role name
        max_concurrency: max concurrency of the actor
        *args:
        **kwargs:

    Returns:

    '''
    import ray

    return ray.remote(
        name=name,
        max_concurrency=max_concurrency)(cls).remote(*args, **kwargs)


def _create_local(cls, *args, **kwargs):
    '''
    Create a local object
    Args:
        cls: the class to be created
        *args:
        **kwargs:

    Returns:

    '''
    return cls(*args, **kwargs)


def create_component(cls,
                     name,
                     remote=False,
                     max_concurrency=1,
                     *args,
                     **kwargs):
    kwargs['remote'] = remote
    kwargs['role'] = name
    if remote:
        return _create_remote(cls, name, max_concurrency, *args, **kwargs)
    else:
        return _create_local(cls, *args, **kwargs)

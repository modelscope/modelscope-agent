from .agent import Agent


def _create_remote(cls,
                   name,
                   max_concurrency=1,
                   force_new=False,
                   *args,
                   **kwargs):
    '''
    Create a remote actor by ray
    Args:
        cls: the class to be created
        name: the name of ray actor, also the role name
        max_concurrency: max concurrency of the actor
        focus_new: force to create a new actor
        *args:
        **kwargs:

    Returns:

    '''
    import ray
    try:
        # try to get an existing actor
        ray_actor = ray.get_actor(name)
        if force_new:
            ray.kill(ray_actor)
        else:
            return ray_actor
    except ValueError:
        pass
    # if failed, create a new actor
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
                     prefix_name=None,
                     *args,
                     **kwargs):
    kwargs['remote'] = remote
    kwargs['role'] = name
    kwargs['prefix_name'] = prefix_name
    if remote:
        if prefix_name is not None:
            name = f'{prefix_name}_{name}'
        return _create_remote(cls, name, max_concurrency, *args, **kwargs)
    else:
        return _create_local(cls, *args, **kwargs)

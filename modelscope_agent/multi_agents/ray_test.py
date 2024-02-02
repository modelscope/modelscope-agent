import ray

# ray.init(local_mode=True)


def llm_run():
    for i in range(5):
        yield str(i)


@ray.remote
def step():
    for i in llm_run():
        yield i
    print(13)


agents = {'agent1': step, 'agent2': step}


@ray.remote
def env_step():
    for _ in range(2):
        for agent_step in agents.values():
            accumulated_results = []
            for obj_ref in agent_step.remote():
                x = ray.get(obj_ref)
                accumulated_results.append(x)
                print(x)
            for result in accumulated_results:
                yield result


for x in env_step.remote():
    print(ray.get(x))

ray.shutdown()

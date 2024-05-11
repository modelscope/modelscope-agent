NODE_PORT_START = 31513
NODE_PORT_END = 65535


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PortGenerator(metaclass=SingletonMeta):

    def __init__(self, start=NODE_PORT_START, end=NODE_PORT_END):
        self.start = start
        self.end = end
        self.allocated = set()

    def __iter__(self):
        return self

    def __next__(self):
        for port in range(self.start, self.end + 1):
            if port not in self.allocated:
                self.allocated.add(port)
                return port
        raise StopIteration  # throw StopIteration when no available port

    def release(self, port):
        self.allocated.discard(
            port)  # rlz: discard is more efficient than remove

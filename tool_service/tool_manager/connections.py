import os

import docker
from sqlmodel import Session, SQLModel, create_engine

# database setting
DATABASE_DIR = os.getenv('DATABASE_DIR', './test.db')
DATABASE_URL = os.path.join('sqlite:///', DATABASE_DIR)

engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def drop_db_and_tables():
    SQLModel.metadata.drop_all(engine)


def get_docker_client():
    # Initialize docker client. Throws an exception if Docker is not reachable.
    try:
        docker_client = docker.from_env()
    except docker.errors.DockerException as e:
        print('Please check Docker is running using `docker ps`.')
        print(f'Error! {e}', flush=True)
        raise e
    return docker_client


def get_session():
    with Session(engine) as session:
        yield session

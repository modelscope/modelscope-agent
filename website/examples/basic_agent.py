"""Minimal runnable SDK example used by the promotional website.

    pip install -U ms-agent
    export OPENAI_API_KEY=...  # your provider's key
    export OPENAI_BASE_URL=... # an OpenAI-compatible endpoint
    python basic_agent.py

Set the model in the adjacent agent.yaml to one supported by your provider.
The example intentionally needs no external tool or web-search credentials.
"""
import asyncio
import os
from pathlib import Path

from ms_agent import LLMAgent
from ms_agent.config import Config


async def main():
    if not os.environ.get('OPENAI_API_KEY') or not os.environ.get(
            'OPENAI_BASE_URL'):
        raise SystemExit(
            'Set OPENAI_API_KEY and OPENAI_BASE_URL before running.')
    config = Config.from_task(str(Path(__file__).with_name('agent.yaml')))
    agent = LLMAgent(config)
    messages = await agent.run(
        'Write a three-bullet outline for a talk about models, tools, and agent execution loops.'
    )
    print(messages[-1].content)


if __name__ == '__main__':
    asyncio.run(main())

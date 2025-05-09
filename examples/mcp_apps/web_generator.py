import asyncio
import argparse
import os
from datetime import datetime
from modelscope_agent.tools.mcp import MCPClient


class WebGeneratorClient(MCPClient):
    default_system = f"""You are an assistant that helps generate comprehensive documentations or webpages from gathered information. Today is {datetime.now().strftime("%Y-%m-%d")}.

## Planning

You need to create a CONCISE, FOCUSED plan with ONLY meaningful, actionable steps, rely on the plan after you made it.

If you are making website, just make one single step for writing code to avoid too much messages. Use proper event delegation or direct event binding

Give your final result(documentation/code) in <result></result> block.

Here shows a plan example:

 ```
1. Research & Content Gathering:
   1.1. Search and collect comprehensive information on [topic] using user's language
   1.2. Identify and crawl authoritative sources for detailed content
   1.3. Crawl enough high-quality medias(e.g. image links) from compatible platforms

2. Content Creation & Organization:
   2.1. Develop main content sections with complete information
   2.3. Organize information with logical hierarchy and flow

3. Design & Animation Implementation:
   3.1. Create responsive layout with modern aesthetic, with all the useful information collected
   3.2. Implement key animations for enhanced user experience
   3.3. Write the final code...
```

History messages of the previous main step will not be kept, so you need to WRITE a concise but essential summary_and_result when calling `notebook---advance_to_next_step` for each sub-step.
In the later steps, you can only see the plans you made and the summary_and_result from the previous steps. So you must MINIMIZE DEPENDENCIES between the the steps in the plan. Note: The URL needs to retain complete information.

Here shows a summary_and_result example:
```
MAIN FINDINGS:
• Topic X has three primary categories: A, B, and C
• Latest statistics show 45% increase in adoption since 2023
• Expert consensus indicates approach Y is most effective

COLLECTED RESOURCES:
• Primary source: https://example.com/comprehensive-guide (contains detailed sections on implementation)
• Images: ["https://example.com/image1.jpg?Expires=a&KeyId=b&Signature=c", "https://example.com/image2.jpg", "https://example.com/diagram.png"]
• Reference documentation: https://docs.example.com/api (sections 3.2-3.4 particularly relevant)

DECISIONS MADE:
• Will focus on mobile-first approach due to 78% of users accessing via mobile devices
• Selected blue/green color scheme based on industry standards and brand compatibility
• Decided to implement tabbed interface for complex data presentation

CODE:
```
...
```
"""


async def main():
    parser = argparse.ArgumentParser()

    # Candidate urls: `https://dashscope.aliyuncs.com/compatible-mode/v1`
    parser.add_argument('--base_url', type=str, default="https://api-inference.modelscope.cn/v1")
    # Candidate models: `claude-3-7-sonnet-20250219`, `deepseek-ai/DeepSeek-V3-0324`
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B")
    parser.add_argument("--token", type=str, default=None)

    args = parser.parse_args()
    if not args.token:
        args.token = os.environ.get('MODEL_TOKEN', '')
    web_gen_client = WebGeneratorClient(
        base_url=args.base_url,
        token=args.token, model=args.model,
                                   # mcp=['amap-maps', 'edgeone-pages-mcp-server'])
                                   # mcp=['notebook'])
                                   mcp=['MiniMax-MCP', 'edgeone-pages-mcp-server'])
                                   # mcp=['crawl4ai', 'web-search', 'edgeone-pages-mcp-server'])

    print(f'>>> exist')
    import sys
    sys.exit()


    kwargs = {}
    if 'qwen3' in args.model.lower():
        kwargs.update({'stream': True, 'get_stream_final_anwser': True})
    elif 'claude' in args.model.lower():
        kwargs.update({'max_tokens': 65536})
    try:
        user_input = "查找阿里云谷园区附近咖啡馆，并生成介绍网站"
        # user_input = "做一个图文并茂，同时有声音的绘本故事，部署成一个网页。"
        await client.connect_all_servers(None)
        async for response in client.process_query(None, user_input, system=True, **kwargs):
            print(response)
            print('\n')
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
import json
import trafilatura
from crawl4ai import AsyncWebCrawler
# from crawl4ai import *
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.browser_manager import BrowserManager
from fastmcp import FastMCP

mcp = FastMCP('crawl4ai')


async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()
    # Fix: https://github.com/unclecode/crawl4ai/issues/842
    BrowserManager._playwright_instance = None


AsyncPlaywrightCrawlerStrategy.__aexit__ = __aexit__


@mcp.tool(
    description='A crawl tool to get the content of a website page, '
    'and simplify the content to pure html content. This tool can be used to get the detail '
    'information in the url')
async def crawl_website(website: str) -> str:
    if not website.startswith('http'):
        website = 'http://' + website
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=website, )
            html = str(result.html)
            html = trafilatura.extract(
                html,
                deduplicate=True,
                favor_precision=True,
                include_comments=False,
                output_format='markdown',
                with_metadata=True,
            )
            if not html:
                html = 'Cannot crawl this web page, please try another web page instead'
            if len(html) > 2048:
                html = html[:2048]
            output = {'text': html}
            media_list = []
            if result.media:
                for key in result.media:
                    media_dict = result.media[key]
                    for idx, row in enumerate(media_dict):
                        if idx > 20:
                            break
                        src = row['src'] or ''
                        if src and not src.startswith('http'):
                            src = src.lstrip('/')
                            src = 'https://' + src
                        media_list.append({
                            'type':
                            key,
                            'description':
                            row['alt'][:100] or row['desc'][:100]
                            or 'No description',
                            'link':
                            src,
                        })
                output['media'] = media_list
            return json.dumps(output, ensure_ascii=False)
    except Exception:
        import traceback
        print(traceback.format_exc())
        return 'Cannot crawl this web page, please try another web page instead'


if __name__ == '__main__':
    mcp.run(transport='stdio')

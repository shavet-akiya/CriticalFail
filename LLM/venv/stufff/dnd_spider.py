# dnd_spider.py
import scrapy

class DndSpider(scrapy.Spider):
    name = "dnd_spider"
    allowed_domains = ["dndbeyond.com", "forgottenrealms.fandom.com"]
    start_urls = [
        "https://forgottenrealms.fandom.com/wiki/Main_Page"
    ]

    def parse(self, response):
        # Extract all page links
        for href in response.css("a::attr(href)").getall():
            if href and href.startswith("/wiki/"):
                yield response.follow(href, self.parse_page)

    def parse_page(self, response):
        text = " ".join(response.css("p::text").getall())
        title = response.css("h1::text").get()
        yield {
            "title": title,
            "text": text,
            "url": response.url
        }
    
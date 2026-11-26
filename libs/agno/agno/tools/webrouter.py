import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from agno.tools import Toolkit
from agno.utils.log import log_debug, log_error

try:
    from web_router import AsyncWebRouter, ExtractResponse, SearchResponse, WebRouter, WebRouterError
    from web_router.routing.url_policy import public_only
except ImportError:
    raise ImportError("`webrouter` not installed (requires Python 3.11+). Please install using `pip install webrouter`")


class WebRouterTools(Toolkit):
    """Web search and page extraction through WebRouter.

    WebRouter is an open-source library that puts one interface in front of
    several web search and extraction providers. Each call goes to one provider,
    and falls back to the next when that call fails.

    With no API key configured, calls go to the free, anonymous tiers of the
    providers that offer one. Any provider API key set in the environment
    (for example ``TAVILY_API_KEY`` or ``EXA_API_KEY``) adds that provider to the
    routing, and calls it serves are billed to that key. Set ``search_provider``
    and ``extract_provider`` to send every call to one provider of your choice.
    """

    def __init__(
        self,
        search_provider: Optional[str] = None,
        extract_provider: Optional[str] = None,
        provider_config: Optional[Dict[str, Dict[str, Any]]] = None,
        enable_search: bool = True,
        enable_extract: bool = True,
        all: bool = False,
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        max_chars_per_result: Optional[int] = 1000,
        max_chars_per_page: Optional[int] = 20000,
        format: Literal["json", "markdown"] = "markdown",
        timeout: Optional[float] = 60,
        **kwargs,
    ):
        """Initialize WebRouterTools with search and extract capabilities.

        Args:
            search_provider: Provider for every web search, e.g. "tavily" or "exa". Defaults to None, which lets
                WebRouter pick among the available providers and fall back between them.
            extract_provider: Provider for every page extraction, e.g. "firecrawl". Defaults to None (routed the
                same way).
            provider_config: Constructor arguments per provider, keyed by provider name, for passing credentials
                explicitly: {"tavily": {"api_key": "..."}}. Defaults to None, which reads keys from the environment.
            enable_search: Enable web search. Defaults to True.
            enable_extract: Enable page extraction. Defaults to True.
            all: Enable all available tools. Defaults to False.
            max_results: Default maximum number of search results. Defaults to 5.
            include_domains: Restrict search results to these domains. Defaults to None.
            exclude_domains: Drop search results from these domains. Defaults to None.
            max_chars_per_result: Truncate each search result's text to this many characters. Defaults to 1000.
                Set to None for no truncation.
            max_chars_per_page: Truncate each extracted page to this many characters. Defaults to 20000.
                Set to None for no truncation.
            format: Output format - json or markdown. Defaults to "markdown".
            timeout: Seconds each call may take, across fallbacks. Defaults to 60. Set to None for no limit.
            **kwargs: Additional arguments passed to Toolkit.
        """
        self.search_provider: Optional[str] = search_provider
        self.extract_provider: Optional[str] = extract_provider
        self.provider_config: Optional[Dict[str, Dict[str, Any]]] = provider_config
        self.max_results: int = max_results
        self.include_domains: Optional[List[str]] = include_domains
        self.exclude_domains: Optional[List[str]] = exclude_domains
        self.max_chars_per_result: Optional[int] = max_chars_per_result
        self.max_chars_per_page: Optional[int] = max_chars_per_page
        self.format: Literal["json", "markdown"] = format
        self.timeout: Optional[float] = timeout
        # Built on first use and shared, so provider clients and rate-limit cooldowns persist across calls.
        self._router: Optional[WebRouter] = None

        tools: List[Any] = []
        async_tools: List[Tuple[Any, str]] = []
        if enable_search or all:
            tools.append(self.web_search)
            async_tools.append((self.aweb_search, "web_search"))
        if enable_extract or all:
            tools.append(self.extract_url_content)
            async_tools.append((self.aextract_url_content, "extract_url_content"))

        super().__init__(name="webrouter_tools", tools=tools, async_tools=async_tools, **kwargs)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "WebRouterTools":
        """Copy toolkit state, sharing the router (which holds locks and connection pools) instead of copying it."""
        from copy import deepcopy

        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        if self._router is not None:
            memo.setdefault(id(self._router), self._router)
        for name, value in self.__dict__.items():
            setattr(copied, name, deepcopy(value, memo))
        return copied

    def web_search(self, query: str, max_results: Optional[int] = None) -> str:
        """Use this function to search the web for a given query.

        Args:
            query (str): Query to search for.
            max_results (Optional[int]): Maximum number of results to return. Defaults to the toolkit's max_results.

        Returns:
            str: Search results as markdown or JSON, depending on the toolkit format.
        """
        try:
            log_debug(f"Searching the web with WebRouter for: {query}")
            response = self._get_router().search(query, **self._search_options(max_results))
        except WebRouterError as e:
            log_error(f"WebRouter search failed: {e}")
            return f"Error searching the web: {e}"
        return self._format_search(response)

    async def aweb_search(self, query: str, max_results: Optional[int] = None) -> str:
        """Use this function to search the web for a given query.

        Args:
            query (str): Query to search for.
            max_results (Optional[int]): Maximum number of results to return. Defaults to the toolkit's max_results.

        Returns:
            str: Search results as markdown or JSON, depending on the toolkit format.
        """
        try:
            log_debug(f"Searching the web with WebRouter for: {query}")
            # An async router's connections belong to the event loop that opened them, so each call opens its own.
            async with AsyncWebRouter(provider_config=self.provider_config) as router:
                response = await router.search(query, **self._search_options(max_results))
        except WebRouterError as e:
            log_error(f"WebRouter search failed: {e}")
            return f"Error searching the web: {e}"
        return self._format_search(response)

    def extract_url_content(self, urls: List[str]) -> str:
        """Use this function to read the content of one or more web pages.

        Args:
            urls (List[str]): URLs of the pages to read.

        Returns:
            str: Each page's content as markdown or JSON, depending on the toolkit format, with the URLs
                that could not be read and why.
        """
        allowed, refused = self._screen_urls(urls)
        if not allowed:
            return self._no_urls_message(refused)
        try:
            log_debug(f"Extracting content with WebRouter from: {allowed}")
            response = self._get_router().extract(allowed, **self._extract_options())
        except WebRouterError as e:
            log_error(f"WebRouter extract failed: {e}")
            return f"Error extracting content: {e}"
        return self._format_extract(response, refused)

    async def aextract_url_content(self, urls: List[str]) -> str:
        """Use this function to read the content of one or more web pages.

        Args:
            urls (List[str]): URLs of the pages to read.

        Returns:
            str: Each page's content as markdown or JSON, depending on the toolkit format, with the URLs
                that could not be read and why.
        """
        allowed, refused = self._screen_urls(urls)
        if not allowed:
            return self._no_urls_message(refused)
        try:
            log_debug(f"Extracting content with WebRouter from: {allowed}")
            # An async router's connections belong to the event loop that opened them, so each call opens its own.
            async with AsyncWebRouter(provider_config=self.provider_config) as router:
                response = await router.extract(allowed, **self._extract_options())
        except WebRouterError as e:
            log_error(f"WebRouter extract failed: {e}")
            return f"Error extracting content: {e}"
        return self._format_extract(response, refused)

    def _get_router(self) -> WebRouter:
        if self._router is None:
            self._router = WebRouter(provider_config=self.provider_config)
        return self._router

    def _search_options(self, max_results: Optional[int]) -> Dict[str, Any]:
        return {
            "provider": self.search_provider,
            "max_results": self.max_results if max_results is None else max_results,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "timeout": self.timeout,
            "data_apis": False,
        }

    def _extract_options(self) -> Dict[str, Any]:
        return {"provider": self.extract_provider, "timeout": self.timeout, "data_apis": False}

    @staticmethod
    def _screen_urls(urls: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Split the URLs into those that may be fetched and those refused, with the reason.

        Refuses non-http(s) URLs and URLs whose host, as written, is private, loopback, link-local or a cloud
        metadata endpoint. It does not resolve DNS, so a public name that points at a private address passes.
        """
        allowed: List[str] = []
        refused: Dict[str, str] = {}
        for url in [u.strip() for u in urls if isinstance(u, str) and u.strip()]:
            reason = public_only(url)
            if reason:
                refused[url] = reason
            elif url not in allowed:
                allowed.append(url)
        return allowed, refused

    @staticmethod
    def _no_urls_message(refused: Dict[str, str]) -> str:
        if not refused:
            return "Error: No valid URLs provided."
        return "Error: No URL could be fetched. " + "; ".join(f"{url}: {reason}" for url, reason in refused.items())

    def _format_search(self, response: SearchResponse) -> str:
        results = [
            {
                "title": result.title,
                "url": result.url,
                # Result text is raw page text; collapse its newlines so it cannot break the markdown layout.
                "content": _truncate(
                    " ".join((result.description or result.content).split()), self.max_chars_per_result
                ),
                "published_date": result.published_date,
            }
            for result in response.results
        ]
        if not results:
            return "No results found."

        if self.format == "json":
            payload: Dict[str, Any] = {"query": response.query, "provider": response.provider, "results": results}
            if response.answer:
                payload["answer"] = response.answer
            return json.dumps(payload, indent=2)

        markdown = f"# {response.query}\n\n"
        if response.answer:
            markdown += f"### Summary\n{response.answer}\n\n"
        for result in results:
            markdown += f"### [{result['title'] or result['url']}]({result['url']})\n"
            if result["published_date"]:
                markdown += f"Published: {result['published_date']}\n"
            markdown += f"{result['content']}\n\n"
        return markdown

    def _format_extract(self, response: ExtractResponse, refused: Dict[str, str]) -> str:
        pages = [
            {
                "url": document.url,
                "title": document.title,
                "content": _truncate(document.content, self.max_chars_per_page),
            }
            for document in response.documents
        ]
        failures = {failure.url: failure.error for failure in response.failures}
        failures.update(refused)

        if self.format == "json":
            payload: Dict[str, Any] = {"provider": response.provider, "pages": pages}
            if failures:
                payload["failures"] = [{"url": url, "error": error} for url, error in failures.items()]
            return json.dumps(payload, indent=2)

        markdown = ""
        for page in pages:
            markdown += f"## {page['title'] or page['url']}\n\nURL: {page['url']}\n\n{page['content']}\n\n"
        if failures:
            markdown += "## Could not extract\n\n"
            markdown += "".join(f"- {url}: {error}\n" for url, error in failures.items())
        return markdown or "Error: No content could be extracted from the provided URL(s)."


def _truncate(text: str, limit: Optional[int]) -> str:
    if limit is not None and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text

"""Unit tests for the WebRouterTools toolkit.

The `webrouter` package (Python 3.11+) is replaced with stand-in modules at import
time, so these tests run on every supported Python version, with no package
installed, no network access and no credentials. One contract test checks the
real package's signatures when it is installed.
"""

import copy
import inspect
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class FakeWebRouterError(Exception):
    pass


def _fake_public_only(url: str) -> Optional[str]:
    if "169.254.169.254" in url:
        return "the host is a cloud metadata endpoint"
    if not url.startswith(("http://", "https://")):
        return "the URL is not http or https"
    return None


_web_router = ModuleType("web_router")
_web_router.WebRouter = MagicMock()  # type: ignore[attr-defined]
_web_router.AsyncWebRouter = MagicMock()  # type: ignore[attr-defined]
_web_router.SearchResponse = object  # type: ignore[attr-defined]
_web_router.ExtractResponse = object  # type: ignore[attr-defined]
_web_router.WebRouterError = FakeWebRouterError  # type: ignore[attr-defined]
_routing = ModuleType("web_router.routing")
_url_policy = ModuleType("web_router.routing.url_policy")
_url_policy.public_only = _fake_public_only  # type: ignore[attr-defined]

with patch.dict(
    sys.modules,
    {"web_router": _web_router, "web_router.routing": _routing, "web_router.routing.url_policy": _url_policy},
):
    import agno.tools.webrouter as webrouter_module
    from agno.tools.webrouter import WebRouterTools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _search_response(results=None, answer=None, query="agno agents", provider="tavily"):
    return SimpleNamespace(query=query, provider=provider, answer=answer, results=results or [])


def _result(url="https://example.com/a", title="Example A", description="Short summary", content="", date=None):
    return SimpleNamespace(url=url, title=title, description=description, content=content, published_date=date)


def _extract_response(documents=None, failures=None, provider="nimble"):
    return SimpleNamespace(provider=provider, documents=documents or [], failures=failures or [])


def _document(url="https://example.com/a", title="Page A", content="Page body"):
    return SimpleNamespace(url=url, title=title, content=content)


def _sync_router(**methods) -> MagicMock:
    """A WebRouter class mock whose instance has the given methods."""
    router_cls = MagicMock()
    for name, value in methods.items():
        setattr(router_cls.return_value, name, value)
    return router_cls


def _async_router(**methods) -> MagicMock:
    """An AsyncWebRouter class mock whose async-context-managed instance has the given methods."""
    instance = MagicMock()
    for name, value in methods.items():
        setattr(instance, name, value)
    router_cls = MagicMock()
    router_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
    router_cls.return_value.__aexit__ = AsyncMock(return_value=None)
    return router_cls


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_default_registers_search_and_extract():
    tools = WebRouterTools()
    assert list(tools.functions) == ["web_search", "extract_url_content"]
    assert list(tools.async_functions) == ["web_search", "extract_url_content"]


def test_enable_flags_select_tools():
    assert list(WebRouterTools(enable_extract=False).functions) == ["web_search"]
    assert list(WebRouterTools(enable_search=False).functions) == ["extract_url_content"]
    assert list(WebRouterTools(enable_search=False, enable_extract=False).functions) == []
    assert list(WebRouterTools(enable_search=False, enable_extract=False, all=True).functions) == [
        "web_search",
        "extract_url_content",
    ]


def test_router_is_built_once_and_shared_with_copies():
    router_cls = _sync_router(search=MagicMock(return_value=_search_response([_result()])))
    tools = WebRouterTools(search_provider="exa", provider_config={"exa": {"api_key": "test"}})
    with patch.object(webrouter_module, "WebRouter", router_cls):
        tools.web_search("first")
        tools.web_search("second")
        copied = copy.deepcopy(tools)
        copied.web_search("third")

    router_cls.assert_called_once_with(provider_config={"exa": {"api_key": "test"}})
    assert copied._router is tools._router
    assert copied.search_provider == "exa"
    assert copied.provider_config == {"exa": {"api_key": "test"}}
    assert copied.provider_config is not tools.provider_config


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def test_web_search_sends_configured_options():
    search = MagicMock(return_value=_search_response([_result()]))
    router_cls = _sync_router(search=search)
    tools = WebRouterTools(
        search_provider="exa",
        max_results=3,
        include_domains=["example.com"],
        exclude_domains=["spam.com"],
        timeout=10,
    )
    with patch.object(webrouter_module, "WebRouter", router_cls):
        tools.web_search("agno agents")
        tools.web_search("agno agents", max_results=7)

    first, second = search.call_args_list
    assert first.args == ("agno agents",)
    assert first.kwargs == {
        "provider": "exa",
        "max_results": 3,
        "include_domains": ["example.com"],
        "exclude_domains": ["spam.com"],
        "timeout": 10,
        "data_apis": False,
    }
    assert second.kwargs["max_results"] == 7


def test_web_search_markdown():
    response = _search_response(
        [
            _result(date="2026-01-01", description="# Heading\n\nShort   summary"),
            _result(url="https://example.com/b", title="", description="", content="Body text"),
        ],
        answer="A summary answer",
    )
    router_cls = _sync_router(search=MagicMock(return_value=response))
    with patch.object(webrouter_module, "WebRouter", router_cls):
        output = WebRouterTools().web_search("agno agents")

    assert output.startswith("# agno agents\n\n### Summary\nA summary answer\n\n")
    # Result text has its newlines collapsed, so it stays inside its own block.
    assert "### [Example A](https://example.com/a)\nPublished: 2026-01-01\n# Heading Short summary\n" in output
    # An untitled result is labelled with its URL, and falls back to its content.
    assert "### [https://example.com/b](https://example.com/b)\nBody text\n" in output


def test_web_search_json():
    response = _search_response([_result()], answer="A summary answer")
    router_cls = _sync_router(search=MagicMock(return_value=response))
    with patch.object(webrouter_module, "WebRouter", router_cls):
        output = WebRouterTools(format="json").web_search("agno agents")

    assert json.loads(output) == {
        "query": "agno agents",
        "provider": "tavily",
        "results": [
            {"title": "Example A", "url": "https://example.com/a", "content": "Short summary", "published_date": None}
        ],
        "answer": "A summary answer",
    }


def test_web_search_truncates_result_text():
    response = _search_response([_result(description="x" * 50)])
    router_cls = _sync_router(search=MagicMock(return_value=response))
    with patch.object(webrouter_module, "WebRouter", router_cls):
        capped = WebRouterTools(format="json", max_chars_per_result=10).web_search("q")
        uncapped = WebRouterTools(format="json", max_chars_per_result=None).web_search("q")

    assert json.loads(capped)["results"][0]["content"] == "x" * 10 + "..."
    assert json.loads(uncapped)["results"][0]["content"] == "x" * 50


def test_web_search_no_results():
    router_cls = _sync_router(search=MagicMock(return_value=_search_response([])))
    with patch.object(webrouter_module, "WebRouter", router_cls):
        assert WebRouterTools().web_search("nothing") == "No results found."


def test_web_search_error_is_returned():
    router_cls = _sync_router(search=MagicMock(side_effect=FakeWebRouterError("query must not be blank")))
    with patch.object(webrouter_module, "WebRouter", router_cls):
        assert WebRouterTools().web_search("   ") == "Error searching the web: query must not be blank"


@pytest.mark.asyncio
async def test_aweb_search_uses_async_router():
    search = AsyncMock(return_value=_search_response([_result()]))
    router_cls = _async_router(search=search)
    with patch.object(webrouter_module, "AsyncWebRouter", router_cls):
        output = await WebRouterTools(max_results=4).aweb_search("agno agents")

    assert "### [Example A](https://example.com/a)" in output
    assert search.await_args.kwargs["max_results"] == 4
    assert search.await_args.kwargs["data_apis"] is False


@pytest.mark.asyncio
async def test_aweb_search_error_is_returned():
    router_cls = _async_router(search=AsyncMock(side_effect=FakeWebRouterError("timed out")))
    with patch.object(webrouter_module, "AsyncWebRouter", router_cls):
        assert await WebRouterTools().aweb_search("q") == "Error searching the web: timed out"


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
def test_extract_markdown_lists_failures():
    response = _extract_response([_document()], [SimpleNamespace(url="https://example.com/b", error="404 Not Found")])
    extract = MagicMock(return_value=response)
    router_cls = _sync_router(extract=extract)
    with patch.object(webrouter_module, "WebRouter", router_cls):
        output = WebRouterTools(extract_provider="firecrawl", timeout=15).extract_url_content(
            ["https://example.com/a", "https://example.com/b"]
        )

    extract.assert_called_once_with(
        ["https://example.com/a", "https://example.com/b"], provider="firecrawl", timeout=15, data_apis=False
    )
    assert output.startswith("## Page A\n\nURL: https://example.com/a\n\nPage body\n\n")
    assert "## Could not extract\n\n- https://example.com/b: 404 Not Found\n" in output


def test_extract_json_and_page_truncation():
    response = _extract_response([_document(content="y" * 30)])
    router_cls = _sync_router(extract=MagicMock(return_value=response))
    with patch.object(webrouter_module, "WebRouter", router_cls):
        output = WebRouterTools(format="json", max_chars_per_page=5).extract_url_content(["https://example.com/a"])

    assert json.loads(output) == {
        "provider": "nimble",
        "pages": [{"url": "https://example.com/a", "title": "Page A", "content": "yyyyy..."}],
    }


def test_extract_refuses_private_hosts_without_sending_them():
    extract = MagicMock(return_value=_extract_response([_document()]))
    router_cls = _sync_router(extract=extract)
    with patch.object(webrouter_module, "WebRouter", router_cls):
        output = WebRouterTools(format="json").extract_url_content(
            ["https://example.com/a", "http://169.254.169.254/latest/meta-data/", " https://example.com/a ", 123]  # type: ignore[list-item]
        )

    # The refused URL never reaches the router; the duplicate is sent once and the non-string item is dropped.
    assert extract.call_args.args == (["https://example.com/a"],)
    assert json.loads(output)["failures"] == [
        {"url": "http://169.254.169.254/latest/meta-data/", "error": "the host is a cloud metadata endpoint"}
    ]


def test_extract_with_no_usable_url_makes_no_call():
    router_cls = _sync_router(extract=MagicMock())
    tools = WebRouterTools()
    with patch.object(webrouter_module, "WebRouter", router_cls):
        assert tools.extract_url_content([]) == "Error: No valid URLs provided."
        assert tools.extract_url_content(["file:///etc/passwd"]) == (
            "Error: No URL could be fetched. file:///etc/passwd: the URL is not http or https"
        )
    router_cls.assert_not_called()


def test_extract_error_is_returned():
    router_cls = _sync_router(extract=MagicMock(side_effect=FakeWebRouterError("no page came back")))
    with patch.object(webrouter_module, "WebRouter", router_cls):
        output = WebRouterTools().extract_url_content(["https://example.com/a"])
    assert output == "Error extracting content: no page came back"


@pytest.mark.asyncio
async def test_aextract_uses_async_router():
    extract = AsyncMock(return_value=_extract_response([_document()]))
    router_cls = _async_router(extract=extract)
    with patch.object(webrouter_module, "AsyncWebRouter", router_cls):
        output = await WebRouterTools().aextract_url_content(["https://example.com/a"])

    assert output.startswith("## Page A\n\n")
    assert extract.await_args.args == (["https://example.com/a"],)
    assert extract.await_args.kwargs["data_apis"] is False


@pytest.mark.asyncio
async def test_aextract_refuses_private_hosts_and_returns_errors():
    router_cls = _async_router(extract=AsyncMock(side_effect=FakeWebRouterError("no page came back")))
    tools = WebRouterTools()
    with patch.object(webrouter_module, "AsyncWebRouter", router_cls):
        refused = await tools.aextract_url_content(["http://169.254.169.254/"])
        failed = await tools.aextract_url_content(["https://example.com/a"])

    assert refused == "Error: No URL could be fetched. http://169.254.169.254/: the host is a cloud metadata endpoint"
    assert failed == "Error extracting content: no page came back"
    router_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Contract with the real package
# ---------------------------------------------------------------------------
def test_real_package_accepts_the_options_sent():
    web_router = pytest.importorskip("web_router")
    from web_router.routing.url_policy import public_only

    search_kwargs = set(WebRouterTools()._search_options(None))
    extract_kwargs = set(WebRouterTools()._extract_options())
    for router in (web_router.WebRouter, web_router.AsyncWebRouter):
        assert search_kwargs <= set(inspect.signature(router.search).parameters)
        assert extract_kwargs <= set(inspect.signature(router.extract).parameters)
        assert "provider_config" in inspect.signature(router).parameters
    assert public_only("http://169.254.169.254/") is not None
    assert public_only("https://example.com/") is None

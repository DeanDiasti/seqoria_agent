"""Frame SSE data fields independently of the provider's JSON event schema."""

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator


class _DataBuffer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def feed(self, line: str) -> str | None:
        if not line:
            return self.flush()
        # Comments, event names, event IDs, and retry hints are not JSON data.
        field, separator, value = line.partition(":")
        if field == "data":
            if separator and value.startswith(" "):
                value = value[1:]
            self.lines.append(value)
        return None

    def flush(self) -> str | None:
        if not self.lines:
            return None
        data = "\n".join(self.lines)
        self.lines.clear()
        return data


def iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield joined data fields for each complete sync SSE event, including EOF."""
    buffer = _DataBuffer()
    for line in lines:
        data = buffer.feed(line)
        if data is not None:
            yield data
    data = buffer.flush()
    if data is not None:
        yield data


async def aiter_sse_data(lines: AsyncIterable[str]) -> AsyncIterator[str]:
    """Yield joined data fields for each complete async SSE event, including EOF."""
    buffer = _DataBuffer()
    async for line in lines:
        data = buffer.feed(line)
        if data is not None:
            yield data
    data = buffer.flush()
    if data is not None:
        yield data

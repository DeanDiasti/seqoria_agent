# Caching

Avoid redundant tool execution by injecting a cache backend that fulfils the lightweight `CacheProtocol`.

```python
from xmcp.client.cache import RedisCache, build_cache_key

ai = AIClient(
    urls=["http://localhost:3333"],
    providers=[OpenAIProvider(model_id="gpt-4o")],
    cache=RedisCache(ttl=300),   # 5-min window
)
```

### Deterministic Keys

Order of arguments does **not** matter:

```python
args1 = {"b": 2, "a": 1}
args2 = {"a": 1, "b": 2}
assert build_cache_key("tool", args1) == build_cache_key("tool", args2)
```

### DictCache (Testing)

```python
from xmcp.client.cache import DictCache
ai = AIClient(urls=[...], providers=[...], cache=DictCache())
```

### NullCache (Disable)

```python
from xmcp.client.cache import NullCache
ai = AIClient(urls=[...], providers=[...], cache=NullCache())
```

### When it Works

1. Client computes the cache key.
2. If present, tool call is **skipped**.
3. Result is re-injected into the conversation history so the LLM behaves identically. 
# Tool-result caching

Caching is disabled unless a cache is supplied to the client. Cache only tool collections whose successful results are safe to reuse. A cache must not be enabled for side-effecting tools that need to execute on every request.

```python
from seqoria_agent.client.cache import DictCache

cache = DictCache(max_entries=1024)
# Pass cache=cache and cache_ttl=60 when constructing AsyncAIClient.
```

`DictCache` uses bounded LRU eviction and a monotonic clock for expiration. It is local to one process. The optional `RedisCache` requires the `redis` extra; close its connections with `await cache.aclose()` after client shutdown.

`DictCache` copies values on storage and retrieval, so mutating a result cannot change later cache hits. `RedisCache` requires JSON-serializable values and rejects unsupported values rather than silently converting them to strings. Invalid cache data raises an error; the executor treats cache errors as misses and executes the tool. A TTL of `None` or `0` means no expiration (`RedisCache` uses its configured default when a call supplies `None`); negative TTLs are rejected.

Executor cache keys include the tool arguments, injected parameters, full trusted request context, and a random namespace belonging to the client instance. Users and permissions within one tenant therefore have separate entries. Unserializable context disables caching for that call. A Redis backend does not share entries between independently constructed clients or workers under this policy.

Supply all identity and permission fields relevant to the result in `context`, derived from authenticated application state. The library cannot infer authorization from a remote server's response. Changed source data can still make a cached response stale; choose a suitable TTL or leave caching disabled.

Low-level `build_cache_key()` only hashes the values its caller supplies. The executor adds its isolation inputs around that helper. Cache backend errors do not prevent tool execution, and tool failures are not written as successful cache entries.

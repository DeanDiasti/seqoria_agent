# Testing & CI

This project ships with **pytest**-based unit and integration tests.  To run everything locally:

```bash
pip install -e '.[dev]'
pytest -q  # short output
```

## Folder Layout

```
 tests/
   cache/                 # DictCache, RedisCache
   client/                # AIClient orchestration
   providers/             # Provider retry + payload
   server/                # FastXMCP runtime
   streaming/             # stream_chat generator
   integration/           # end-to-end demos
   fuzz/                  # optional race-condition stress
```

## Markers

| Marker | Purpose |
|--------|---------|
| `requires_redis` | Skip tests if Redis client is not installed / reachable. |
| `integration` | Slow, real server + model stubs. |
| `fuzz` | High-concurrency stress tests. |

Configure in `pytest.ini`:

```ini
[pytest]
markers =
    requires_redis: needs redis server
    integration: slower end-to-end run
    fuzz: race-condition stress
```

## Continuous Integration

Add to **GitHub Actions**:

```yaml
- name: Run tests
  run: pytest -m "not integration and not fuzz" -q
- name: Integration tests
  run: pytest -m integration -q
```

## Writing New Tests

1. Prefer **pure functions**; stub network & provider calls.
2. Use `monkeypatch` to override slow operations.
3. Follow *Arrange → Act → Assert* and keep fixtures next to tests unless reused. 
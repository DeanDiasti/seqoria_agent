# Multi-user orchestration benchmark

Install the checkout with `python -m pip install -e '.[dev]'`, then run:

```bash
python benchmarks/multi_user.py --users 100 --rounds 5 --tool-latency-ms 5 --max-concurrency 16
```

The JSON output records the Python/platform/package version, inputs, request
throughput, mean/p50/p95/p99 latency, and peak concurrent tools. Each user has a
separate conversation history and authenticated local identity. A deterministic
HTTPX transport simulates a model that requests a tool and then answers from its
result. No network request leaves the process, and no API key is needed.

The script checks identity isolation, the configured concurrency cap, cancellation
cleanup, recovery after cancellation, HTTP-client closure, and absence of leaked
background tasks. A failed correctness check exits unsuccessfully. CI runs a small
smoke scenario and retains its JSON output; timing has no pass/fail threshold.

Compare runs using the same hardware, Python/dependency versions, and arguments.
Repeat runs to account for warmup and scheduler noise. Tool latency is an artificial
sleep, and tools are asynchronous. These numbers describe local orchestration and
queueing overhead. They do not establish hosted-model latency, database capacity,
remote MCP throughput, sync-thread cancellation, or production deployment size.

# Examples

Run these commands from the repository root with Python 3.12 or newer. Install the library first:

```bash
python -m pip install -e .
```

The examples also ship in the PyPI source distribution. Use `python -m pip install .` after extracting it. None of the examples require development extras.

## Start offline

These four examples finish automatically and need no model, API key, network access, or running service. Their scripted providers demonstrate orchestration; they are not language models.

| Example | Purpose | Run |
| --- | --- | --- |
| [Offline quickstart](quickstart/offline.py) | Generate once, stream deltas, and inspect illustrative usage counts | `python examples/quickstart/offline.py` |
| [Local tools](tools/local_tools.py) | Declare a validated schema, call tools directly, and handle Pydantic validation errors | `python examples/tools/local_tools.py` |
| [Manual agent loop](tools/manual_loop.py) | Own a bounded model/tool loop and append requests, outcomes, and replies to history | `python examples/tools/manual_loop.py` |
| [Concurrent requests](conversation/request_context.py) | Share a client while isolating conversation histories and trusted identity parameters | `python examples/conversation/request_context.py` |

The concurrent example supplies identity through authenticated application context and `ParameterRegistry`. A web application must provide its own authentication, authorization, conversation storage, and serialization of simultaneous writes to the same conversation.

## MCP tools without a model

[The stdio client](mcp/stdio_client.py) launches [the utility server](mcp/server2.py) as a child process, calls `divide`, and closes both the session and process. It needs no API key, network service, or open TCP port:

```bash
python examples/mcp/stdio_client.py
```

To try HTTP transport instead, start the official MCP SDK servers in separate terminals. These local demonstration servers bind to loopback and provide no authentication:

```bash
python examples/mcp/server1.py
```

```bash
python examples/mcp/server2.py
```

Then run [the HTTP client](mcp/client.py):

```bash
python examples/mcp/client.py
```

The servers listen on `127.0.0.1:8000/mcp` and `127.0.0.1:8005/mcp`. The client discovers their tools, computes `8 / 2`, and exits. Stop the HTTP servers with Ctrl-C. Both servers also support `--transport stdio` for use by a process launcher. The examples use the official SDK directly; Seqoria Agent supplies the client orchestration.

## External model providers

[The provider streaming example](providers/stream.py) makes one model request and exits. Choose an installed or accessible model explicitly; the library does not choose one or download Ollama models for this example. Hosted requests use your provider account and may incur charges.

Set `SEQORIA_MODEL` to your model ID, then configure the selected provider:

| `--provider` | Required environment | Other prerequisites |
| --- | --- | --- |
| `openai` | `SEQORIA_MODEL`, `OPENAI_API_KEY` | Access to the selected model through OpenAI's Responses API |
| `anthropic` | `SEQORIA_MODEL`, `ANTHROPIC_API_KEY` | Access to the selected model through Anthropic's Messages API |
| `compatible` | `SEQORIA_MODEL`, `SEQORIA_API_KEY`, `SEQORIA_BASE_URL` | An OpenAI-compatible Chat Completions endpoint; base URL excludes `/chat/completions` |
| `ollama` | `SEQORIA_MODEL` | A running Ollama server and an already pulled model; `OLLAMA_HOST` optionally overrides localhost |

For example, after setting the required environment variables:

```bash
python examples/providers/stream.py --provider openai
python examples/providers/stream.py --provider anthropic
python examples/providers/stream.py --provider compatible
python examples/providers/stream.py --provider ollama --prompt "Explain bounded tool concurrency."
```

Keep credentials in environment variables. Each example owns and closes its provider, client, or MCP session; a shared client belongs to the application lifetime, while history and identity belong to the request or conversation.

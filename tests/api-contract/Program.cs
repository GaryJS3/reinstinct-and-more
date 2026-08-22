using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

var options = ContractOptions.Parse(args);
if (options.ShowHelp)
{
    ContractOptions.PrintHelp();
    return 0;
}

using var client = new ContractClient(options.BaseUri, options.Timeout);
var suite = new ContractSuite(client, options);
return await suite.RunAsync();

sealed class ContractOptions
{
    public required Uri BaseUri { get; init; }
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(120);
    public bool ShowHelp { get; init; }

    public static ContractOptions Parse(string[] args)
    {
        if (args.Any(a => a is "--help" or "-h"))
            return new ContractOptions { ShowHelp = true, BaseUri = new Uri("http://127.0.0.1/") };

        string? baseUrl = null;
        var timeout = TimeSpan.FromSeconds(120);
        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--base-url" when i + 1 < args.Length:
                    baseUrl = args[++i];
                    break;
                case "--timeout-seconds" when i + 1 < args.Length
                    && double.TryParse(args[++i], out var seconds) && seconds > 0:
                    timeout = TimeSpan.FromSeconds(seconds);
                    break;
                default:
                    throw new ArgumentException($"Unknown or incomplete option '{args[i]}'. Use --help for usage.");
            }
        }

        baseUrl ??= Environment.GetEnvironmentVariable("REINSTINCT_BASE_URL");
        if (string.IsNullOrWhiteSpace(baseUrl))
            throw new ArgumentException("Provide --base-url or set REINSTINCT_BASE_URL.");
        if (!Uri.TryCreate(baseUrl, UriKind.Absolute, out var uri)
            || uri.Scheme is not ("http" or "https"))
            throw new ArgumentException($"Base URL must be an absolute http(s) URL: '{baseUrl}'.");

        return new ContractOptions { BaseUri = EnsureTrailingSlash(uri), Timeout = timeout };
    }

    public static void PrintHelp()
    {
        Console.WriteLine("ReInstinct HTTP contract suite");
        Console.WriteLine();
        Console.WriteLine("Usage: dotnet run --project tests/api-contract -- --base-url http://127.0.0.1:8006");
        Console.WriteLine("       REINSTINCT_BASE_URL=http://127.0.0.1:8006 dotnet run --project tests/api-contract");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --base-url URL          HTTP endpoint to test (or REINSTINCT_BASE_URL)");
        Console.WriteLine("  --timeout-seconds N     Per-request client timeout; default 120");
        Console.WriteLine("  --help                  Show this help");
    }

    private static Uri EnsureTrailingSlash(Uri uri)
    {
        var text = uri.AbsoluteUri.EndsWith('/') ? uri.AbsoluteUri : uri.AbsoluteUri + "/";
        return new Uri(text, UriKind.Absolute);
    }
}

sealed class ContractClient : IDisposable
{
    private readonly HttpClient _http;

    public ContractClient(Uri baseUri, TimeSpan timeout)
    {
        _http = new HttpClient { BaseAddress = baseUri, Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("reinstinct-api-contract/1.0");
        // Authentication is intentionally a reverse-proxy concern. OpenAI
        // SDKs commonly send a bearer token even when the local backend has
        // auth disabled, so the backend must tolerate this header.
        _http.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", "contract-test-noauth");
        Timeout = timeout;
    }

    public TimeSpan Timeout { get; }

    public async Task<HttpResponseMessage> SendJsonAsync(
        HttpMethod method, string path, object? payload, CancellationToken cancellationToken = default,
        string? actionHeader = null)
    {
        using var request = new HttpRequestMessage(method, path);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        if (actionHeader is not null)
            request.Headers.TryAddWithoutValidation("X-ReInstinct-Action", actionHeader);
        if (payload is not null)
        {
            var json = JsonSerializer.Serialize(payload, JsonOptions.Default);
            request.Content = new StringContent(json, Encoding.UTF8, "application/json");
        }

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(Timeout);
        return await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
    }

    public async Task<HttpResponseMessage> SendStreamingAsync(object payload, CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "v1/chat/completions");
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        var json = JsonSerializer.Serialize(payload, JsonOptions.Default);
        request.Content = new StringContent(json, Encoding.UTF8, "application/json");

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(Timeout);
        return await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
    }

    public void Dispose() => _http.Dispose();
}

sealed class ContractSuite
{
    private const string DefaultPrompt = "Reply with one short sentence about contract testing.";
    private readonly ContractClient _client;
    private readonly ContractOptions _options;
    private string? _model;

    public ContractSuite(ContractClient client, ContractOptions options)
    {
        _client = client;
        _options = options;
    }

    public async Task<int> RunAsync()
    {
        var tests = new (string Name, Func<Task> Test)[]
        {
            ("status page and OpenAPI", StatusAndOpenApiAsync),
            ("operations telemetry and model catalog", OperationsAsync),
            ("health", HealthAsync),
            ("readiness", ReadinessAsync),
            ("model discovery", ModelsAsync),
            ("model retrieval", ModelRetrievalAsync),
            ("text chat", TextChatAsync),
            ("run history and detail", RunHistoryAsync),
            ("text streaming and usage", TextStreamingAsync),
            ("JPEG image chat", () => ImageChatAsync("jpeg")),
            ("PNG image chat", () => ImageChatAsync("png")),
            ("invalid request errors", InvalidRequestsAsync),
            ("request compatibility", RequestCompatibilityAsync),
            ("decoded-image limit", DecodedImageLimitAsync),
            ("request-size limit", RequestSizeLimitAsync),
            ("timeout recovery", TimeoutRecoveryAsync),
            ("client disconnect recovery", DisconnectRecoveryAsync),
        };

        var failures = 0;
        Console.WriteLine($"Endpoint: {_options.BaseUri}");
        foreach (var (name, test) in tests)
        {
            var stopwatch = Stopwatch.StartNew();
            try
            {
                await test();
                Console.WriteLine($"PASS  {name} ({stopwatch.Elapsed.TotalMilliseconds:0} ms)");
            }
            catch (Exception ex)
            {
                failures++;
                Console.WriteLine($"FAIL  {name}: {ex.Message}");
            }
        }

        Console.WriteLine(failures == 0
            ? "Contract suite passed."
            : $"Contract suite failed: {failures} test(s).");
        return failures == 0 ? 0 : 1;
    }

    private async Task StatusAndOpenApiAsync()
    {
        using (var index = await _client.SendJsonAsync(HttpMethod.Get, "", null))
        {
            var html = await index.Content.ReadAsStringAsync();
            Check(index.StatusCode == HttpStatusCode.OK, $"status page expected 200, got {(int)index.StatusCode}");
            Check(index.Content.Headers.ContentType?.MediaType == "text/html", "root must return HTML");
            Check(html.Contains("ReInstinct", StringComparison.Ordinal), "status page branding missing");
            Check(html.Contains("Recent runs", StringComparison.Ordinal), "run history table missing");
            Check(html.Contains("Engine logs", StringComparison.Ordinal), "engine log panel missing");
            Check(html.Contains("Model catalog", StringComparison.Ordinal), "model catalog panel missing");
        }

        using (var status = await _client.SendJsonAsync(HttpMethod.Get, "api/status", null))
        using (var document = await ReadJsonAsync(status, HttpStatusCode.OK))
        {
            var root = document.RootElement;
            Check(root.GetProperty("service").GetProperty("name").GetString() == "reinstinct",
                "status service name is invalid");
            Check(root.GetProperty("model").GetProperty("max_context_tokens").GetInt64() > 0,
                "status context setting is missing");
            Check(root.GetProperty("gpu").ValueKind == JsonValueKind.Object, "status GPU object missing");
            Check(root.GetProperty("performance").ValueKind == JsonValueKind.Object,
                "status performance object missing");
            Check(root.GetProperty("run_history").GetProperty("capacity").GetInt32() >= 1,
                "status run-history metadata missing");
            Check(root.GetProperty("network").GetProperty("receive_mbps").GetDouble() >= 0,
                "status network telemetry missing");
            Check(root.GetProperty("logs").GetProperty("capacity").GetInt32() >= 100,
                "log metadata missing");
            Check(root.GetProperty("management").GetProperty("switch_requires_action_header").GetBoolean(),
                "model-switch action-header guard missing");
        }

        using (var spec = await _client.SendJsonAsync(HttpMethod.Get, "openapi.json", null))
        using (var document = await ReadJsonAsync(spec, HttpStatusCode.OK))
        {
            var root = document.RootElement;
            Check(root.GetProperty("openapi").GetString() == "3.1.0", "OpenAPI version must be 3.1.0");
            Check(root.GetProperty("paths").TryGetProperty("/v1/chat/completions", out _),
                "OpenAPI chat-completions operation missing");
            Check(root.GetProperty("paths").TryGetProperty("/api/status", out _),
                "OpenAPI status operation missing");
            Check(root.GetProperty("paths").TryGetProperty("/api/runs", out _),
                "OpenAPI run-list operation missing");
            Check(root.GetProperty("paths").TryGetProperty("/api/runs/{id}", out _),
                "OpenAPI run-detail operation missing");
            Check(root.GetProperty("paths").TryGetProperty("/api/logs", out _),
                "OpenAPI log operation missing");
            Check(root.GetProperty("paths").TryGetProperty("/api/models", out _),
                "OpenAPI model-catalog operation missing");
            Check(root.GetProperty("paths").TryGetProperty("/readyz", out _),
                "OpenAPI readiness operation missing");
        }
    }

    private async Task OperationsAsync()
    {
        using (var response = await _client.SendJsonAsync(HttpMethod.Get, "api/logs", null))
        using (var document = await ReadJsonAsync(response, HttpStatusCode.OK))
        {
            var root = document.RootElement;
            Check(root.GetProperty("capacity").GetInt32() >= 100, "log capacity missing");
            Check(root.GetProperty("lines").ValueKind == JsonValueKind.Array, "log lines missing");
        }

        using (var response = await _client.SendJsonAsync(HttpMethod.Get, "api/models", null))
        using (var document = await ReadJsonAsync(response, HttpStatusCode.OK))
        {
            var root = document.RootElement;
            Check(root.GetProperty("root").GetString() is { Length: > 0 }, "model catalog root missing");
            var models = root.GetProperty("models");
            Check(models.ValueKind == JsonValueKind.Array, "model catalog entries missing");
            if (models.GetArrayLength() > 0)
            {
                var first = models[0];
                Check(first.GetProperty("path").GetString() is { Length: > 0 }, "model real path missing");
                Check(first.GetProperty("size_bytes").GetInt64() > 0, "model size missing");
                Check(first.TryGetProperty("image_projector", out _), "projector association field missing");
            }
        }

        using (var cached = await _client.SendJsonAsync(HttpMethod.Get, "api/models", null))
        using (var document = await ReadJsonAsync(cached, HttpStatusCode.OK))
            Check(document.RootElement.GetProperty("cached").GetBoolean(), "catalog cache was not reused");

        using var invalid = await _client.SendJsonAsync(HttpMethod.Post, "api/models/switch",
            new { path = "/outside/catalog/not-a-model.gguf" });
        Check(invalid.StatusCode == HttpStatusCode.BadRequest,
            $"missing-action switch expected 400, got {(int)invalid.StatusCode}");
        using var invalidPath = await _client.SendJsonAsync(HttpMethod.Post, "api/models/switch",
            new { path = "/outside/catalog/not-a-model.gguf" }, actionHeader: "switch-model");
        Check(invalidPath.StatusCode == HttpStatusCode.BadRequest,
            $"out-of-catalog switch expected 400, got {(int)invalidPath.StatusCode}");
    }

    private async Task HealthAsync()
    {
        using var response = await _client.SendJsonAsync(HttpMethod.Get, "healthz", null);
        var body = await response.Content.ReadAsStringAsync();
        Check(response.StatusCode == HttpStatusCode.OK, $"expected 200, got {(int)response.StatusCode}");
        Check(body.Trim() == "ok", $"expected health body 'ok', got '{body.Trim()}'");
    }

    private async Task ReadinessAsync()
    {
        using var response = await _client.SendJsonAsync(HttpMethod.Get, "readyz", null);
        var body = await response.Content.ReadAsStringAsync();
        Check(response.StatusCode == HttpStatusCode.OK, $"expected readyz 200, got {(int)response.StatusCode}");
        Check(body.Trim() == "ready", $"expected ready body 'ready', got '{body.Trim()}'");
    }

    private async Task ModelsAsync()
    {
        using var response = await _client.SendJsonAsync(HttpMethod.Get, "v1/models", null);
        using var document = await ReadJsonAsync(response, HttpStatusCode.OK);
        var root = document.RootElement;
        Check(root.GetProperty("object").GetString() == "list", "models.object must be list");
        var data = root.GetProperty("data");
        Check(data.ValueKind == JsonValueKind.Array, "models.data must be an array");
        Check(data.GetArrayLength() > 0, "model discovery returned no models");
        var model = data[0];
        Check(model.GetProperty("object").GetString() == "model", "model.object must be model");
        _model = model.GetProperty("id").GetString();
        Check(!string.IsNullOrWhiteSpace(_model), "model.id must be non-empty");
    }

    private async Task ModelRetrievalAsync()
    {
        Check(!string.IsNullOrWhiteSpace(_model), "model discovery must run first");
        using (var response = await _client.SendJsonAsync(
            HttpMethod.Get, $"v1/models/{Uri.EscapeDataString(_model!)}", null))
        using (var document = await ReadJsonAsync(response, HttpStatusCode.OK))
        {
            Check(document.RootElement.GetProperty("id").GetString() == _model,
                "model retrieval returned the wrong model");
            Check(document.RootElement.GetProperty("object").GetString() == "model",
                "model retrieval object must be model");
        }

        await AssertErrorAsync(HttpMethod.Get, "v1/models/not-the-loaded-model", null,
            HttpStatusCode.NotFound, "unknown model retrieval", "model", "model_not_found");
    }

    private async Task TextChatAsync()
    {
        using var response = await _client.SendJsonAsync(HttpMethod.Post, "v1/chat/completions",
            ChatPayload(DefaultPrompt, maxTokens: 4));
        using var document = await ReadJsonAsync(response, HttpStatusCode.OK);
        ValidateChatCompletion(document.RootElement);
    }

    private async Task RunHistoryAsync()
    {
        long id;
        using (var response = await _client.SendJsonAsync(HttpMethod.Get, "api/runs", null))
        using (var document = await ReadJsonAsync(response, HttpStatusCode.OK))
        {
            var root = document.RootElement;
            Check(root.GetProperty("capacity").GetInt32() >= 1, "run-history capacity missing");
            var runs = root.GetProperty("runs");
            Check(runs.GetArrayLength() > 0, "completed text request was not retained");
            var run = runs[0];
            id = run.GetProperty("id").GetInt64();
            Check(run.GetProperty("client_ip").GetString() is { Length: > 0 },
                "run client IP missing");
            Check(run.GetProperty("state").GetString() == "complete", "run must be complete");
            var stats = run.GetProperty("stats");
            Check(stats.GetProperty("prompt_tokens").GetInt64() > 0, "run prompt tokens missing");
            Check(stats.GetProperty("completion_tokens").GetInt64() > 0, "run completion tokens missing");
            Check(stats.GetProperty("prompt_tokens_per_second").GetDouble() > 0,
                "run prompt throughput missing");
            Check(stats.GetProperty("generation_tokens_per_second").GetDouble() > 0,
                "run generation throughput missing");
        }

        using (var response = await _client.SendJsonAsync(HttpMethod.Get, $"api/runs/{id}", null))
        using (var document = await ReadJsonAsync(response, HttpStatusCode.OK))
        {
            var root = document.RootElement;
            Check(root.GetProperty("request").ValueKind == JsonValueKind.Object,
                "run detail request missing");
            Check(root.GetProperty("response").ValueKind == JsonValueKind.Object,
                "run detail response missing");
        }

        await AssertErrorAsync(HttpMethod.Get, "api/runs/999999999", null,
            HttpStatusCode.NotFound, "unknown run", "id", "run_not_found");
    }

    private async Task TextStreamingAsync()
    {
        using var response = await _client.SendStreamingAsync(ChatPayload(
            "Reply with two short words.", maxTokens: 8, stream: true, includeUsage: true));
        Check(response.StatusCode == HttpStatusCode.OK,
            $"expected 200, got {(int)response.StatusCode}: {await response.Content.ReadAsStringAsync()}");
        Check(response.Content.Headers.ContentType?.MediaType == "text/event-stream",
            "stream response must be text/event-stream");

        await using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);
        var chunks = new List<JsonElement>();
        var sawDone = false;
        while (await reader.ReadLineAsync() is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal))
                continue;
            var data = line[6..];
            if (data == "[DONE]")
            {
                sawDone = true;
                break;
            }
            using var chunk = JsonDocument.Parse(data);
            chunks.Add(chunk.RootElement.Clone());
        }

        Check(sawDone, "SSE stream did not terminate with data: [DONE]");
        Check(chunks.Count >= 2, "SSE stream must contain role and content/finish chunks");
        Check(chunks[0].GetProperty("object").GetString() == "chat.completion.chunk",
            "first SSE object must be chat.completion.chunk");
        var firstDelta = chunks[0].GetProperty("choices")[0].GetProperty("delta");
        Check(firstDelta.GetProperty("role").GetString() == "assistant",
            "first SSE chunk must carry assistant role");
        Check(chunks.Any(c => c.GetProperty("choices")[0].GetProperty("delta")
            .TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String),
            "SSE stream must carry content deltas");
        Check(chunks.Any(c => c.GetProperty("choices")[0].GetProperty("finish_reason").ValueKind == JsonValueKind.String),
            "SSE stream must carry a finish_reason");

        var usage = chunks.FirstOrDefault(c => c.TryGetProperty("usage", out _));
        Check(usage.ValueKind != JsonValueKind.Undefined, "include_usage must emit a usage chunk");
        ValidateUsage(usage.GetProperty("usage"));
    }

    private async Task ImageChatAsync(string format)
    {
        var fixture = Fixtures.For(format);
        var prompt = $"Describe this {format} fixture in one short sentence.";
        using var response = await _client.SendJsonAsync(HttpMethod.Post, "v1/chat/completions",
            ChatPayload(prompt, maxTokens: 8, image: fixture.DataUrl));
        using var document = await ReadJsonAsync(response, HttpStatusCode.OK);
        ValidateChatCompletion(document.RootElement);
        var text = document.RootElement.GetProperty("choices")[0].GetProperty("message")
            .GetProperty("content").GetString();
        Check(!string.IsNullOrWhiteSpace(text), $"{format} response content was empty");
    }

    private async Task InvalidRequestsAsync()
    {
        await AssertErrorAsync("v1/chat/completions", "{", HttpStatusCode.BadRequest, "invalid JSON");
        await AssertErrorAsync("v1/chat/completions", ImagePayload("data:image/png;base64,not-base64"),
            HttpStatusCode.BadRequest, "invalid base64");
        await AssertErrorAsync("v1/chat/completions", ImagePayload("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP"),
            HttpStatusCode.BadRequest, "image type");
        await AssertErrorAsync("v1/chat/completions", ImagePayload("data:image/png;base64,"),
            HttpStatusCode.BadRequest, "empty image");
        await AssertErrorAsync("v1/chat/completions", ImagePayload("https://example.test/image.png"),
            HttpStatusCode.BadRequest, "remote image");
        var multiple = new
        {
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = new object[]
                    {
                        new { type = "image_url", image_url = new { url = Fixtures.Png.DataUrl } },
                        new { type = "image_url", image_url = new { url = Fixtures.Jpeg.DataUrl } },
                    },
                },
            },
        };
        await AssertErrorAsync("v1/chat/completions", multiple, HttpStatusCode.BadRequest, "multiple images");
        await AssertErrorAsync("v1/no-such-route", new { }, HttpStatusCode.NotFound, "unknown route");
    }

    private async Task RequestCompatibilityAsync()
    {
        const string stopMarker = "__CONTRACT_STOP_MARKER__";
        using (var response = await _client.SendJsonAsync(HttpMethod.Post, "v1/chat/completions",
                   BasicChatPayload("Reply briefly.", maxCompletionTokens: 2, stop: stopMarker)))
        using (var document = await ReadJsonAsync(response, HttpStatusCode.OK))
        {
            ValidateChatCompletion(document.RootElement);
            var content = document.RootElement.GetProperty("choices")[0]
                .GetProperty("message").GetProperty("content").GetString() ?? "";
            Check(!content.Contains(stopMarker, StringComparison.Ordinal),
                "stop marker must not appear in the response");
        }

        await AssertErrorAsync("v1/chat/completions",
            BasicChatPayload("Reply briefly.", maxTokens: 1, maxCompletionTokens: 2),
            HttpStatusCode.BadRequest, "conflicting token limits");
        await AssertErrorAsync("v1/chat/completions",
            BasicChatPayload("Reply briefly.", n: 2),
            HttpStatusCode.BadRequest, "multiple choices");
        await AssertErrorAsync("v1/chat/completions",
            new
            {
                model = _model,
                messages = new[] { new { role = "user", content = "Reply briefly." } },
                tools = Array.Empty<object>(),
            }, HttpStatusCode.BadRequest, "unsupported tools");
    }

    private async Task RequestSizeLimitAsync()
    {
        // Keep the request just above the documented 8 MiB HTTP cap. The
        // server must reject it before attempting to parse JSON or enqueue it.
        var oversized = new { prompt = new string('x', 8 * 1024 * 1024) };
        await AssertErrorAsync("v1/completions", oversized, HttpStatusCode.BadRequest, "8 MiB body");
    }

    private async Task DecodedImageLimitAsync()
    {
        // This intentionally crosses the documented 6 MiB decoded-image
        // boundary. Depending on JSON/base64 overhead, the independent 8 MiB
        // HTTP body guard may reject it first; both are required 4xx guards.
        var bytes = new byte[6 * 1024 * 1024 + 1];
        var dataUrl = $"data:image/png;base64,{Convert.ToBase64String(bytes)}";
        await AssertErrorAsync("v1/chat/completions", ImagePayload(dataUrl),
            HttpStatusCode.BadRequest, "6 MiB decoded image");
    }

    private async Task TimeoutRecoveryAsync()
    {
        var payload = ChatPayload("Generate a long numbered list.", maxTokens: 4096,
            timeoutSeconds: 0.1);
        using var response = await _client.SendJsonAsync(HttpMethod.Post, "v1/chat/completions", payload);
        using var document = await ReadJsonAsync(response, HttpStatusCode.OK);
        ValidateChatCompletion(document.RootElement);

        // A timed-out generation must not strand the single worker.
        await TextChatAsync();
    }

    private async Task DisconnectRecoveryAsync()
    {
        using var cancellation = new CancellationTokenSource(_client.Timeout);
        using var response = await _client.SendStreamingAsync(ChatPayload(
            "Generate several short paragraphs.", maxTokens: 128, stream: true), cancellation.Token);
        Check(response.StatusCode == HttpStatusCode.OK,
            $"disconnect probe expected 200, got {(int)response.StatusCode}");
        await using (var stream = await response.Content.ReadAsStreamAsync(cancellation.Token))
        using (var reader = new StreamReader(stream))
        {
            while (await reader.ReadLineAsync(cancellation.Token) is { } line)
            {
                if (line.StartsWith("data: ", StringComparison.Ordinal))
                    break;
            }
        }
        // Disposing the response closes the socket before the stream ends.
        await Task.Delay(250);
        await TextChatAsync();
    }

    private async Task AssertErrorAsync(string path, object payload, HttpStatusCode expected,
        string label, string? expectedParam = null, string? expectedCode = null)
    {
        await AssertErrorAsync(HttpMethod.Post, path, payload, expected, label,
            expectedParam, expectedCode);
    }

    private async Task AssertErrorAsync(HttpMethod method, string path, object? payload,
        HttpStatusCode expected, string label, string? expectedParam = null,
        string? expectedCode = null)
    {
        using var response = await _client.SendJsonAsync(method, path, payload);
        var body = await response.Content.ReadAsStringAsync();
        Check(response.StatusCode == expected,
            $"{label}: expected {(int)expected}, got {(int)response.StatusCode}: {body}");
        using var document = JsonDocument.Parse(body);
        var error = document.RootElement.GetProperty("error");
        Check(error.GetProperty("type").ValueKind == JsonValueKind.String,
            $"{label}: error.type missing");
        Check(!string.IsNullOrWhiteSpace(error.GetProperty("message").GetString()),
            $"{label}: error.message missing");
        Check(error.TryGetProperty("param", out _), $"{label}: error.param missing");
        Check(error.TryGetProperty("code", out _), $"{label}: error.code missing");
        if (expectedParam is not null)
            Check(error.GetProperty("param").GetString() == expectedParam,
                $"{label}: error.param mismatch");
        if (expectedCode is not null)
            Check(error.GetProperty("code").GetString() == expectedCode,
                $"{label}: error.code mismatch");
    }

    private async Task<JsonDocument> ReadJsonAsync(HttpResponseMessage response, HttpStatusCode expected)
    {
        var body = await response.Content.ReadAsStringAsync();
        Check(response.StatusCode == expected,
            $"expected {(int)expected}, got {(int)response.StatusCode}: {body}");
        try
        {
            return JsonDocument.Parse(body);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"response was not valid JSON: {ex.Message}");
        }
    }

    private void ValidateChatCompletion(JsonElement root)
    {
        Check(root.GetProperty("object").GetString() == "chat.completion",
            "object must be chat.completion");
        Check(root.GetProperty("id").GetString() is { Length: > 0 }, "id must be non-empty");
        Check(root.GetProperty("model").GetString() is { Length: > 0 }, "model must be non-empty");
        var choices = root.GetProperty("choices");
        Check(choices.GetArrayLength() == 1, "expected exactly one choice");
        var choice = choices[0];
        Check(choice.GetProperty("message").GetProperty("role").GetString() == "assistant",
            "message.role must be assistant");
        Check(choice.GetProperty("message").GetProperty("content").ValueKind == JsonValueKind.String,
            "message.content must be a string");
        Check(choice.GetProperty("finish_reason").GetString() is "stop" or "length",
            "finish_reason must be stop or length");
        ValidateUsage(root.GetProperty("usage"));
    }

    private static void ValidateUsage(JsonElement usage)
    {
        var prompt = usage.GetProperty("prompt_tokens").GetInt64();
        var completion = usage.GetProperty("completion_tokens").GetInt64();
        var total = usage.GetProperty("total_tokens").GetInt64();
        Check(prompt >= 0 && completion >= 0 && total == prompt + completion,
            "usage token counts are inconsistent");
    }

    private object ChatPayload(string prompt, int maxTokens, bool stream = false,
        bool includeUsage = false, string? image = null, double? timeoutSeconds = null)
    {
        object content = image is null
            ? prompt
            : new object[]
            {
                new { type = "text", text = prompt },
                new { type = "image_url", image_url = new { url = image } },
            };
        return new
        {
            model = _model ?? "contract-test-model",
            messages = new[] { new { role = "user", content } },
            max_tokens = maxTokens,
            temperature = 0.0,
            seed = 7,
            use_speculative = false,
            stream,
            stream_options = includeUsage ? new { include_usage = true } : null,
            request_timeout_seconds = timeoutSeconds,
        };
    }

    private object BasicChatPayload(string prompt, int? maxTokens = null,
        int? maxCompletionTokens = null, int? n = null, string? stop = null) => new
    {
        model = _model ?? "contract-test-model",
        messages = new[] { new { role = "user", content = prompt } },
        max_tokens = maxTokens,
        max_completion_tokens = maxCompletionTokens,
        n,
        stop,
        temperature = 0.0,
        use_speculative = false,
    };

    private object ImagePayload(string url) => new
    {
        model = _model ?? "contract-test-model",
        messages = new[]
        {
            new
            {
                role = "user",
                content = new object[]
                {
                    new { type = "text", text = "What is shown?" },
                    new { type = "image_url", image_url = new { url } },
                },
            },
        },
        max_tokens = 2,
        use_speculative = false,
    };

    private static void Check(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
    }
}

static class Fixtures
{
    public static readonly Fixture Png = new(
        "png",
        Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="),
        "image/png");

    public static readonly Fixture Jpeg = new(
        "jpeg",
        Convert.FromBase64String("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AX//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AX//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Av/EABQQAQAAAAAAAAAAAAAAAAAAACD/2gAIAQEAAT8hP//Z"),
        "image/jpeg");

    public static Fixture For(string format) => format switch
    {
        "png" => Png,
        "jpeg" => Jpeg,
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    };
}

sealed record Fixture(string Format, byte[] Bytes, string MimeType)
{
    public string DataUrl => $"data:{MimeType};base64,{Convert.ToBase64String(Bytes)}";
}

static class JsonOptions
{
    public static readonly JsonSerializerOptions Default = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}

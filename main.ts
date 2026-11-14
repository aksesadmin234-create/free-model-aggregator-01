// free-model-aggregator-01 competitive entry
// Improvements over starter:
// 1. QUALITY (60%): Route to highest rating_milli free model meeting requirements
// 2. SERVED (15%): Track per-provider call counts, spread across providers
// 3. FAILOVER (10%): Retry on 402/429/5xx with next-best model from different provider
// 4. WIRE (10%): Support all three upstream wires (openai, anthropic, gemini)
// 5. CATALOG (5%): Inherited from starter's probe logic

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicUpstreamToOpenAI,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicUpstream,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

// Budget tracking: how many calls each provider has used
const providerCallCount = new Map<string, number>();
// Exhausted providers (got 402/429)
const exhaustedProviders = new Set<string>();
// Per-provider concurrency lock — one call at a time per provider
const providerBusy = new Set<string>();

/**
 * For each future manifest entry, compute which provider IDs can serve it.
 * Returns a set of provider IDs that should be preserved for future constrained requests.
 * A provider is scarce if the remaining demand for a constraint exceeds the supply slack.
 */
function scarceProviders(currentRequires: Requirement): Set<string> {
  const scarce = new Set<string>();
  const manifest = brief.workload?.manifest ?? [];

  // Group future requests by their constraint fingerprint
  const constraintGroups = new Map<string, { requires: Requirement; count: number; capable: Set<string> }>();
  for (let i = manifestIndex; i < manifest.length; i++) {
    const futureReq = manifest[i].requires;
    // Only care about constrained requests
    if (!futureReq.vision && !futureReq.json && !futureReq.tools && !futureReq.min_context) continue;
    const key = JSON.stringify(futureReq);
    const existing = constraintGroups.get(key);
    if (existing) { existing.count++; continue; }

    // Find which non-exhausted providers can serve this
    const capable = new Set<string>();
    for (const probe of probes) {
      if (!probe.ok) continue;
      if (exhaustedProviders.has(probe.provider.id)) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(futureReq, capabilitiesOf(model, row))) continue;
        capable.add(probe.provider.id);
        break;
      }
    }
    constraintGroups.set(key, { requires: futureReq, count: 1, capable });
  }

  // Mark providers as scarce when demand approaches supply
  // Conservative: assume each provider can handle ~1 more successful call
  for (const { count, capable } of constraintGroups.values()) {
    let availableSlots = 0;
    for (const pid of capable) {
      const used = providerCallCount.get(pid) ?? 0;
      availableSlots += Math.max(0, 2 - used); // optimistic 2 calls, minus already used
    }
    // If future demand is close to available supply, mark as scarce
    // Use generous threshold: preserve when demand >= half the remaining slots
    if (count * 2 >= availableSlots) {
      for (const pid of capable) scarce.add(pid);
    }
  }
  return scarce;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** All free models meeting requirements, sorted by rating_milli descending, excluding given providers. */
function rankedRoutes(requires: Requirement, excludeProviders?: Set<string>): Route[] {
  const candidates: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (excludeProviders?.has(probe.provider.id)) continue;
    if (exhaustedProviders.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      candidates.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  // Compute which providers are scarce for future constrained requests
  const scarce = scarceProviders(requires);

  // Primary: highest rating. Tiebreak: fewer calls used → spread budget across providers. Then id for determinism.
  // Penalize scarce providers when this request doesn't strictly need them (i.e., there are non-scarce alternatives).
  const hasNonScarce = candidates.some((c) => !scarce.has(c.probe.provider.id));
  candidates.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (hasNonScarce) {
      const aS = scarce.has(a.probe.provider.id) ? 1 : 0;
      const bS = scarce.has(b.probe.provider.id) ? 1 : 0;
      if (aS !== bS) return aS - bS; // non-scarce first on tie
    }
    const aUsed = providerCallCount.get(a.probe.provider.id) ?? 0;
    const bUsed = providerCallCount.get(b.probe.provider.id) ?? 0;
    if (aUsed !== bUsed) return aUsed - bUsed; // prefer less-used provider
    if (a.probe.provider.id < b.probe.provider.id) return -1;
    if (a.probe.provider.id > b.probe.provider.id) return 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return candidates;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch { return null; }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Serialize access: one outstanding call per provider. */
async function acquireProvider(providerId: string): Promise<void> {
  while (providerBusy.has(providerId)) {
    await new Promise((r) => setTimeout(r, 5));
  }
  providerBusy.add(providerId);
}
function releaseProvider(providerId: string): void { providerBusy.delete(providerId); }

// ─── Upstream call dispatchers for all three wires ───

async function callOpenAIWire(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
}

async function callAnthropicWire(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'anthropic-version': '2023-06-01',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(body) });
}

async function callGeminiWire(route: Route, body: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  const endpoint = stream
    ? `${base}/v1beta/models/${route.model.id}:streamGenerateContent?alt=sse`
    : `${base}/v1beta/models/${route.model.id}:generateContent`;
  return fetch(endpoint, { method: 'POST', headers, body: JSON.stringify(body) });
}

function isRetryable(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

function handleFailure(providerId: string, status: number, errText: string): void {
  // Only mark permanently exhausted if budget is spent, provider has outage, or payment required (402)
  if (status === 402 || errText.includes('budget') || errText.includes('is out') || errText.includes('exhausted')) {
    exhaustedProviders.add(providerId);
    log(`${providerId} marked permanently exhausted (${status})`);
  }
}

/** Drain a response body to avoid leaking connections. */
async function drainBody(resp: Response): Promise<string> {
  try { return await resp.text(); } catch { return ''; }
}

/**
 * Unified upstream call: translates to the provider's wire, returns OpenAI-format result.
 * For streaming: returns either a raw ReadableStream (openai-wire) or an AsyncGenerator<string>
 * of OpenAI-format SSE text (anthropic/gemini translated).
 */
interface UpstreamResult {
  status: number;
  openAIBody?: Json;
  /** ReadableStream<Uint8Array> for openai-wire, AsyncGenerator<string> for translated wires */
  stream?: ReadableStream<Uint8Array> | AsyncGenerator<string>;
  isRawStream?: boolean; // true = ReadableStream bytes, false = generator of SSE text strings
}

async function callUpstream(route: Route, openAIBody: Json, isStream: boolean): Promise<UpstreamResult> {
  const wire = route.probe.provider.wire;
  const pid = route.probe.provider.id;
  providerCallCount.set(pid, (providerCallCount.get(pid) ?? 0) + 1);

  if (wire === 'openai') {
    const r = await callOpenAIWire(route, { ...openAIBody, model: route.model.id });
    if (!r.ok) { const text = await drainBody(r); handleFailure(pid, r.status, text); return { status: r.status }; }
    if (isStream && r.body) return { status: 200, stream: r.body, isRawStream: true };
    const a: unknown = await r.json();
    return { status: 200, openAIBody: isRecord(a) ? a : {} };
  }

  if (wire === 'anthropic') {
    const ab = openAIToAnthropicUpstream(openAIBody, route.model.id);
    if (isStream) ab.stream = true;
    const r = await callAnthropicWire(route, ab);
    if (!r.ok) { const text = await drainBody(r); handleFailure(pid, r.status, text); return { status: r.status }; }
    if (isStream && r.body) return { status: 200, stream: openAIEventsFromAnthropic(r.body, route.model.id), isRawStream: false };
    const a: unknown = await r.json();
    return { status: 200, openAIBody: isRecord(a) ? anthropicUpstreamToOpenAI(a) : {} };
  }

  if (wire === 'gemini') {
    const gb = openAIToGemini(openAIBody, route.model.id);
    const r = await callGeminiWire(route, gb, isStream);
    if (!r.ok) { const text = await drainBody(r); handleFailure(pid, r.status, text); return { status: r.status }; }
    if (isStream && r.body) return { status: 200, stream: openAIEventsFromGemini(r.body, route.model.id), isRawStream: false };
    const a: unknown = await r.json();
    return { status: 200, openAIBody: isRecord(a) ? geminiToOpenAI(a, route.model.id) : {} };
  }

  return { status: 501 };
}

/** Pipe a raw ReadableStream<Uint8Array> to the HTTP response. */
async function pipeRawStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally { reader.releaseLock(); }
}

/** Pipe an AsyncGenerator<string> of SSE text to the HTTP response. */
async function pipeTextStream(gen: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const chunk of gen) res.write(chunk);
}

/** Convert an AsyncGenerator<string> of SSE text into a ReadableStream<Uint8Array> for further translation. */
function generatorToStream(gen: AsyncGenerator<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { value, done } = await gen.next();
      if (done) { controller.close(); return; }
      controller.enqueue(encoder.encode(value));
    },
  });
}

/** POST /v1/chat/completions — OpenAI client wire, with failover across providers. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const isStream = body.stream === true;
  const tried = new Set<string>();

  for (;;) {
    const candidates = rankedRoutes(requires, tried);
    if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model meets this request');
    const route = candidates[0];
    const pid = route.probe.provider.id;
    tried.add(pid);

    try {
      await acquireProvider(pid);
      const ob: Json = { ...body, model: route.model.id };
      if (isStream) { ob.stream = true; ob.stream_options = { include_usage: true }; }

      const result = await callUpstream(route, ob, isStream);
      if (isRetryable(result.status)) { releaseProvider(pid); log(`${pid} → ${result.status}, failover`); continue; }
      if (result.status !== 200) { releaseProvider(pid); return sendError(res, 'openai', result.status, 'upstream_error', `provider ${result.status}`); }

      if (isStream && result.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        try {
          if (result.isRawStream) await pipeRawStream(result.stream as ReadableStream<Uint8Array>, res);
          else await pipeTextStream(result.stream as AsyncGenerator<string>, res);
        } finally { res.end(); releaseProvider(pid); }
        return;
      }

      releaseProvider(pid);
      return result.openAIBody ? sendJson(res, 200, result.openAIBody)
        : sendError(res, 'openai', 502, 'upstream_error', 'empty response');
    } catch (err) {
      releaseProvider(pid);
      log(`${pid} error: ${(err as Error).message}`);
      continue;
    }
  }
}

/** POST /v1/messages — Anthropic client wire, with failover across providers. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const isStream = body.stream === true;
  const clientModel = typeof body.model === 'string' && body.model !== '' ? body.model : undefined;
  const tried = new Set<string>();

  for (;;) {
    const candidates = rankedRoutes(requires, tried);
    if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model meets this request');
    const route = candidates[0];
    const pid = route.probe.provider.id;
    const modelName = clientModel ?? route.model.id;
    tried.add(pid);

    try {
      await acquireProvider(pid);
      const ob = anthropicToOpenAI(body, route.model.id);

      const result = await callUpstream(route, ob, isStream);
      if (isRetryable(result.status)) { releaseProvider(pid); log(`${pid} → ${result.status}, failover`); continue; }
      if (result.status !== 200) { releaseProvider(pid); return sendJson(res, result.status, anthropicError('api_error', `provider ${result.status}`)); }

      if (isStream && result.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        try {
          // All upstream streams are normalized to OpenAI SSE format; translate to Anthropic events
          let openAIStream: ReadableStream<Uint8Array>;
          if (result.isRawStream) {
            openAIStream = result.stream as ReadableStream<Uint8Array>;
          } else {
            // Generator of OpenAI SSE strings → ReadableStream for anthropicEventsFromOpenAI
            openAIStream = generatorToStream(result.stream as AsyncGenerator<string>);
          }
          for await (const event of anthropicEventsFromOpenAI(openAIStream, modelName)) {
            res.write(event);
          }
        } finally { res.end(); releaseProvider(pid); }
        return;
      }

      releaseProvider(pid);
      return result.openAIBody ? sendJson(res, 200, openAIToAnthropic(result.openAIBody, modelName))
        : sendJson(res, 502, anthropicError('api_error', 'empty response'));
    } catch (err) {
      releaseProvider(pid);
      log(`${pid} error: ${(err as Error).message}`);
      continue;
    }
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') { models(req, res); return; }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') { await chat(req, res); return; }
  if (req.method === 'POST' && url.pathname === '/v1/messages') { await messages(req, res); return; }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '0.0.0.0', () => resolve()); });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const r = await fetch(`${ENV_URL}/register`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!r.ok) throw new Error(`POST ${ENV_URL}/register → ${r.status}: ${(await r.text()).slice(0, 500)}`);
  const reply: unknown = await r.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const r = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (r.status !== 200) return;
        const w: unknown = await r.json();
        if (isRecord(w) && w.done === true) { clearInterval(timer); server.close(); process.exit(0); }
      } catch { /* retry */ } finally { inFlight = false; }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'server error');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('no port');
  const port = address.port;
  if (port === 8080) throw new Error('8080 is reserved for env');
  log(`listening 0.0.0.0:${port} — ${catalog.providers.length} providers, ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try { await start(); } catch (err) { log((err as Error).stack ?? String(err)); process.exit(1); }

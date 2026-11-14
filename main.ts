// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone. The brief is at the path in
// $MAKEFUN_PROMPT_FILE and carries three JSON documents: the provider directory (ten simulated
// free-tier providers, each with its id, wire, auth scheme, key environment variable, the key the
// service takes, and the base path to join to $MAKEFUN_ENV_URL), the quality table (every model's
// public rating, what it can do, and whether it is free), and the workload manifest (the twenty
// requests the client will send, in order, with each one's api, whether it streams, its weight and
// the requirements a model must meet to serve it).
//
// The server reads the brief, probes each provider's catalog route in the provider's own wire,
// builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (in the shape of the auth header it got),
//      POST /v1/chat/completions and POST /v1/messages, each with a JSON body or with
//      server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//
// HOW A REQUEST IS SERVED
//   Every request is answered by ONE provider call, always as a plain JSON call (never streamed
//   upstream): the provider answers with the whole ANSWER text, its usage counts and its stop
//   reason, and lib/wire.ts turns that one answer into whatever wire the client used. The client
//   stream is synthesized here, in the client's event order, from that one answer.
//
//   Routing ranks every free model that meets the request's requirements by the quality table's
//   rating_milli, best first. A provider's per-trial budget is drawn inside [calls_min, calls_max]
//   in the kit's env, so the plan treats calls_min as the provider's safe capacity: a request goes
//   to a provider with planned capacity left before one that has used its plan up, which spreads
//   the load over the strong providers instead of burning the best one's budget on the first
//   requests and failing the rest over. A provider that answers "budget is spent" or "is out" is
//   retired for the trial; a provider that "refused this call" only spent one call number, so it
//   stays in the running and its next call may serve.
//
//   Failover: a 402, 429 or 5xx is never forwarded. The request is retried on the next candidate in
//   the ranking until one answers 2xx. A request whose first provider call failed and is still
//   served is what the failover column counts.
//
//   One wire limit: a JSON-mode request on the OpenAI wire needs the provider to echo json_mode,
//   which only the openai wire (response_format) and the gemini wire (responseMimeType) can carry,
//   so an anthropic-wire provider is never a candidate for one.
//
// DETERMINISM
//   No wall clock, no unseeded randomness: every choice comes from the brief, the requests and the
//   providers' answers. The port the OS picks is not part of any scored state. The catalog's
//   base_url is each provider's own path, never joined to the env URL, so a live pass and its
//   replay register the same document.
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
// register body's catalog must be that same document. Build it once, as this file does, and use
// the one object.

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
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswerFrom,
  anthropicError,
  anthropicModelList,
  anthropicStreamFrom,
  anthropicToGemini,
  anthropicToOpenAI,
  errorMessageOf,
  extractAnthropicAnswer,
  extractGeminiAnswer,
  extractOpenAIAnswer,
  openAIAnswerFrom,
  openAIError,
  openAIModelList,
  openAIStreamFrom,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/**
 * The safe planned capacity of each provider, in calls: the kit's env draws a trial's budget for a
 * provider inside [calls_min, calls_max], so planning to calls_min never over-promises — a provider
 * planned to its minimum can still answer every planned call. A provider that runs past its plan
 * (up to its drawn maximum) is still usable; it is only deprioritized, never excluded.
 */
const PLANNED_CAPACITY: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** What this trial has learned about one provider: calls that spent its budget, and whether it is dead. */
interface ProviderState {
  calls: number;
  dead: boolean;
}
const states = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  for (const probe of probes) states.set(probe.provider.id, { calls: 0, dead: false });
}

/** Whether a provider still has planned capacity left in this trial. */
function hasCapacity(providerId: string): boolean {
  const cap = PLANNED_CAPACITY[providerId];
  if (cap === undefined) return true;
  return (states.get(providerId)?.calls ?? 0) < cap;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. A provider that is dead (its
 * budget spent, or its outage reached) is out. Among equal ratings, a provider with planned
 * capacity left comes before one that used its plan up, then provider id, then model id — so the
 * order is fixed no matter how the brief lists them.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (states.get(probe.provider.id)?.dead === true) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const capA = hasCapacity(a.probe.provider.id) ? 1 : 0;
    const capB = hasCapacity(b.probe.provider.id) ? 1 : 0;
    if (capB !== capA) return capB - capA;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
    return 0;
  });
  return out;
}

/** The candidates to try, in order: those with planned capacity left first, then the rest. */
function orderedCandidates(candidates: Candidate[]): Candidate[] {
  return [...candidates.filter((c) => hasCapacity(c.probe.provider.id)), ...candidates.filter((c) => !hasCapacity(c.probe.provider.id))];
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
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

/** The free models across the providers that answered, in probe order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

type UpstreamResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; message: string };

/**
 * One JSON call to one candidate provider, in its own wire. The request is the client's body
 * translated to that wire, naming the candidate model; the answer is the provider's whole text with
 * its usage counts. A refusal that names the budget or an outage retires the provider for the
 * trial; any other refusal only spends one of its calls.
 */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const state = states.get(provider.id);
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  let requestBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    requestBody =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id }
        : anthropicToOpenAI(body, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    requestBody =
      clientApi === 'openai'
        ? openAIToAnthropicRequest(body, candidate.model.id)
        : { ...body, model: candidate.model.id };
  } else {
    const built = clientApi === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
    url = `${base}${built.path}`;
    requestBody = built.body;
  }
  // Upstream is always plain JSON: the client's stream, when it asked for one, is synthesized
  // from this one answer, so no upstream call ever streams.
  requestBody.stream = false;
  delete requestBody.stream_options;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(requestBody), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, message: `the provider could not be reached: ${(err as Error).message}` };
  }
  if (state !== undefined) state.calls += 1;
  const text = await res.text();
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    // Not JSON: an error body the shape cannot name.
  }
  if (!res.ok) {
    const message = errorMessageOf(parsed);
    if (state !== undefined && (message.includes('budget is spent') || message.includes(' is out'))) state.dead = true;
    return { ok: false, status: res.status, message: message !== '' ? message : `the provider answered HTTP ${res.status}` };
  }
  const answer =
    provider.wire === 'openai'
      ? extractOpenAIAnswer(isRecord(parsed) ? parsed : {})
      : provider.wire === 'anthropic'
        ? extractAnthropicAnswer(isRecord(parsed) ? parsed : {})
        : extractGeminiAnswer(isRecord(parsed) ? parsed : {});
  if (answer === null) return { ok: false, status: 502, message: 'the provider answered in a shape this server cannot read' };
  return { ok: true, answer };
}

interface Served {
  status: number;
  contentType: string;
  text: string;
}

/** One client request, answered by the best free model that can serve it, with failover. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Served> {
  const requires = requirementsFor(body);
  const jsonMode =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = orderedCandidates(candidatesFor(requires, clientApi, jsonMode));
  let lastStatus = 503;
  let lastError = 'no free model on the directory meets this request';
  for (const candidate of candidates) {
    const result = await callUpstream(candidate, clientApi, body);
    if (result.ok) {
      const answer = result.answer;
      if (clientApi === 'openai') {
        if (stream) return { status: 200, contentType: 'text/event-stream', text: openAIStreamFrom(answer, candidate.model.id) };
        return { status: 200, contentType: 'application/json', text: JSON.stringify(openAIAnswerFrom(answer, candidate.model.id)) };
      }
      if (stream) return { status: 200, contentType: 'text/event-stream', text: anthropicStreamFrom(answer, candidate.model.id) };
      return { status: 200, contentType: 'application/json', text: JSON.stringify(anthropicAnswerFrom(answer, candidate.model.id)) };
    }
    lastStatus = result.status;
    lastError = result.message;
    // A 400 or 401 is this server's own fault (a bad request or a bad key), not a failover case.
    if (result.status === 400 || result.status === 401) break;
    // A 402, 429 or 5xx, or an unreachable provider, is a failover case: try the next candidate.
  }
  const status = lastStatus === 503 ? 503 : 502;
  return {
    status,
    contentType: 'application/json',
    text: JSON.stringify(clientApi === 'anthropic' ? anthropicError('api_error', lastError) : openAIError(lastError, 'api_error', null)),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
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
  } catch {
    return null;
  }
}

function sendServed(res: ServerResponse, served: Served): void {
  const headers: Record<string, string> = { 'content-type': served.contentType, 'content-length': String(Buffer.byteLength(served.text)) };
  if (served.contentType === 'text/event-stream') {
    headers['cache-control'] = 'no-cache';
    headers.connection = 'keep-alive';
  }
  res.writeHead(served.status, headers);
  res.end(served.text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') {
    const text = JSON.stringify(anthropicError(type, message));
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  } else {
    const text = JSON.stringify(openAIError(message, type, null));
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    const text = JSON.stringify(anthropicModelList(list));
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  } else {
    const text = JSON.stringify(openAIModelList(list));
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  sendServed(res, await serve('openai', body, body.stream === true));
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  sendServed(res, await serve('anthropic', body, body.stream === true));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

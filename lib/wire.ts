// Client wires and ALL THREE upstream wires: openai, anthropic, gemini.
//
// Client-facing:
//   - POST /v1/chat/completions (OpenAI client)
//   - POST /v1/messages (Anthropic client)
//   - GET /v1/models (both)
//
// Upstream:
//   - openai-wire: POST <base>/v1/chat/completions (passthrough for OpenAI clients, translate for Anthropic)
//   - anthropic-wire: POST <base>/v1/messages with x-api-key + anthropic-version: 2023-06-01
//   - gemini-wire: POST <base>/v1beta/models/<model>:generateContent or :streamGenerateContent?alt=sse

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch { /* not JSON */ }
  }
  return {};
}

export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

// ─── Anthropic client → OpenAI upstream ───

export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

// ─── OpenAI upstream → Anthropic client ───

export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ─── OpenAI upstream SSE → Anthropic client events ───

export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream', type: 'message', role: 'assistant', model,
          content: [], stop_reason: null, stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b))
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try { chunk = JSON.parse(payload); } catch { continue; }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex++;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex++;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(sseEvent('content_block_start', {
            type: 'content_block_start', index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }));
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ═══════════════════════════════════════════════════════════════════
// ANTHROPIC UPSTREAM WIRE
// ═══════════════════════════════════════════════════════════════════

/** OpenAI chat completions body → Anthropic Messages body for upstream anthropic-wire provider. */
export function openAIToAnthropicUpstream(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system: string | undefined;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      system = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      continue;
    }
    if (raw.role === 'tool') {
      // tool result → Anthropic tool_result block
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
          content: typeof raw.content === 'string' ? raw.content : '',
        }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      if (typeof raw.content === 'string' && raw.content !== '') content.push({ type: 'text', text: raw.content });
      if (Array.isArray(raw.tool_calls)) {
        for (const tc of raw.tool_calls) {
          if (!isRecord(tc) || !isRecord(tc.function)) continue;
          content.push({
            type: 'tool_use',
            id: typeof tc.id === 'string' ? tc.id : '',
            name: typeof tc.function.name === 'string' ? tc.function.name : '',
            input: argumentsOf(tc.function.arguments),
          });
        }
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      else if (typeof raw.content === 'string') messages.push({ role: 'assistant', content: raw.content });
      continue;
    }
    // user
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
    } else if (Array.isArray(raw.content)) {
      const blocks: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          blocks.push({ type: 'text', text: part.text });
        } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url as string;
          if (url.startsWith('data:')) {
            const match = url.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              blocks.push({ type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } });
            }
          } else {
            blocks.push({ type: 'image', source: { type: 'url', url } });
          }
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 4096 };
  if (system) out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (body.stream === true) out.stream = true;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function)) return [];
      const fn = raw.function;
      return [{
        name: typeof fn.name === 'string' ? fn.name : '',
        description: typeof fn.description === 'string' ? fn.description : '',
        input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (typeof body.tool_choice === 'string') {
    if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
    else if (body.tool_choice === 'none') out.tool_choice = { type: 'auto' }; // closest
  } else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) {
    out.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
  }
  return out;
}

/** Anthropic Messages response → OpenAI Chat Completions response. */
export function anthropicUpstreamToOpenAI(body: Json): Json {
  const content: Json[] = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const stopReason = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  let finishReason = 'stop';
  if (stopReason === 'max_tokens') finishReason = 'length';
  else if (stopReason === 'tool_use') finishReason = 'tool_calls';
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text || null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-anthropic',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: typeof body.model === 'string' ? body.model : '',
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** Parse Anthropic SSE stream → yield OpenAI-format SSE events (so existing pipe/translate code works). */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let chunkId = 'chatcmpl-anthropic-stream';

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const ev of parseAnthropicFrame(frame, model, chunkId)) yield ev;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const ev of parseAnthropicFrame(buffer, model, chunkId)) yield ev;
    }
  } finally {
    reader.releaseLock();
  }

  function parseAnthropicFrame(frame: string, model: string, id: string): string[] {
    const out: string[] = [];
    let eventType = '';
    let dataStr = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event: ')) eventType = line.slice(7).trim();
      else if (line.startsWith('data: ')) dataStr = line.slice(6);
    }
    if (!dataStr) return out;
    let data: unknown;
    try { data = JSON.parse(dataStr); } catch { return out; }
    if (!isRecord(data)) return out;

    if (eventType === 'message_start' && isRecord(data.message)) {
      const msg = data.message;
      if (isRecord(msg.usage)) inputTokens = numberOrZero(msg.usage.input_tokens);
      // emit first chunk with role
      out.push(`data: ${JSON.stringify({
        id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
        choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
      })}\n\n`);
    } else if (eventType === 'content_block_delta' && isRecord(data.delta)) {
      const delta = data.delta;
      if (delta.type === 'text_delta' && typeof delta.text === 'string') {
        out.push(`data: ${JSON.stringify({
          id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
          choices: [{ index: 0, delta: { content: delta.text }, finish_reason: null }],
        })}\n\n`);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        out.push(`data: ${JSON.stringify({
          id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
          choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: delta.partial_json } }] }, finish_reason: null }],
        })}\n\n`);
      }
    } else if (eventType === 'content_block_start' && isRecord(data.content_block)) {
      const cb = data.content_block;
      if (cb.type === 'tool_use') {
        out.push(`data: ${JSON.stringify({
          id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
          choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: cb.id, type: 'function', function: { name: cb.name, arguments: '' } }] }, finish_reason: null }],
        })}\n\n`);
      }
    } else if (eventType === 'message_delta' && isRecord(data.delta)) {
      const stopReason = typeof data.delta.stop_reason === 'string' ? data.delta.stop_reason : 'end_turn';
      let finishReason = 'stop';
      if (stopReason === 'max_tokens') finishReason = 'length';
      else if (stopReason === 'tool_use') finishReason = 'tool_calls';
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      out.push(`data: ${JSON.stringify({
        id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
        choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      })}\n\n`);
    } else if (eventType === 'message_stop') {
      out.push('data: [DONE]\n\n');
    }
    return out;
  }
}

// ═══════════════════════════════════════════════════════════════════
// GEMINI UPSTREAM WIRE
// ═══════════════════════════════════════════════════════════════════

/** OpenAI chat completions body → Gemini generateContent body. */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemInstruction: Json | undefined;
  const source = Array.isArray(body.messages) ? body.messages : [];

  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      systemInstruction = { parts: [{ text }] };
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          parts.push({ text: part.text });
        } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url as string;
          const match = url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
          }
        }
      }
    }
    // Handle tool calls from assistant
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const tc of raw.tool_calls) {
        if (!isRecord(tc) || !isRecord(tc.function)) continue;
        parts.push({
          functionCall: {
            name: typeof tc.function.name === 'string' ? tc.function.name : '',
            args: argumentsOf(tc.function.arguments),
          },
        });
      }
    }
    // Handle tool results
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : '';
      let response: Json;
      try { response = JSON.parse(text); if (!isRecord(response)) response = { result: text }; } catch { response = { result: text }; }
      contents.push({
        role: 'user', // ponytail: Gemini uses 'user' role for function responses but with functionResponse part
        parts: [{ functionResponse: { name: typeof raw.name === 'string' ? raw.name : 'function', response } }],
      });
      continue;
    }
    if (parts.length > 0) contents.push({ role, parts });
  }

  const out: Json = { contents };
  if (systemInstruction) out.systemInstruction = systemInstruction;

  // Generation config
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  // JSON mode
  if (isRecord(body.response_format)) {
    if (body.response_format.type === 'json_object') config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;

  // Tools
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function)) return [];
      const fn = raw.function;
      return [{
        name: typeof fn.name === 'string' ? fn.name : '',
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }

  return out;
}

/** Gemini generateContent response → OpenAI Chat Completions response. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];

  let text = '';
  const toolCalls: Json[] = [];
  let toolIdx = 0;
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolIdx}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
      toolIdx++;
    }
  }

  const finishMap: Record<string, string> = { STOP: 'stop', MAX_TOKENS: 'length', SAFETY: 'content_filter' };
  const finishReason = typeof candidate.finishReason === 'string' ? (finishMap[candidate.finishReason] ?? 'stop') : 'stop';
  if (toolCalls.length > 0 && finishReason === 'stop') { /* keep as tool_calls */ }

  const message: Json = { role: 'assistant', content: text || null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;

  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: toolCalls.length > 0 ? 'tool_calls' : finishReason }],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** Gemini streamGenerateContent SSE → OpenAI-format SSE events. */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const id = 'chatcmpl-gemini-stream';

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const ev of parseGeminiFrame(frame)) yield ev;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const ev of parseGeminiFrame(buffer)) yield ev;
    }
  } finally {
    reader.releaseLock();
  }
  yield 'data: [DONE]\n\n';

  function parseGeminiFrame(frame: string): string[] {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      let data: unknown;
      try { data = JSON.parse(payload); } catch { continue; }
      if (!isRecord(data)) continue;

      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (!candidate) continue;
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];

      for (const part of parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string') {
          out.push(`data: ${JSON.stringify({
            id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
            choices: [{ index: 0, delta: { content: part.text }, finish_reason: null }],
          })}\n\n`);
        }
        if (isRecord(part.functionCall)) {
          out.push(`data: ${JSON.stringify({
            id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
            choices: [{ index: 0, delta: { tool_calls: [{
              index: 0, id: `call_0`, type: 'function',
              function: { name: part.functionCall.name, arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
            }] }, finish_reason: null }],
          })}\n\n`);
        }
      }

      // finish reason on last chunk
      if (typeof candidate.finishReason === 'string') {
        const finishMap: Record<string, string> = { STOP: 'stop', MAX_TOKENS: 'length', SAFETY: 'content_filter' };
        const fr = finishMap[candidate.finishReason] ?? 'stop';
        const usage = isRecord(data.usageMetadata) ? data.usageMetadata : {};
        out.push(`data: ${JSON.stringify({
          id, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
          choices: [{ index: 0, delta: {}, finish_reason: fr }],
          usage: {
            prompt_tokens: numberOrZero(usage.promptTokenCount),
            completion_tokens: numberOrZero(usage.candidatesTokenCount),
            total_tokens: numberOrZero(usage.totalTokenCount),
          },
        })}\n\n`);
      }
    }
    return out;
  }
}

// ponytail: anthropicEventsPassthrough removed — all streams normalize to OpenAI SSE then
// translate to client wire via anthropicEventsFromOpenAI. Direct passthrough would save a
// translate round-trip; add when profiling shows it matters.

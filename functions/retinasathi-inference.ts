import { createClient } from 'npm:@insforge/sdk';

const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 60_000;
const ALLOWED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const ALLOWED_ORIGINS = new Set([
  'https://69exmaqk.insforge.site',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
]);

function corsHeaders(request: Request): Record<string, string> {
  const origin = request.headers.get('origin');
  return {
    'Access-Control-Allow-Origin': origin && ALLOWED_ORIGINS.has(origin)
      ? origin
      : 'https://69exmaqk.insforge.site',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Authorization, Content-Type',
    'Vary': 'Origin',
  };
}

function json(request: Request, status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(request), 'Content-Type': 'application/json' },
  });
}

export default async function (request: Request): Promise<Response> {
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders(request) });
  }
  if (request.method !== 'POST') {
    return json(request, 405, { error: 'METHOD_NOT_ALLOWED', message: 'Use POST for screening.' });
  }

  const authHeader = request.headers.get('authorization');
  const accessToken = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : '';
  if (!accessToken) {
    return json(request, 401, { error: 'UNAUTHORIZED', message: 'Sign in before running a screening.' });
  }

  const baseUrl = Deno.env.get('INSFORGE_BASE_URL') ?? '';
  const azureUrl = (Deno.env.get('AZURE_INFERENCE_URL') ?? '').replace(/\/$/, '');
  const inferenceKey = Deno.env.get('AZURE_INFERENCE_API_KEY') ?? '';
  if (!baseUrl || !azureUrl || !inferenceKey) {
    return json(request, 503, {
      error: 'MODEL_UNAVAILABLE',
      message: 'Cloud inference is not configured.',
      next_action: 'Contact the technical operator.',
    });
  }

  const insforge = createClient({ baseUrl, accessToken });
  const { data: userData, error: userError } = await insforge.auth.getCurrentUser();
  if (userError || !userData?.user?.id) {
    return json(request, 401, { error: 'UNAUTHORIZED', message: 'Your session is invalid or expired.' });
  }

  let incoming: FormData;
  try {
    incoming = await request.formData();
  } catch {
    return json(request, 400, { error: 'INVALID_FILE', message: 'Upload a retinal image file.' });
  }
  const file = incoming.get('file');
  if (!(file instanceof File) || !ALLOWED_IMAGE_TYPES.has(file.type)) {
    return json(request, 415, { error: 'INVALID_FILE', message: 'Upload a JPEG, PNG, or WebP retinal image.' });
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return json(request, 413, { error: 'IMAGE_TOO_LARGE', message: 'Image exceeds the 15 MB limit.' });
  }

  const outgoing = new FormData();
  outgoing.append('file', file, file.name);
  try {
    const upstream = await fetch(`${azureUrl}/predict`, {
      method: 'POST',
      headers: { 'X-Inference-Key': inferenceKey },
      body: outgoing,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const payload = await upstream.text();
    if (!upstream.ok) {
      let detail: { code?: string; message?: string; next_action?: string } | string | undefined;
      try {
        detail = (JSON.parse(payload) as { detail?: typeof detail }).detail;
      } catch {
        detail = undefined;
      }
      const structured = typeof detail === 'object' && detail !== null ? detail : undefined;
      return json(request, upstream.status, {
        error: structured?.code ?? 'INFERENCE_FAILED',
        message: structured?.message ?? (typeof detail === 'string' ? detail : 'Azure inference could not analyze this image.'),
        next_action: structured?.next_action ?? 'Retry or contact the technical operator.',
      });
    }
    return new Response(payload, {
      status: 200,
      headers: { ...corsHeaders(request), 'Content-Type': upstream.headers.get('content-type') ?? 'application/json' },
    });
  } catch {
    return json(request, 503, {
      error: 'MODEL_UNAVAILABLE',
      message: 'Azure inference could not be reached.',
      next_action: 'Retry after the cloud model finishes starting.',
    });
  }
}

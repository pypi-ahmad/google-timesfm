// Fetch wrapper for the FastAPI backend (mounted at /api/v1) plus small
// shared helpers for building query strings and normalizing error/unknown
// values into displayable text. See lib/types.ts for the response shapes
// this returns and lib/spec.ts for the request payload shape.
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
  }
}

export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData))
    headers.set("Content-Type", "application/json");
  const response = await fetch(`/api/v1${path}`, { ...options, headers });
  if (!response.ok) {
    // `detail` here is untrusted server JSON: either a plain string, or a
    // list of FastAPI/pydantic validation errors (each with a field path in
    // `loc`). `loc[0]` is the request-part marker (e.g. "body"), so it's
    // dropped before joining the remaining path into a readable field name.
    const body = await response.json().catch(() => null);
    const detail = body?.detail ?? body?.message;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map(
                (item: { msg?: string; loc?: string[] }) =>
                  `${item.loc?.slice(1).join(".") ?? "Input"}: ${item.msg}`,
              )
              .join("; ")
          : `Request failed (${response.status}). Check that the API is running and try again.`;
    throw new ApiError(response.status, message, detail);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}
export function json(value: unknown) {
  return JSON.stringify(value);
}
export function errorMessage(value: unknown): string {
  if (!value) return "";
  if (value instanceof Error) return value.message;
  if (typeof value === "string") return value;
  if (
    typeof value === "object" &&
    "message" in value &&
    typeof value.message === "string"
  )
    return value.message;
  return JSON.stringify(value);
}
export function query(
  values: Record<string, string | number | boolean | null | undefined>,
) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values))
    if (value !== undefined && value !== null && value !== "")
      params.set(key, String(value));
  return params.toString();
}

/**
 * Wire types for the popup <-> FastAPI backend contract (see server/app.py).
 */

/** Body of `POST /ask/stream`. Mirrors the Pydantic model `QueryRequest`. */
export interface AskRequest {
  question: string;
}

/**
 * Outcome of a `POST /ask/stream` call.
 *
 * On success the backend replies `200 text/plain; charset=utf-8` and streams
 * the answer as raw UTF-8 text (not JSON, not SSE), so the successful payload
 * is just the accumulated `answer` string. Any non-2xx reply (e.g. FastAPI's
 * `422 {"detail": [...]}` for a malformed body) is surfaced with its raw body
 * text, which the popup shows verbatim.
 */
export type AskOutcome =
  | { readonly kind: "answer"; readonly answer: string }
  | { readonly kind: "empty" }
  | { readonly kind: "http-error"; readonly status: number; readonly body: string };

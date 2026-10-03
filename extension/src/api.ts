import { ASK_STREAM_URL } from "./config";
import type { AskOutcome, AskRequest } from "./types";

export type FetchFn = (input: string, init: RequestInit) => Promise<Response>;

/**
 * Reads a byte stream as UTF-8 text, invoking `onText` with the full
 * accumulated text after every chunk. Returns the final text.
 */
export async function readTextStream(
  stream: ReadableStream<Uint8Array>,
  onText: (textSoFar: string) => void,
): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let text = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    text += decoder.decode(value, { stream: true });
    onText(text);
  }

  return text;
}

/**
 * Sends `question` to `POST /ask/stream` and streams the answer back through
 * `onProgress`. Network failures and stream errors are thrown to the caller;
 * HTTP errors and empty answers are returned as typed outcomes.
 */
export async function askStream(
  question: string,
  onProgress: (answerSoFar: string) => void,
  fetchFn: FetchFn = (input, init) => fetch(input, init),
): Promise<AskOutcome> {
  const body: AskRequest = { question };
  const response = await fetchFn(ASK_STREAM_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    return { kind: "http-error", status: response.status, body: await response.text() };
  }

  if (response.body === null) {
    return { kind: "empty" };
  }

  const answer = await readTextStream(response.body, onProgress);
  return answer.trim() === "" ? { kind: "empty" } : { kind: "answer", answer };
}

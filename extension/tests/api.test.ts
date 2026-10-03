import { describe, expect, it, vi } from "vitest";
import { askStream, readTextStream, type FetchFn } from "../src/api";
import { ASK_STREAM_URL } from "../src/config";

const encoder = new TextEncoder();

function byteStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

function textResponse(chunks: string[], init?: ResponseInit): Response {
  return new Response(byteStream(chunks.map((c) => encoder.encode(c))), init);
}

describe("readTextStream", () => {
  it("reports the accumulated text after every chunk", async () => {
    const seen: string[] = [];
    const text = await readTextStream(
      byteStream(["## Steps", "\n1. Boil"].map((c) => encoder.encode(c))),
      (t) => seen.push(t),
    );
    expect(text).toBe("## Steps\n1. Boil");
    expect(seen).toEqual(["## Steps", "## Steps\n1. Boil"]);
  });

  it("decodes multi-byte UTF-8 characters split across chunks", async () => {
    const bytes = encoder.encode("🍽️");
    const text = await readTextStream(
      byteStream([bytes.slice(0, 2), bytes.slice(2)]),
      () => undefined,
    );
    expect(text).toBe("🍽️");
  });
});

describe("askStream", () => {
  it("POSTs an AskRequest JSON body to /ask/stream", async () => {
    const fetchFn = vi.fn<FetchFn>(async () => textResponse(["hi"]));
    await askStream("fish pie?", () => undefined, fetchFn);

    expect(fetchFn).toHaveBeenCalledOnce();
    const [url, init] = fetchFn.mock.calls[0] ?? [];
    expect(url).toBe(ASK_STREAM_URL);
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
    expect(init?.body).toBe(JSON.stringify({ question: "fish pie?" }));
  });

  it("streams progress and returns the full answer", async () => {
    const progress: string[] = [];
    const outcome = await askStream("q", (t) => progress.push(t), async () =>
      textResponse(["Here are ", "some recipes"]),
    );
    expect(progress).toEqual(["Here are ", "Here are some recipes"]);
    expect(outcome).toEqual({ kind: "answer", answer: "Here are some recipes" });
  });

  it("returns the raw body for non-2xx responses", async () => {
    const body = '{"detail":[{"type":"missing","loc":["body","question"],"msg":"Field required"}]}';
    const outcome = await askStream("q", () => undefined, async () =>
      new Response(body, { status: 422 }),
    );
    expect(outcome).toEqual({ kind: "http-error", status: 422, body });
  });

  it("treats a blank stream as an empty answer", async () => {
    const outcome = await askStream("q", () => undefined, async () => textResponse([" ", "\n"]));
    expect(outcome).toEqual({ kind: "empty" });
  });

  it("treats a missing body as an empty answer", async () => {
    const outcome = await askStream("q", () => undefined, async () => new Response(null));
    expect(outcome).toEqual({ kind: "empty" });
  });

  it("propagates network failures", async () => {
    await expect(
      askStream("q", () => undefined, async () => {
        throw new TypeError("Failed to fetch");
      }),
    ).rejects.toThrow("Failed to fetch");
  });
});

import { describe, expect, it } from "vitest";
import { formatBackendError, formatUnexpectedError } from "../src/messages";

describe("formatBackendError", () => {
  it("prefixes the raw backend body", () => {
    expect(formatBackendError('{"detail":"x"}')).toBe('Backend error:\n{"detail":"x"}');
  });
});

describe("formatUnexpectedError", () => {
  it("uses the message of an Error", () => {
    expect(formatUnexpectedError(new TypeError("Failed to fetch"))).toBe("Error: Failed to fetch");
  });

  it("stringifies non-Error values", () => {
    expect(formatUnexpectedError("boom")).toBe("Error: boom");
  });
});

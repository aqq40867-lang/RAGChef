import { describe, expect, it } from "vitest";
import { MESSAGES } from "../src/messages";
import { validateQuestion } from "../src/validation";

describe("validateQuestion", () => {
  it("trims and accepts a non-empty question", () => {
    expect(validateQuestion("  How do I make fish pie?\n")).toEqual({
      ok: true,
      question: "How do I make fish pie?",
    });
  });

  it.each(["", "   ", "\n\t "])("rejects blank input %j", (raw) => {
    expect(validateQuestion(raw)).toEqual({ ok: false, message: MESSAGES.emptyQuestion });
  });
});

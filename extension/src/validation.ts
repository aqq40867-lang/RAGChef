import { MESSAGES } from "./messages";

export type QuestionValidation =
  | { readonly ok: true; readonly question: string }
  | { readonly ok: false; readonly message: string };

/** Trims the raw textarea value and rejects blank input. */
export function validateQuestion(raw: string): QuestionValidation {
  const question = raw.trim();
  if (question === "") {
    return { ok: false, message: MESSAGES.emptyQuestion };
  }
  return { ok: true, question };
}

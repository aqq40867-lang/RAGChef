/** User-facing strings shown in the popup's answer area. */
export const MESSAGES = {
  emptyQuestion: "Please enter a recipe question.",
  loading: "Thinking...",
  emptyAnswer: "The backend returned an empty answer. Please try again.",
} as const;

export function formatBackendError(body: string): string {
  return `Backend error:\n${body}`;
}

/** Formats anything thrown (network failure, aborted stream, ...) for display. */
export function formatUnexpectedError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return `Error: ${message}`;
}

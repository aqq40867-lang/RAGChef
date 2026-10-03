import { askStream } from "./api";
import { getElement } from "./dom";
import { renderMarkdown } from "./markdown";
import { MESSAGES, formatBackendError, formatUnexpectedError } from "./messages";
import { validateQuestion } from "./validation";

const questionInput = getElement("question", HTMLTextAreaElement);
const askButton = getElement("askBtn", HTMLButtonElement);
const answerDiv = getElement("answer", HTMLDivElement);

function showText(text: string): void {
  answerDiv.innerText = text;
}

function showMarkdown(markdown: string): void {
  // Safe: renderMarkdown escapes all HTML before adding its own tags.
  answerDiv.innerHTML = renderMarkdown(markdown);
}

async function handleAsk(): Promise<void> {
  const validation = validateQuestion(questionInput.value);
  if (!validation.ok) {
    showText(validation.message);
    return;
  }

  showText(MESSAGES.loading);

  try {
    // Re-rendering the whole answer-so-far on every chunk is cheap at recipe
    // length and avoids needing an incremental Markdown parser.
    const outcome = await askStream(validation.question, showMarkdown);

    switch (outcome.kind) {
      case "answer":
        // Already rendered progressively via showMarkdown.
        break;
      case "empty":
        showText(MESSAGES.emptyAnswer);
        break;
      case "http-error":
        showText(formatBackendError(outcome.body));
        break;
    }
  } catch (error: unknown) {
    showText(formatUnexpectedError(error));
  }
}

askButton.addEventListener("click", () => {
  void handleAsk();
});

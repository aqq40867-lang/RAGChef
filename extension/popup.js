// Toggle this to test against your local backend (docker compose up / uvicorn)
// vs. the deployed Render backend. Switch back to the Render URL before
// packaging the extension for real use.
const BACKEND_BASE = "http://localhost:8000";
// const BACKEND_BASE = "https://recipe-rag-extension.onrender.com";

// /ask/stream streams the answer as plain text chunks as DeepSeek generates
// them, instead of /ask's "wait for the whole answer, then respond" -- the
// popup renders each chunk as it arrives so the answer appears
// incrementally rather than sitting on "Thinking..." for the full
// generation time.
const BACKEND_URL = BACKEND_BASE + "/ask/stream";

// Minimal, dependency-free Markdown renderer. Escapes HTML first on every
// line (so the LLM's response can never inject markup), *then* looks for
// Markdown syntax and turns it into real HTML elements: "## " headings,
// "- "/"* " bullet lists, "1. " numbered lists, plus inline **bold** and
// *italic*. Everything else is rendered as a plain paragraph.
//
// This exists because rag.py's "detail" answer mode asks the LLM for
// structured Markdown (## Overview / ## Ingredients / ## Steps / ## Tips,
// with bullet and numbered lists) -- the previous version of this function
// only understood bold/italic, so those headings and lists showed up as
// raw "##"/"-"/"1." characters instead of being rendered.
function renderMarkdown(text) {
  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Applies inline formatting (bold/italic) to a single line of already-safe
  // text. Escaping happens first, so **/*  characters introduced by the
  // regex replacements below are the only markup ever added.
  const inline = (line) =>
    escapeHtml(line)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");

  const html = [];
  let openList = null; // "ul" | "ol" | null -- tracks a list block in progress

  const closeList = () => {
    if (openList) {
      html.push(`</${openList}>`);
      openList = null;
    }
  };

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();

    if (line === "") {
      // Blank lines only exist to separate blocks; spacing between blocks
      // comes from CSS margins, not from preserved blank lines.
      closeList();
      continue;
    }

    const heading = line.match(/^#{1,3}\s+(.*)$/);
    if (heading) {
      closeList();
      html.push(`<h4 class="md-heading">${inline(heading[1])}</h4>`);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      if (openList !== "ul") {
        closeList();
        html.push("<ul>");
        openList = "ul";
      }
      html.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }

    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (numbered) {
      if (openList !== "ol") {
        closeList();
        html.push("<ol>");
        openList = "ol";
      }
      html.push(`<li>${inline(numbered[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p class="md-p">${inline(line)}</p>`);
  }
  closeList();

  return html.join("");
}

document.getElementById("askBtn").addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  const answerDiv = document.getElementById("answer");

  if (!question) {
    answerDiv.innerText = "Please enter a recipe question.";
    return;
  }

  answerDiv.innerText = "Thinking...";

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question: question })
    });

    if (!response.ok) {
      const text = await response.text();
      answerDiv.innerText = "Backend error:\n" + text;
      return;
    }

    // Read the streamed response incrementally and re-render the
    // accumulated Markdown after each chunk, so the answer visibly grows
    // instead of appearing all at once after the full generation finishes.
    // Re-parsing the whole answer-so-far on every chunk is wasteful in
    // principle, but at recipe-answer length (a few hundred words) it's
    // cheap enough not to matter -- not worth the complexity of an
    // incremental Markdown parser for this.
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let answer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      answer += decoder.decode(value, { stream: true });
      answerDiv.innerHTML = renderMarkdown(answer);
    }

  } catch (error) {
    answerDiv.innerText = "Error: " + error.message;
  }
});
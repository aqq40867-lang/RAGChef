/**
 * Minimal, dependency-free Markdown renderer.
 *
 * Escapes HTML first on every line (so the LLM's response can never inject
 * markup), *then* looks for Markdown syntax and turns it into real HTML
 * elements: "#"/"##"/"###" headings, "- "/"* " bullet lists, "1. "/"1) "
 * numbered lists, plus inline **bold** and *italic*. Everything else is
 * rendered as a plain paragraph.
 *
 * This exists because rag.py's "detail" answer mode asks the LLM for
 * structured Markdown (## Overview / ## Ingredients / ## Steps / ## Tips,
 * with bullet and numbered lists).
 */

type ListTag = "ul" | "ol";

const HEADING = /^#{1,3}\s+(.*)$/;
const BULLET = /^[-*]\s+(.*)$/;
const NUMBERED = /^\d+[.)]\s+(.*)$/;

export function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Applies inline formatting (bold/italic) to a single line. Escaping happens
 * first, so the tags added by these replacements are the only markup emitted.
 */
function renderInline(line: string): string {
  return escapeHtml(line)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

export function renderMarkdown(text: string): string {
  const html: string[] = [];
  let openList: ListTag | null = null;

  const closeList = (): void => {
    if (openList !== null) {
      html.push(`</${openList}>`);
      openList = null;
    }
  };

  const pushListItem = (tag: ListTag, content: string): void => {
    if (openList !== tag) {
      closeList();
      html.push(`<${tag}>`);
      openList = tag;
    }
    html.push(`<li>${renderInline(content)}</li>`);
  };

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();

    if (line === "") {
      // Blank lines only separate blocks; spacing comes from CSS margins.
      closeList();
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      closeList();
      html.push(`<h4 class="md-heading">${renderInline(heading[1] ?? "")}</h4>`);
      continue;
    }

    const bullet = BULLET.exec(line);
    if (bullet) {
      pushListItem("ul", bullet[1] ?? "");
      continue;
    }

    const numbered = NUMBERED.exec(line);
    if (numbered) {
      pushListItem("ol", numbered[1] ?? "");
      continue;
    }

    closeList();
    html.push(`<p class="md-p">${renderInline(line)}</p>`);
  }
  closeList();

  return html.join("");
}

import { describe, expect, it } from "vitest";
import { escapeHtml, renderMarkdown } from "../src/markdown";

describe("escapeHtml", () => {
  it("escapes &, < and >", () => {
    expect(escapeHtml("a & <b>")).toBe("a &amp; &lt;b&gt;");
  });
});

describe("renderMarkdown", () => {
  it("renders plain lines as paragraphs", () => {
    expect(renderMarkdown("Hello\nWorld")).toBe(
      '<p class="md-p">Hello</p><p class="md-p">World</p>',
    );
  });

  it("renders #, ## and ### headings as h4", () => {
    expect(renderMarkdown("# A\n## B\n### C")).toBe(
      '<h4 class="md-heading">A</h4><h4 class="md-heading">B</h4><h4 class="md-heading">C</h4>',
    );
  });

  it("does not treat #### or a bare # as a heading", () => {
    expect(renderMarkdown("#### Deep")).toBe('<p class="md-p">#### Deep</p>');
    expect(renderMarkdown("#tag")).toBe('<p class="md-p">#tag</p>');
  });

  it("groups consecutive bullets into one list and closes it on a blank line", () => {
    expect(renderMarkdown("- one\n* two\n\nafter")).toBe(
      '<ul><li>one</li><li>two</li></ul><p class="md-p">after</p>',
    );
  });

  it("renders numbered lists with . or ) and switches list type cleanly", () => {
    expect(renderMarkdown("1. a\n2) b\n- c")).toBe(
      "<ol><li>a</li><li>b</li></ol><ul><li>c</li></ul>",
    );
  });

  it("applies bold and italic inline formatting", () => {
    expect(renderMarkdown("**Salt** and *pepper*")).toBe(
      '<p class="md-p"><strong>Salt</strong> and <em>pepper</em></p>',
    );
  });

  it("escapes HTML from the model before adding markup", () => {
    const html = renderMarkdown('## <img src=x onerror="alert(1)">\n- <script>x</script>');
    expect(html).not.toContain("<img");
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
  });

  it("returns an empty string for empty or whitespace-only input", () => {
    expect(renderMarkdown("")).toBe("");
    expect(renderMarkdown("  \n \n")).toBe("");
  });
});

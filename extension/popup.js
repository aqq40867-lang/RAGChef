// Toggle this to test against your local backend (docker compose up / uvicorn)
// vs. the deployed Render backend. Switch back to the Render URL before
// packaging the extension for real use.
const BACKEND_URL = "http://localhost:8000/ask";
// const BACKEND_URL = "https://recipe-rag-extension.onrender.com/ask";

// Minimal, dependency-free Markdown renderer: escapes HTML first (so the
// LLM's response can never inject markup), then converts **bold** and
// *italic*. Line breaks are handled by the #answer CSS (white-space: pre-wrap).
function renderMarkdown(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
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

    const text = await response.text();

    if (!response.ok) {
      answerDiv.innerText = "Backend error:\n" + text;
      return;
    }

    const data = JSON.parse(text);
    answerDiv.innerHTML = renderMarkdown(data.answer);

  } catch (error) {
    answerDiv.innerText = "Error: " + error.message;
  }
});
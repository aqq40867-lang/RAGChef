document.getElementById("askBtn").addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  const answerDiv = document.getElementById("answer");

  if (!question) {
    answerDiv.innerText = "Please enter a recipe question.";
    return;
  }

  answerDiv.innerText = "Thinking...";

  try {
    const response = await fetch("https://recipe-rag-extension.onrender.com/ask", {
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
    answerDiv.innerText = data.answer;

  } catch (error) {
    answerDiv.innerText = "Error: " + error.message;
  }
});
/**
 * Backend location. This is the only place the URL lives.
 *
 * Toggle this to test against your local backend (docker compose up / uvicorn)
 * vs. the deployed Render backend. Switch back to the Render URL before
 * packaging the extension for real use. Whichever origin you use must also be
 * listed under `host_permissions` in manifest.json.
 */
export const BACKEND_BASE_URL = "http://localhost:8000";
// export const BACKEND_BASE_URL = "https://recipe-rag-extension.onrender.com";

/**
 * `POST /ask/stream` streams the answer as plain-text chunks as DeepSeek
 * generates them, instead of `/ask`'s "wait for the whole answer, then
 * respond". The popup renders each chunk as it arrives, so the answer appears
 * incrementally rather than sitting on "Thinking..." for the full generation.
 */
export const ASK_STREAM_URL = `${BACKEND_BASE_URL}/ask/stream`;

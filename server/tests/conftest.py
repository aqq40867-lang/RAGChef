import os
import sys

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from dotenv import load_dotenv  # noqa: E402

# Load a real server/.env if present, without overriding anything already set
# in the environment (e.g. by CI).
load_dotenv(os.path.join(SERVER_DIR, ".env"))

# Unit tests that never touch the network still need SimpleRAG to construct
# an OpenAI client, which requires *some* API key. Fall back to a placeholder
# if no real key is configured.
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-for-unit-tests")

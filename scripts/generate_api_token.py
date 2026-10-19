"""Generate a fresh bearer token for API_TOKEN in .env.

Usage: poetry run python scripts/generate_api_token.py
"""

import secrets

if __name__ == "__main__":
    print(secrets.token_urlsafe(32))

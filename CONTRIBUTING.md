# Contributing

Thanks for considering a contribution.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Run checks before opening a pull request:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/face-verify-demo-pycache python -m py_compile backend/app/*.py tests/test_verification_guards.py
PYTHONPYCACHEPREFIX=/private/tmp/face-verify-demo-pycache python -m unittest discover -s tests -v
node --check backend/static/app.js
node scripts/test_frontend_static.js
```

## Pull Requests

- Keep changes scoped and include the reason for behavior changes.
- Do not commit real face images, camera captures, biometric templates, secrets, `.env`, virtual environments, or downloaded model weights.
- Include manual verification notes for liveness or anti-spoofing behavior, because automated tests cannot replace camera and replay-attack samples.
- Document the source and license for any third-party model asset used in examples or tests.

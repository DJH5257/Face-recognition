# Test deployment

The test service listens on `127.0.0.1:9004` and reads/writes the test chbzg MySQL database through the DSN in `testing.env.example`. Keep port 9004 behind the HTTPS reverse proxy; do not expose MySQL or the face service directly to the public network.

Before starting the service:

1. Install `requirements.txt` in the shared virtual environment.
2. Copy `testing.env.example` to the server's environment directory and replace every `CHANGE_ME` value.
3. Verify `/api/ready` returns the business table columns and `ok: true`.
4. Run a template sync, login event, failed verification, passed verification and proof finalize against a test doctor.

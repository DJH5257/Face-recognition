# Test deployment

The test service listens on `127.0.0.1:19004` and reads/writes the test chbzg MySQL database through the DSN in `testing.env.example`. Nginx terminates HTTPS on public port `9004` and proxies to `127.0.0.1:19004`; do not expose MySQL or Uvicorn directly to the public network.

Before starting the service:

1. Install `requirements.txt` in the shared virtual environment.
2. Copy `testing.env.example` to the server's environment directory and replace every `CHANGE_ME` value.
3. Verify `/api/ready` returns the business table columns and `ok: true`.
4. Run a template sync, login event, failed verification, passed verification and proof finalize against a test doctor.

After the repository is deployed to `APP_ROOT`, run `sudo deploy/apply-testing.sh`. The script expects the filled `testing.env` file and an existing TLS certificate for `shualian.chbzg.com.cn`; it does not create or print secrets.

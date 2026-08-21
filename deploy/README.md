# Test deployment

The test service listens on `127.0.0.1:19004`. After each doctor verification it writes the result to the existing `fa_face_verify_regulator_status` table and, when enabled, associates it with `fa_doctor_face_verify_log` through the DSN in `testing.env.example`; it does not modify other business tables. Nginx terminates HTTPS on the dedicated doctor face-verification port `9004` and proxies to `127.0.0.1:19004`. The existing business site on `443` is unchanged; only doctor login/verification responses should return a `verify_url` on port `9004`.

Before starting the service:

1. Install `requirements.txt` in the shared virtual environment.
2. Copy `testing.env.example` to the server's environment directory and replace every `CHANGE_ME` value.
3. Verify `/api/ready` returns both enabled result-table schemas and `ok: true`.
4. Run a failed verification and a passed verification against a test doctor, then correlate `fa_face_verify_regulator_status` and `fa_doctor_face_verify_log` by `business_event_id`.

The business login policy must keep `face_required=false` for non-doctor flows. For a doctor, after password verification, return the face session's `verify_url`, `session_id`, `upload_token`, actions and labels; the browser then uploads frames to port `9004` and completes the login proof through the existing business callback.

After the repository is deployed to `APP_ROOT`, run `sudo deploy/apply-testing.sh`. The script expects the filled `testing.env` file and an existing TLS certificate for `shualian.chbzg.com.cn`; it does not create or print secrets.

# Security Policy

This repository is a demo and is not a production identity-verification system.

## Reporting a Vulnerability

Please report vulnerabilities privately to the repository maintainers. Include:

- affected version or commit,
- reproduction steps,
- expected and actual impact,
- any relevant request payloads or logs with personal data removed.

## Biometric Data

Do not open public issues or pull requests containing real face images, camera captures, face embeddings, identity documents, or private user data.

## Known Security Limits

- The demo uses in-memory state and has no authentication, authorization, rate limiting, persistence, audit trail, or fraud-risk engine.
- Browser frames and timestamps are client supplied and can be manipulated by an attacker.
- RGB camera liveness and anti-spoofing models are probabilistic and can fail under new cameras, lighting, replay media, masks, deepfakes, or adversarial inputs.
- Default thresholds are demo defaults only and must be calibrated on representative live and attack samples.

Use a professional biometric SDK, device attestation, server-side risk controls, privacy review, legal review, and dedicated PAD evaluation before production use.

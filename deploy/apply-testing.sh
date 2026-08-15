#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${1:-/opt/face-verify-demo/current}"
ENV_ROOT="${2:-/opt/face-verify-demo/shared/env}"
SERVICE_NAME="face-verify-testing"

test -f "$ENV_ROOT/testing.env" || {
  echo "Missing $ENV_ROOT/testing.env; copy testing.env.example and fill secrets first." >&2
  exit 1
}
test -f "$APP_ROOT/deploy/face-verify-testing.service"
test -f "$APP_ROOT/deploy/nginx-testing-9004.conf"

install -m 0644 "$APP_ROOT/deploy/face-verify-testing.service" "/etc/systemd/system/$SERVICE_NAME.service"
install -m 0644 "$APP_ROOT/deploy/nginx-testing-9004.conf" /etc/nginx/conf.d/face-verify-testing-9004.conf

systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"
nginx -t
systemctl reload nginx

ss -lntp | grep -E ':(9004|19004)\b'
curl --fail --silent --show-error --max-time 15 \
  -H 'Host: shualian.chbzg.com.cn' \
  https://127.0.0.1:9004/api/health \
  --insecure
printf '\nTest face service is listening on HTTPS port 9004.\n'

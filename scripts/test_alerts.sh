#!/usr/bin/env bash
# Run promtool unit tests + syntax check against monitoring/alerts/*.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> promtool check rules (syntax)"
docker run --rm --entrypoint promtool \
  -v "${ROOT}/monitoring/alerts:/etc/prometheus/alerts:ro" \
  prom/prometheus:v2.54.1 \
  check rules /etc/prometheus/alerts/serving.yml /etc/prometheus/alerts/system.yml /etc/prometheus/alerts/infra.yml

echo
echo "==> promtool test rules (unit)"
docker run --rm --entrypoint promtool \
  -v "${ROOT}/monitoring/alerts:/etc/prometheus/alerts:ro" \
  -v "${ROOT}/tests:/tests:ro" \
  prom/prometheus:v2.54.1 \
  test rules /tests/alerts_test.yml

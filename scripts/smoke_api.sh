#!/usr/bin/env bash
# Smoke test for the running API + dashboard containers (used by CI).
set -u
for i in $(seq 1 40); do curl -sf localhost:8000/health >/dev/null && break; sleep 3; done
echo "health: $(curl -s localhost:8000/health)"
body='{"request_id":"ci-smoke","features":{"Gender":"F","Age":26,"Neighbourhood":"JARDIM DA PENHA","Scholarship":0,"SMS_received":1,"lead_time_days":35,"appointment_dow":2,"appointment_month":5,"prior_appointments":4,"prior_noShows":3}}'
code=$(curl -s -o /tmp/predict.json -w '%{http_code}' -X POST localhost:8000/predict -H 'Content-Type: application/json' -d "$body")
echo "predict -> $code: $(cat /tmp/predict.json)"
[ "$code" = "200" ] || { docker compose logs api --tail=40; exit 1; }
bad=$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"features":{"Age":40}}')
echo "invalid input -> $bad"; [ "$bad" = "422" ] || exit 1
for i in $(seq 1 40); do curl -sf localhost:8501/_stcore/health >/dev/null && break; sleep 3; done
echo "dashboard: $(curl -s localhost:8501/_stcore/health)"; curl -sf localhost:8501/_stcore/health >/dev/null || exit 1
echo "explanation store: $(docker compose exec -T api tail -n 1 logs/explanations.jsonl)"

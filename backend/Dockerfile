FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Opt-in model runtime (torch + scikit-learn, about 1 GB). Off by default: without it an approved trained model cannot
# run in this service and the placeholder keeps scoring (GET /api/models/status says so). Turn on with
# --build-arg WITH_MODEL_RUNTIME=1 once a model has been trained, approved and cleared to leave the Workspace.
ARG WITH_MODEL_RUNTIME=0
COPY requirements-inference.txt .
RUN if [ "$WITH_MODEL_RUNTIME" = "1" ]; then pip install --no-cache-dir -r requirements-inference.txt; fi

COPY . .

ENV PORT=8000
EXPOSE 8000

# --proxy-headers / --forwarded-allow-ips='*': Render terminates TLS at its
# edge and forwards plain HTTP to this container, setting X-Forwarded-Proto/
# X-Forwarded-For. Without these flags uvicorn ignores those headers, so the
# app can't tell it's actually being reached over HTTPS, and auth/routes.py's
# client-IP extraction (used by the audit log) would see Render's internal
# proxy address instead of the real client. '*' is safe here specifically
# because Render's edge is the only thing that can reach this container —
# it isn't exposed directly to the internet.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]

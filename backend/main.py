"""
CardioAI Pro — Main Engine Entrypoint
=========================================
Serves the API under /api/*, the live WebSocket at /ws/stream, and the
dashboard frontend as static files at /. One process, one Render service —
backend and frontend are connected by construction, not configuration.

REAL, ORGANIZATION-SCOPED RBAC, REPLACING THE EARLIER SHARED HTTP BASIC
AUTH ENTIRELY: a single shared username/password had no concept of
individual users, roles, or approval — anyone with the one credential
had full access to everything. This is a genuine identity system: real
Postgres-backed users and organizations (auth/), the first person to
sign up for an organization becomes its super_admin automatically,
every subsequent signup for that same organization is created as
pending and routed to that org's admin(s) for approval before they can
log in at all, and each of the three gated surfaces below is only
reachable by the roles that actually belong there.

THE DATA ROOM IS DELIBERATELY OUTSIDE THIS ENTIRELY, NOT A FOURTH
GATED SURFACE — due-diligence investors come from many different
firms, not one organization's account system, so requiring them to
sign up for and be approved into the company's own org would be wrong
for how a data room actually gets used. It has its own separate,
single-shared-password mechanism instead (still real, still
server-enforced — see DataRoomGate below — just intentionally not
tied to any user/org identity at all).

THIS IS NOT A HIPAA OR SOC 2 COMPLIANCE CLAIM. It's real authentication
and real authorization, which are necessary pieces of both — but SOC 2
requires an actual third-party audit over an observation period, and
HIPAA requires Business Associate Agreements with every vendor touching
PHI (this deployment's current infrastructure has none), a formal risk
assessment, and organizational policies that no amount of code
produces. This system should keep running on synthetic data, not real
PHI, until those separate, non-technical requirements are actually met.

SESSION MECHANISM: an httpOnly cookie, not a Bearer token, for both the
RBAC system and the Data Room gate. A Bearer token only reaches the
server when JavaScript deliberately attaches it — a plain browser
navigation (typing the URL, a bookmark, a reload) cannot attach a
custom header at all, so gating page loads behind a Bearer-only check
would mean the pages could never actually load except via JS-initiated
fetch(). An httpOnly cookie is sent automatically by the browser on
every request to the origin, including plain page loads, and isn't
readable by JavaScript — the standard mechanism for this.

RAW ASGI MIDDLEWARE, NOT STARLETTE'S BaseHTTPMiddleware, for the same
reason as before: BaseHTTPMiddleware only sees HTTP requests and
silently lets WebSocket connections straight through unauthenticated,
which would leave /ws/stream open regardless of anything installed
above it. Operating at the ASGI level catches both scope types.

FAILS CLOSED: if a request carries no valid session (or, for the Data
Room, no valid access code), access is refused — never silently
downgraded to open. /healthz, /login.html (and the small set of static
assets it needs), /data_room.html itself (so the password prompt can
load — see DataRoomGate for what's actually protected), and the
signup/login/data-room-unlock endpoints are the only exemptions, and
they're exemptions from *authentication*, not from anything else —
each still enforces its own real check inside its own route handler.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from api.routes import router as api_router, ws_router
from integrations.iomt_bridge import router as iomt_bridge_router
from auth.routes import router as auth_router
from auth.db import init_db
from auth.security import decode_session_token
from auth.models import Role
from auth.deps import SESSION_COOKIE_NAME, DATA_ROOM_COOKIE_NAME

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Monitoring/alerting — optional, real, off by default. Closes a gap the
# data room's own Technical Architecture Packet disclosed plainly:
# "no ... monitoring/alerting tooling found in the codebase today." This
# is genuine Sentry error tracking (unhandled exceptions, and a sample of
# request traces), not a stub — but it only activates once a real
# SENTRY_DSN is set (render.yaml adds the env var slot, sync: false, so
# it's never committed), because there is no Sentry project to point at
# yet. Import is wrapped so a deploy where sentry-sdk somehow isn't
# installed still boots the app rather than crashing on a monitoring
# dependency — monitoring tooling failing closed and taking down the
# whole service would be the wrong failure mode for this specific piece.
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=os.environ.get("RENDER_SERVICE_NAME", "cardioai-pro-engine"),
            # 10% of requests get full performance traces — enough to see
            # real latency/error patterns without the overhead or the
            # Sentry-quota cost of tracing every single request.
            traces_sample_rate=0.1,
            send_default_pii=False,  # never send request bodies/headers that could carry PHI
        )
    except Exception:
        # A monitoring-tool misconfiguration (bad DSN, package issue)
        # should never be why the actual application fails to start.
        pass

app = FastAPI(title="CardioAI Pro — Main Engine", version="0.3.0")


@app.on_event("startup")
def _create_tables_on_startup():
    # Creates the organizations/users tables if they don't exist yet —
    # safe to call on every startup (SQLAlchemy's create_all is a no-op
    # for tables that already exist), so a redeploy never needs a
    # separate manual migration step for this initial schema.
    init_db()


@app.on_event("startup")
def _seed_legal_docs_into_data_room():
    # Real BAA/Privacy Statement/Terms of Use, uploaded into the actual
    # Document Room registry (not just served as static pages) so they
    # show up in the Data Room and satisfy the corresponding checklist
    # rows (7.9/7.10/2.2, 7.10, 7.11) automatically, without an admin
    # having to manually upload them. Checks for an existing match by
    # filename first — idempotent on restart, which matters once R2 is
    # configured and documents genuinely persist across redeploys;
    # re-seeding duplicates every restart would defeat that persistence.
    from orchestrator.orchestrator import orchestrator
    seeds = [
        ("Business_Associate_Agreement.html", "baa.html", "regulatory", "Business Associate Agreement (BAA)"),
        ("Privacy_Statement.html", "privacy-statement.html", "regulatory", "Privacy Statement"),
        # Filed under "regulatory" (not "other") so all three legal
        # documents land together under Regulatory & Compliance in the
        # Document Room and the Data Room Index, rather than splitting
        # Terms of Use off into a different category for no functional
        # reason.
        ("Terms_of_Use.html", "terms-of-use.html", "regulatory", "Terms of Use"),
    ]
    existing_filenames = {d.filename for d in orchestrator.document_registry.list_all()}
    for display_name, source_name, category, description in seeds:
        if display_name in existing_filenames:
            continue
        source_path = FRONTEND_DIR / "legal" / source_name
        if not source_path.exists():
            continue
        content = source_path.read_bytes()
        orchestrator.document_registry.save(display_name, category, description, content)


# Which role set each org-RBAC-gated surface allows. Data Room is
# deliberately absent — it isn't part of this system at all; see
# DataRoomGate below for how it's actually protected.
SURFACE_ROLES: dict[str, set[Role]] = {
    "main_engine": {Role.super_admin, Role.admin, Role.clinician},
    "clinician_dashboard": {Role.super_admin, Role.admin, Role.clinician},
    "payer_portal": {Role.super_admin, Role.admin, Role.payer_analyst},
    "org_admin": {Role.super_admin, Role.admin},
}


def _surface_for_path(path: str) -> str | None:
    """Maps a request path to the org-RBAC surface it belongs to, or None if unrestricted (still requires login, just no role check beyond being an active member). Data Room paths are never mapped here — DataRoomGate handles them separately, before this is even consulted."""
    if path in ("/", "/index.html"):
        return "main_engine"
    if path == "/clinician_full_dashboard.html":
        return "clinician_dashboard"
    if path == "/payer_portal.html":
        return "payer_portal"
    if path == "/admin.html":
        return "org_admin"
    if path.startswith("/api/payer/"):
        return "payer_portal"
    if path in ("/ws/stream",) or (path.startswith("/api/") and not path.startswith("/api/auth/") and not path.startswith("/api/data-room/")):
        return "main_engine"  # the shared clinical API surface (ingest, patients, agents, care-continuum, audit, etc.) — clinician role covers this too since SURFACE_ROLES["main_engine"] includes it
    return None


PUBLIC_PATHS = {"/healthz", "/login.html", "/data_room.html", "/api/auth/signup", "/api/auth/login", "/api/data-room/unlock", "/api/data-room/lock"}
PUBLIC_PREFIXES = ("/assets/", "/legal/")  # /assets/ = logo etc. for unauthenticated pages; /legal/ = Terms/Privacy/BAA, readable before signup and by anyone with the link

# Data Room API paths — protected by DataRoomGate's separate single-password
# check below, never by org-RBAC. /data_room.html itself is in PUBLIC_PATHS
# above (the page, including its password-prompt UI, loads for anyone); it's
# specifically the endpoints serving the real uploaded due-diligence
# documents that require the access code.
DATA_ROOM_API_PREFIX = "/api/data-room/"


def _cookie_value(scope: Scope, name: str) -> str | None:
    headers = dict(scope.get("headers", []))
    cookie_header = headers.get(b"cookie", b"").decode("latin1")
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith(f"{name}="):
            return part[len(name) + 1:]
    return None


class RBACSessionMiddleware:
    """Raw ASGI middleware — see module docstring for why not BaseHTTPMiddleware, and why a cookie not a Bearer token. Handles both the org-RBAC system and the separate Data Room single-password gate, since a request is routed to at most one of the two."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def _refuse(self, scope: Scope, send: Send, status: int, body: bytes) -> None:
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
        else:
            await send({"type": "websocket.close", "code": 1008})

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES) or path.startswith("/api/auth/"):
            # /api/auth/* beyond signup/login (e.g. /me, /approve) still authenticates —
            # via FastAPI's own Depends(get_current_user)/Depends(require_admin) inside
            # auth/routes.py, which read the same cookie. Exempting the *path* here just
            # means this outer layer doesn't duplicate that check; the route still does it.
            await self.app(scope, receive, send)
            return

        # Data Room API paths: entirely separate from org-RBAC, checked first
        # so they never fall through to the organization login requirement below.
        if path.startswith(DATA_ROOM_API_PREFIX):
            token = _cookie_value(scope, DATA_ROOM_COOKIE_NAME)
            payload = decode_session_token(token) if token else None
            if not payload or payload.get("scope") != "data_room_access":
                await self._refuse(scope, send, 401, b'{"detail":"Data room access code required."}')
                return
            await self.app(scope, receive, send)
            return

        # Everything else: the org-RBAC system.
        token = _cookie_value(scope, SESSION_COOKIE_NAME)
        payload = decode_session_token(token) if token else None
        surface = _surface_for_path(path)

        if not payload:
            await self._refuse(scope, send, 401, b'{"detail":"Not logged in.","login_url":"/login.html"}')
            return

        if surface is not None:
            allowed_roles = {r.value for r in SURFACE_ROLES[surface]}
            if payload.get("role") not in allowed_roles:
                await self._refuse(scope, send, 403, b'{"detail":"Your role does not have access to this."}')
                return

        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """
    Raw ASGI middleware (same reasoning as RBACSessionMiddleware above —
    BaseHTTPMiddleware silently lets WebSocket scopes through
    unmodified, which is fine for header injection specifically, but
    kept consistent with the rest of this file rather than mixing
    middleware styles). Adds HSTS so a browser that has ever loaded this
    app over HTTPS refuses to downgrade to plain HTTP for the pinned
    duration, even if something upstream (a misconfigured proxy, a
    stale link) ever serves it over HTTP — defense in depth on top of,
    not a replacement for, Render terminating TLS at its edge.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"strict-transport-security", b"max-age=63072000; includeSubDomains"))
                headers.append((b"x-content-type-options", b"nosniff"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your hospital/payer domains in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,  # required for the session cookie to be sent on cross-origin requests, if the frontend is ever split onto a different origin than the API
)

app.include_router(api_router, prefix="/api")
app.include_router(iomt_bridge_router, prefix="/api")  # /api/iomt-bridge/ingest — the seam with the external cardioailiverpm.com system
app.include_router(auth_router, prefix="/api")  # /api/auth/* — signup, login, approval workflow, AND /api/auth/data-room/unlock+lock (see auth/routes.py)
app.include_router(ws_router)  # /ws/stream — kept off the /api prefix so it isn't shadowed by the static mount


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# Serve the dashboard. Mounted last so /api and /healthz take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Wraps the whole ASGI app, including the static mount and the WebSocket
# router — added last (outermost) so nothing added above it can bypass it.
# SecurityHeadersMiddleware is outermost of all: it must see every
# response, including the 401s RBACSessionMiddleware itself generates.
app = RBACSessionMiddleware(app)
app = SecurityHeadersMiddleware(app)

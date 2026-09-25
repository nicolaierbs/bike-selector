"""Command line tools: OAuth bootstrap, webhook management, training, dry runs."""

from __future__ import annotations

import http.server
import json
import logging
import threading
import urllib.parse
import webbrowser
from typing import Annotated

import typer

from .config import get_settings
from .model import BikeClassifier, InsufficientData
from .service import BikeSelectorService
from .strava import StravaGateway

app = typer.Typer(add_completion=False, help="Strava bike selector utilities.")
webhook_app = typer.Typer(help="Manage the Strava push subscription.")
app.add_typer(webhook_app, name="webhook")

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

SCOPES = ["read", "activity:read_all", "activity:write", "profile:read_all"]


def _echo(message: str, *, err: bool = False) -> None:
    typer.echo(message, err=err)


# --------------------------------------------------------------------- auth


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None
    error: str | None = None

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _CallbackHandler.code = (params.get("code") or [None])[0]
        _CallbackHandler.error = (params.get("error") or [None])[0]
        body = (
            b"<h2>Authorised.</h2><p>You can close this tab and return to the terminal.</p>"
            if _CallbackHandler.code
            else b"<h2>Authorisation failed.</h2><p>Check the terminal.</p>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence the default access log
        return


@app.command()
def auth(
    no_browser: Annotated[bool, typer.Option(help="Print the URL instead of opening it.")] = False,
) -> None:
    """Run the one-time OAuth flow and print the refresh token to put in .env."""
    settings = get_settings()
    if not (settings.strava_client_id and settings.strava_client_secret):
        _echo("Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET first (see .env.example).", err=True)
        raise typer.Exit(1)

    redirect_uri = f"http://localhost:{settings.oauth_local_port}/exchange"
    from stravalib import Client

    client = Client()
    url = client.authorization_url(
        client_id=settings.strava_client_id,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        approval_prompt="force",
    )

    _echo("Make sure your Strava app's Authorization Callback Domain is: localhost")
    _echo(f"\nAuthorise here:\n{url}\n")
    if not no_browser:
        webbrowser.open(url)

    server = http.server.HTTPServer(("127.0.0.1", settings.oauth_local_port), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    thread.join(timeout=300)
    server.server_close()

    if _CallbackHandler.error:
        _echo(f"Strava returned an error: {_CallbackHandler.error}", err=True)
        raise typer.Exit(1)
    if not _CallbackHandler.code:
        _echo("Timed out waiting for the callback.", err=True)
        raise typer.Exit(1)

    info = client.exchange_code_for_token(
        client_id=settings.strava_client_id,
        client_secret=settings.strava_client_secret,
        code=_CallbackHandler.code,
    )
    _echo("\nAdd this to your .env and to the Render environment:\n")
    _echo(f"STRAVA_REFRESH_TOKEN={info['refresh_token']}")


# -------------------------------------------------------------------- gear


@app.command()
def bikes() -> None:
    """List the bikes on the Strava account."""
    for bike in StravaGateway().bikes():
        flags = " ".join(f for f, on in (("primary", bike.primary), ("retired", bike.retired)) if on)
        _echo(f"{bike.id:<16} {bike.name:<32} {bike.distance_m / 1000:>9.0f} km  {flags}")


# ------------------------------------------------------------------ model


@app.command()
def train(
    limit: Annotated[int, typer.Option(help="How many past activities to pull.")] = 0,
    save: Annotated[bool, typer.Option(help="Write the model to the state dir.")] = True,
) -> None:
    """Train the classifier and print a cross-validated report."""
    settings = get_settings()
    gateway = StravaGateway(settings)
    classifier = BikeClassifier(settings)
    activities = gateway.recent_activities(limit=limit or None)
    _echo(f"Fetched {len(activities)} activities.")
    bike_names = gateway.bike_names()
    service = BikeSelectorService(gateway=gateway, classifier=classifier, settings=settings)
    untrusted = service.untrusted_label_ids(activities, bike_names)
    if untrusted:
        _echo(f"Ignoring {len(untrusted)} recent ride(s) whose bike is still this app's guess.")
    try:
        report = classifier.train(activities, bike_names=bike_names, exclude_ids=untrusted)
    except InsufficientData as exc:
        _echo(f"Cannot train: {exc}", err=True)
        raise typer.Exit(1) from exc
    _echo(report.summary())
    if save:
        _echo(f"\nSaved to {classifier.save()}")


@app.command()
def predict(
    activity_id: Annotated[int, typer.Argument(help="Strava activity id.")],
) -> None:
    """Show the probabilities for one ride without changing anything."""
    outcome = BikeSelectorService().process_activity(activity_id, force=True, dry_run=True)
    _echo(json.dumps(outcome.as_dict(), indent=2))


@app.command()
def apply(
    activity_id: Annotated[int, typer.Argument(help="Strava activity id.")],
    force: Annotated[bool, typer.Option(help="Re-run even if already annotated.")] = False,
) -> None:
    """Predict and actually update the activity on Strava."""
    outcome = BikeSelectorService().process_activity(activity_id, force=force)
    _echo(json.dumps(outcome.as_dict(), indent=2))


@app.command()
def backfill(
    days: Annotated[int, typer.Option(help="Look back this many days.")] = 30,
    limit: Annotated[int, typer.Option(help="Max activities to scan.")] = 50,
    include_labelled: Annotated[bool, typer.Option(help="Also touch rides that have gear.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Report without writing.")] = True,
) -> None:
    """Process recent rides a missed webhook left behind."""
    outcomes = BikeSelectorService().backfill(
        days=days, limit=limit, only_missing_gear=not include_labelled, dry_run=dry_run
    )
    if not outcomes:
        _echo("Nothing to do.")
        return
    for outcome in outcomes:
        _echo(f"{outcome.activity_id:<14} {outcome.status:<10} {outcome.detail}")
    if dry_run:
        _echo("\nDry run — pass --no-dry-run to write these to Strava.")


# ---------------------------------------------------------------- webhooks


@webhook_app.command("list")
def webhook_list() -> None:
    """Show the current push subscription, if any."""
    _echo(json.dumps(StravaGateway().list_subscriptions(), indent=2))


@webhook_app.command("subscribe")
def webhook_subscribe(
    callback_url: Annotated[str, typer.Option(help="Overrides PUBLIC_BASE_URL.")] = "",
) -> None:
    """Register the webhook. The service must already be deployed and reachable."""
    settings = get_settings()
    url = callback_url or settings.callback_url
    if not url.startswith("https://"):
        _echo(f"Callback must be public HTTPS, got: {url!r}", err=True)
        raise typer.Exit(1)
    _echo(f"Registering {url}")
    result = StravaGateway(settings).create_subscription(url, settings.webhook_verify_token)
    _echo(json.dumps(result, indent=2))


@webhook_app.command("delete")
def webhook_delete(
    subscription_id: Annotated[int, typer.Argument(help="Id from `webhook list`.")],
) -> None:
    """Remove the push subscription (Strava allows only one per app)."""
    StravaGateway().delete_subscription(subscription_id)
    _echo(f"Deleted subscription {subscription_id}.")


# ------------------------------------------------------------------ server


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
) -> None:
    """Run the webhook service locally."""
    import uvicorn

    uvicorn.run("bike_selector.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()

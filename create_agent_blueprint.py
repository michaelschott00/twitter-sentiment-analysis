#!/usr/bin/env python3
"""
Create an Agent Identity Blueprint via Microsoft Graph API (local-dev only).

Implements the minimal workflow from:
https://learn.microsoft.com/en-us/entra/agent-id/create-blueprint?tabs=microsoft-graph-api

Steps:
  1. Acquire token (service principal or user) for Microsoft Graph.
  2. POST /applications/microsoft.graph.agentIdentityBlueprint -> create blueprint
  3. POST /applications/{objectId}/addPassword -> client secret (local dev)
  4. POST /servicePrincipals/microsoft.graph.agentIdentityBlueprintPrincipal -> principal

Auth secrets are read from environment variables (never hard-coded, no .env).
Remaining settings are CLI arguments.

Environment variables (service principal):
  ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET
  Optional: ACCESS_TOKEN (pre-acquired Graph token, skips acquisition)
  For user auth the same ARM_TENANT_ID/ARM_CLIENT_ID are reused (no secret needed).

Usage:
  # Service principal (client credentials)
  export ARM_TENANT_ID=... ARM_CLIENT_ID=... ARM_CLIENT_SECRET=...
  python create_agent_blueprint.py --sponsor-id <user-object-id> --verbose

  # User authentication - device code (delegated, interactive browser prompt)
  export ARM_TENANT_ID=... ARM_CLIENT_ID=...
  python create_agent_blueprint.py --auth-mode user-device-code --sponsor-upn sponsor@contoso.com --verbose

  # User authentication - interactive browser
  python create_agent_blueprint.py --auth-mode user-interactive --sponsor-id <id> --verbose

  # Azure CLI (uses `az account get-access-token`)
  az login
  python create_agent_blueprint.py --auth-mode azure-cli --sponsor-id <id>

  # Auto-detect (default): tries service principal if ARM_CLIENT_SECRET set, otherwise device code
  python create_agent_blueprint.py --sponsor-id <id> --dry-run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import click
import requests

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
DEFAULT_SCOPE = "https://graph.microsoft.com/.default"

# Delegated scopes required for blueprint creation (used for user auth).
# .default can also be used if permissions are pre-consented.
DELEGATED_SCOPES = [
    "https://graph.microsoft.com/AgentIdentityBlueprint.Create",
    "https://graph.microsoft.com/AgentIdentityBlueprint.AddRemoveCreds.All",
    "https://graph.microsoft.com/AgentIdentityBlueprintPrincipal.Create",
    "https://graph.microsoft.com/User.Read",
]


def _redact(s: str | None, keep: int = 4) -> str:
    if not s:
        return "<empty>"
    if len(s) <= keep:
        return "***"
    return s[:keep] + "***" + s[-keep:]


def _acquire_token_service_principal(
    tenant: str, client_id: str, client_secret: str, verbose: bool = False
) -> str:
    url = TOKEN_URL_TMPL.format(tenant=tenant)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": DEFAULT_SCOPE,
        "grant_type": "client_credentials",
    }
    if verbose:
        click.echo(
            f"[service-principal] Requesting token from {url} client_id={_redact(client_id)} tenant={_redact(tenant)}"
        )
    resp = requests.post(url, data=data, timeout=30)
    if not resp.ok:
        click.echo(
            f"Token acquisition failed: {resp.status_code} {resp.text}", err=True
        )
        click.echo(
            "Ensure the app has Application permissions granted by Privileged Role Administrator:",
            err=True,
        )
        click.echo(
            "  AgentIdentityBlueprint.Create, AgentIdentityBlueprint.AddRemoveCreds.All, AgentIdentityBlueprintPrincipal.Create",
            err=True,
        )
        resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        click.echo(f"No access_token in response: {resp.text}", err=True)
        sys.exit(1)
    click.echo("Token acquired successfully (service principal)")
    return token


def _acquire_token_device_code(
    tenant: str, client_id: str, scopes: list[str], verbose: bool = False
) -> str:
    try:
        import msal
    except ImportError:
        click.echo("MSAL not installed. Install with: pip install msal", err=True)
        sys.exit(1)
    authority = f"https://login.microsoftonline.com/{tenant}"
    app = msal.PublicClientApplication(client_id, authority=authority)
    flow = app.initiate_device_flow(scopes=scopes)
    if "user_code" not in flow:
        click.echo(
            f"Failed to create device flow: {json.dumps(flow, indent=2)}", err=True
        )
        click.echo(
            "Ensure the app registration allows public client flows (Enable public client flows = Yes) and has delegated permissions consented.",
            err=True,
        )
        sys.exit(1)
    # flow["message"] already contains instructions: "To sign in, use a web browser to open ... and enter code ..."
    click.echo(flow["message"])
    if verbose:
        click.echo(
            f"[device-code] scopes={scopes} authority={authority} client_id={_redact(client_id)}"
        )
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        click.echo(f"Device code flow failed: {json.dumps(result, indent=2)}", err=True)
        sys.exit(1)
    click.echo("Token acquired successfully (user / device code)")
    return result["access_token"]


def _acquire_token_interactive(
    tenant: str, client_id: str, scopes: list[str], verbose: bool = False
) -> str:
    try:
        import msal
    except ImportError:
        click.echo("MSAL not installed. Install with: pip install msal", err=True)
        sys.exit(1)
    authority = f"https://login.microsoftonline.com/{tenant}"
    app = msal.PublicClientApplication(client_id, authority=authority)
    if verbose:
        click.echo(
            f"[interactive] authority={authority} client_id={_redact(client_id)} scopes={scopes}"
        )
    result = app.acquire_token_interactive(scopes=scopes)
    if "access_token" not in result:
        click.echo(f"Interactive auth failed: {json.dumps(result, indent=2)}", err=True)
        sys.exit(1)
    click.echo("Token acquired successfully (user / interactive)")
    return result["access_token"]


def _acquire_token_azure_cli(verbose: bool = False) -> str:
    # Uses `az account get-access-token --resource https://graph.microsoft.com`
    cmd = [
        "az",
        "account",
        "get-access-token",
        "--resource",
        "https://graph.microsoft.com",
        "--query",
        "accessToken",
        "-o",
        "tsv",
    ]
    if verbose:
        click.echo(f"[azure-cli] Running: {' '.join(cmd)}")
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30).strip()
    except FileNotFoundError:
        click.echo(
            "Azure CLI not found. Install with: pip install azure-cli and run `az login`.",
            err=True,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        click.echo(f"az account get-access-token failed: {e}", err=True)
        click.echo(
            "Run `az login` and `az account set --subscription <id>` first.", err=True
        )
        sys.exit(1)
    if not out or " " in out:
        # tsv should be a single JWT; if empty or error
        click.echo(f"Unexpected az output: {out!r}", err=True)
        sys.exit(1)
    click.echo("Token acquired successfully (azure-cli)")
    return out


def acquire_token(auth_mode: str, verbose: bool = False) -> str:
    access_token = os.getenv("ACCESS_TOKEN", "").strip()
    if access_token:
        click.echo("Using ACCESS_TOKEN from environment (skipping token acquisition)")
        return access_token

    tenant = os.getenv("ARM_TENANT_ID", "").strip()
    client_id = os.getenv("ARM_CLIENT_ID", "").strip()
    client_secret = os.getenv("ARM_CLIENT_SECRET", "").strip()

    # Normalize auth_mode
    mode = auth_mode.lower()
    if mode == "auto":
        # Prefer service principal if secret is available, otherwise device code
        if client_secret:
            mode = "service-principal"
        else:
            mode = "user-device-code"

    if verbose:
        click.echo(f"Auth mode resolved: requested={auth_mode} -> effective={mode}")

    if mode == "service-principal":
        missing = [
            k
            for k in ("ARM_TENANT_ID", "ARM_CLIENT_ID", "ARM_CLIENT_SECRET")
            if not os.getenv(k, "").strip()
        ]
        if missing:
            click.echo(
                f"Missing required environment variables for service principal: {', '.join(missing)}",
                err=True,
            )
            click.echo(
                "Set ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET, or use --auth-mode user-device-code / azure-cli.",
                err=True,
            )
            sys.exit(1)
        return _acquire_token_service_principal(
            tenant, client_id, client_secret, verbose=verbose
        )

    if mode in ("user-device-code", "device-code"):
        if not tenant or not client_id:
            click.echo("Missing ARM_TENANT_ID / ARM_CLIENT_ID for user auth", err=True)
            click.echo(
                "Set ARM_TENANT_ID and ARM_CLIENT_ID (no secret needed for public client).",
                err=True,
            )
            sys.exit(1)
        return _acquire_token_device_code(
            tenant, client_id, DELEGATED_SCOPES, verbose=verbose
        )

    if mode in ("user-interactive", "interactive"):
        if not tenant or not client_id:
            click.echo("Missing ARM_TENANT_ID / ARM_CLIENT_ID for user auth", err=True)
            sys.exit(1)
        return _acquire_token_interactive(
            tenant, client_id, DELEGATED_SCOPES, verbose=verbose
        )

    if mode == "azure-cli":
        return _acquire_token_azure_cli(verbose=verbose)

    click.echo(
        f"Unknown auth mode: {auth_mode} (choices: auto, service-principal, user-device-code, user-interactive, azure-cli)",
        err=True,
    )
    sys.exit(1)


def resolve_user_id(token: str, upn: str, verbose: bool = False) -> str:
    """Resolve a UPN to objectId via GET /users/{upn}."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{GRAPH_BASE}/users/{upn}"
    if verbose:
        click.echo(f"Resolving UPN {upn} -> {url}")
    resp = requests.get(url, headers=headers, timeout=30)
    if not resp.ok:
        click.echo(
            f"Failed to resolve UPN {upn}: {resp.status_code} {resp.text}", err=True
        )
        resp.raise_for_status()
    return resp.json()["id"]


def create_blueprint(
    token: str,
    display_name: str,
    sponsor_id: str,
    owner_id: str | None,
    verbose: bool = False,
) -> dict:
    """
    POST /applications/microsoft.graph.agentIdentityBlueprint
    Returns JSON with `id` (objectId) and `appId`.
    Requires header OData-Version: 4.0
    """
    url = f"{GRAPH_BASE}/applications/microsoft.graph.agentIdentityBlueprint"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "OData-Version": "4.0",
    }
    body: dict = {
        "@odata.type": "Microsoft.Graph.AgentIdentityBlueprint",
        "displayName": display_name,
    }
    if sponsor_id:
        body["sponsors@odata.bind"] = [
            f"https://graph.microsoft.com/v1.0/users/{sponsor_id}"
        ]
    else:
        click.echo("ERROR: sponsor is required to create a blueprint", err=True)
        sys.exit(1)
    if owner_id:
        body["owners@odata.bind"] = [
            f"https://graph.microsoft.com/v1.0/users/{owner_id}"
        ]

    if verbose:
        click.echo(f"POST {url}\n{json.dumps(body, indent=2)}")
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if not resp.ok:
        if resp.status_code in (400, 404):
            click.echo(
                f"Primary endpoint failed ({resp.status_code}), trying fallback POST /applications",
                err=True,
            )
            fallback_url = f"{GRAPH_BASE}/applications"
            resp2 = requests.post(fallback_url, headers=headers, json=body, timeout=30)
            if resp2.ok:
                resp = resp2
            else:
                click.echo(
                    f"Create blueprint failed: {resp.status_code} {resp.text}", err=True
                )
                click.echo(
                    f"Fallback also failed: {resp2.status_code} {resp2.text}", err=True
                )
                resp.raise_for_status()
        else:
            click.echo(
                f"Create blueprint failed: {resp.status_code} {resp.text}", err=True
            )
            resp.raise_for_status()

    data = resp.json()
    click.echo(
        f"Blueprint created: displayName={data.get('displayName')} id={data.get('id')} appId={data.get('appId')}"
    )
    if verbose:
        click.echo(json.dumps(data, indent=2))
    return data


def add_password_credential(
    token: str,
    app_object_id: str,
    display_name: str,
    end_date: str | None,
    verbose: bool = False,
) -> dict:
    """POST /applications/{id}/addPassword"""
    url = f"{GRAPH_BASE}/applications/{app_object_id}/addPassword"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if not end_date:
        dt = datetime.now(timezone.utc) + timedelta(days=365)
        end_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        click.echo(f"No --credential-end-date set, defaulting to {end_date}")

    body = {
        "passwordCredential": {"displayName": display_name, "endDateTime": end_date}
    }
    if verbose:
        click.echo(f"POST {url}\n{json.dumps(body, indent=2)}")
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if not resp.ok:
        click.echo(f"addPassword failed: {resp.status_code} {resp.text}", err=True)
        if "credential lifetime" in resp.text.lower():
            click.echo(
                "Hint: tenant may restrict max lifetime; try a shorter endDateTime",
                err=True,
            )
        resp.raise_for_status()
    data = resp.json()
    secret = data.get("secretText")
    if secret:
        click.echo(
            f"Secret created: displayName={display_name} (STORE NOW — won't be shown again)"
        )
        if verbose:
            click.echo(f"secretText: {secret}")
        else:
            click.echo("secretText hidden (run with --verbose to display)")
    else:
        click.echo(f"Password credential response: {json.dumps(data, indent=2)}")
    return data


def create_principal(token: str, app_id: str, verbose: bool = False) -> dict:
    """POST /servicePrincipals/microsoft.graph.agentIdentityBlueprintPrincipal"""
    url = f"{GRAPH_BASE}/servicePrincipals/microsoft.graph.agentIdentityBlueprintPrincipal"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "OData-Version": "4.0",
    }
    body = {"appId": app_id}
    if verbose:
        click.echo(f"POST {url}\n{json.dumps(body, indent=2)}")
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if not resp.ok:
        click.echo(f"Create principal failed: {resp.status_code} {resp.text}", err=True)
        resp.raise_for_status()
    data = resp.json()
    click.echo(
        f"Blueprint principal created: id={data.get('id')} appId={data.get('appId')}"
    )
    if verbose:
        click.echo(json.dumps(data, indent=2))
    return data


@click.command()
@click.option(
    "--display-name",
    default="My Agent Identity Blueprint",
    show_default=True,
    help="Blueprint displayName",
)
@click.option(
    "--sponsor-id",
    default=None,
    help="Sponsor user objectId (required unless --sponsor-upn)",
)
@click.option(
    "--sponsor-upn",
    default=None,
    help="Sponsor UPN (resolved via Graph if --sponsor-id not set)",
)
@click.option("--owner-id", default=None, help="Owner user objectId (optional)")
@click.option("--owner-upn", default=None, help="Owner UPN (resolved via Graph)")
@click.option(
    "--credential-display-name",
    default="My Secret",
    show_default=True,
    help="Display name for client secret",
)
@click.option(
    "--credential-end-date",
    default=None,
    help="ISO8601 endDateTime for secret, e.g. 2026-08-05T23:59:59Z (defaults to +365d)",
)
@click.option(
    "--skip-principal", is_flag=True, help="Skip blueprint principal creation"
)
@click.option(
    "--auth-mode",
    type=click.Choice(
        [
            "auto",
            "service-principal",
            "user-device-code",
            "user-interactive",
            "azure-cli",
        ],
        case_sensitive=False,
    ),
    default="auto",
    show_default=True,
    help="Authentication mode: auto (SP if ARM_CLIENT_SECRET set else device code), service-principal (client credentials), user-device-code, user-interactive, azure-cli",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate args and show planned actions without calling Graph",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose logging including secrets (use locally only)",
)
def main(
    display_name,
    sponsor_id,
    sponsor_upn,
    owner_id,
    owner_upn,
    credential_display_name,
    credential_end_date,
    skip_principal,
    auth_mode,
    dry_run,
    verbose,
):
    """Create an Agent Identity Blueprint programmatically (local-dev, client secret only)."""
    if verbose:
        click.echo(
            f"Args: display_name={display_name} sponsor_id={sponsor_id or '<none>'} "
            f"sponsor_upn={sponsor_upn or '<none>'} owner_id={owner_id or '<none>'} "
            f"owner_upn={owner_upn or '<none>'} credential={credential_display_name} auth_mode={auth_mode}"
        )

    token: str | None = None
    if not dry_run:
        token = acquire_token(auth_mode, verbose=verbose)
        # Resolve UPN -> ID if needed
        if not sponsor_id and sponsor_upn:
            sponsor_id = resolve_user_id(token, sponsor_upn, verbose=verbose)
            click.echo(f"Resolved SPONSOR_UPN {sponsor_upn} -> {sponsor_id}")
        if not owner_id and owner_upn:
            owner_id = resolve_user_id(token, owner_upn, verbose=verbose)
            click.echo(f"Resolved OWNER_UPN {owner_upn} -> {owner_id}")
        if not sponsor_id:
            click.echo(
                "Missing sponsor: provide --sponsor-id or --sponsor-upn. "
                "Find ID via: GET https://graph.microsoft.com/v1.0/users?$select=id,displayName,userPrincipalName",
                err=True,
            )
            sys.exit(1)
    else:
        click.echo(
            f"[dry-run] Would acquire token via auth_mode={auth_mode} and validate sponsor/owner"
        )
        if not sponsor_id and not sponsor_upn:
            click.echo(
                "[dry-run] WARNING: no --sponsor-id/--sponsor-upn — will fail at runtime",
                err=True,
            )

    # --- Create blueprint ---
    if dry_run:
        click.echo(
            f"[dry-run] Would create blueprint '{display_name}' sponsor={sponsor_id or '<missing>'} owner={owner_id or '<none>'}"
        )
        fake_object_id = "<object-id>"
        fake_app_id = "<app-id>"
    else:
        assert token is not None
        bp = create_blueprint(
            token, display_name, sponsor_id, owner_id or None, verbose=verbose
        )
        fake_object_id = bp.get("id")
        fake_app_id = bp.get("appId")
        if not fake_object_id or not fake_app_id:
            click.echo(f"Unexpected create response (missing id/appId): {bp}", err=True)
            sys.exit(1)
        click.echo(f"Recorded appId={fake_app_id} objectId={fake_object_id}")

    # --- Client secret (local dev only) ---
    if dry_run:
        if not credential_end_date:
            click.echo("[dry-run] Would default credential endDateTime to +365d")
        click.echo(
            f"[dry-run] Would POST /applications/{fake_object_id}/addPassword displayName={credential_display_name}"
        )
    else:
        assert token is not None
        add_password_credential(
            token,
            fake_object_id,
            credential_display_name,
            credential_end_date,
            verbose=verbose,
        )

    # --- Principal ---
    if skip_principal:
        click.echo("Skipping principal creation (--skip-principal)")
    else:
        if dry_run:
            click.echo(
                f"[dry-run] Would POST /servicePrincipals/microsoft.graph.agentIdentityBlueprintPrincipal appId={fake_app_id}"
            )
        else:
            assert token is not None
            create_principal(token, fake_app_id, verbose=verbose)

    click.echo(
        "\nDone. Blueprint is visible at https://entra.microsoft.com -> Entra ID -> Agents -> Agent blueprints"
    )
    click.echo(
        "Next: create agent identities from this blueprint (see https://learn.microsoft.com/en-us/entra/agent-id/create-delete-agent-identities)"
    )
    if not dry_run:
        click.echo(f"Summary: appId={fake_app_id} objectId={fake_object_id}")


if __name__ == "__main__":
    main()

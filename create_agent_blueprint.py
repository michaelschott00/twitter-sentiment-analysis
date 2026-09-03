#!/usr/bin/env python3
"""
Create an Agent Identity Blueprint and Agent Identities via Microsoft Graph API (local-dev only).

Implements workflows from:
- https://learn.microsoft.com/en-us/entra/agent-id/create-blueprint?tabs=microsoft-graph-api
- https://learn.microsoft.com/en-us/entra/agent-id/create-delete-agent-identities?tabs=microsoft-graph-api

Blueprint steps (subcommand create-blueprint):
  1. Acquire token (service principal or azure-cli) for Microsoft Graph.
  2. POST /applications/microsoft.graph.agentIdentityBlueprint -> create blueprint
  3. POST /applications/{objectId}/addPassword -> client secret (local dev)
  4. POST /servicePrincipals/microsoft.graph.agentIdentityBlueprintPrincipal -> principal

Agent identity steps (subcommands):
  create-identity: POST /beta/serviceprincipals/Microsoft.Graph.AgentIdentity
  delete-identity: DELETE /beta/serviceprincipals/{id}
  Token for agent identity is obtained either via blueprint client credentials
  (preferred for local dev: --blueprint-id + --blueprint-secret) or via
  the same --auth-mode used for blueprints (service-principal / azure-cli).

Auth secrets are read from environment variables (never hard-coded, no .env).
Remaining settings are CLI arguments.

Environment variables (service principal):
  ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET
  Optional: ACCESS_TOKEN (pre-acquired Graph token, skips acquisition)
  For create-identity via blueprint: BLUEPRINT_CLIENT_SECRET (or --blueprint-secret)

Usage:
  # Blueprint
  export ARM_TENANT_ID=... ARM_CLIENT_ID=... ARM_CLIENT_SECRET=...
  python create_agent_blueprint.py create-blueprint --sponsor-id <user-object-id> --verbose
  python create_agent_blueprint.py create-blueprint --sponsor-upn sponsor@contoso.com --display-name "My Blueprint" --dry-run

  # Agent identity via blueprint credentials (local dev)
  python create_agent_blueprint.py create-identity --blueprint-id <blueprint-appId> --display-name "My Agent" --sponsor-id <id> --blueprint-secret <secret>
  # or using env var
  export BLUEPRINT_CLIENT_SECRET=...
  python create_agent_blueprint.py create-identity --blueprint-id <appId> --display-name "My Agent" --sponsor-id <id>

  # Agent identity via service principal / azure-cli token (if that principal is blueprint owner)
  python create_agent_blueprint.py create-identity --blueprint-id <appId> --display-name "My Agent" --sponsor-id <id> --auth-mode azure-cli
  python create_agent_blueprint.py create-identity --blueprint-id <appId> --display-name "My Agent" --sponsor-id <id> --auth-mode service-principal

  # Delete
  python create_agent_blueprint.py delete-identity --identity-id <agent-identity-object-id> --blueprint-id <appId> --blueprint-secret <secret>
  python create_agent_blueprint.py delete-identity --identity-id <id> --auth-mode azure-cli

  # Blueprint with azure-cli
  az login
  python create_agent_blueprint.py create-blueprint --auth-mode azure-cli --sponsor-id <id>
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
GRAPH_BETA = "https://graph.microsoft.com/beta"
TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
DEFAULT_SCOPE = "https://graph.microsoft.com/.default"


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


def _acquire_token_blueprint(
    tenant: str, blueprint_id: str, blueprint_secret: str, verbose: bool = False
) -> str:
    """Acquire token using agent blueprint client credentials (client_secret flow)."""
    url = TOKEN_URL_TMPL.format(tenant=tenant)
    data = {
        "client_id": blueprint_id,
        "client_secret": blueprint_secret,
        "scope": DEFAULT_SCOPE,
        "grant_type": "client_credentials",
    }
    if verbose:
        click.echo(
            f"[blueprint] Requesting token for blueprint {_redact(blueprint_id)} tenant={_redact(tenant)}"
        )
    resp = requests.post(url, data=data, timeout=30)
    if not resp.ok:
        click.echo(
            f"Blueprint token acquisition failed: {resp.status_code} {resp.text}",
            err=True,
        )
        resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        click.echo(f"No access_token in response: {resp.text}", err=True)
        sys.exit(1)
    click.echo("Token acquired successfully (blueprint)")
    return token


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
        # Prefer service principal if secret is available, otherwise azure-cli
        if client_secret:
            mode = "service-principal"
        else:
            mode = "azure-cli"

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
                "Set ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET, or use --auth-mode azure-cli.",
                err=True,
            )
            sys.exit(1)
        return _acquire_token_service_principal(
            tenant, client_id, client_secret, verbose=verbose
        )

    if mode == "azure-cli":
        return _acquire_token_azure_cli(verbose=verbose)

    click.echo(
        f"Unknown auth mode: {auth_mode} (choices: auto, service-principal, azure-cli)",
        err=True,
    )
    sys.exit(1)


def acquire_token_for_identity(
    blueprint_id: str | None,
    blueprint_secret: str | None,
    auth_mode: str,
    verbose: bool = False,
) -> str:
    """Resolve token for agent identity create/delete.

    Preference:
      1. If blueprint_secret provided (or env BLUEPRINT_CLIENT_SECRET), use blueprint token flow.
      2. Else use normal acquire_token(auth_mode).
    """
    # Check env fallback for blueprint secret
    if not blueprint_secret:
        blueprint_secret = (
            os.getenv("BLUEPRINT_CLIENT_SECRET", "").strip()
            or os.getenv("AGENT_BLUEPRINT_SECRET", "").strip()
        )

    if blueprint_id and blueprint_secret:
        tenant = os.getenv("ARM_TENANT_ID", "").strip()
        if not tenant:
            click.echo("ARM_TENANT_ID required for blueprint token flow", err=True)
            sys.exit(1)
        return _acquire_token_blueprint(
            tenant, blueprint_id, blueprint_secret, verbose=verbose
        )

    # Check ACCESS_TOKEN bypass first
    access_token = os.getenv("ACCESS_TOKEN", "").strip()
    if access_token:
        click.echo("Using ACCESS_TOKEN from environment (skipping token acquisition)")
        return access_token

    # Fall back to service-principal / azure-cli token (must be owner/sponsor of blueprint)
    if blueprint_secret and not blueprint_id:
        click.echo("--blueprint-secret provided without --blueprint-id", err=True)
        sys.exit(1)

    return acquire_token(auth_mode, verbose=verbose)


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


def create_agent_identity(
    token: str,
    display_name: str,
    blueprint_id: str,
    sponsor_id: str | None,
    sponsor_upn: str | None,
    owner_id: str | None,
    owner_upn: str | None,
    verbose: bool = False,
) -> dict:
    """POST /beta/serviceprincipals/Microsoft.Graph.AgentIdentity

    Doc: https://learn.microsoft.com/en-us/entra/agent-id/create-delete-agent-identities?tabs=microsoft-graph-api
    Requires OData-Version: 4.0 header and @odata.type.
    """
    # Resolve UPNs if needed (uses same token)
    if not sponsor_id and sponsor_upn:
        sponsor_id = resolve_user_id(token, sponsor_upn, verbose=verbose)
        click.echo(f"Resolved sponsor UPN {sponsor_upn} -> {sponsor_id}")
    if not owner_id and owner_upn:
        owner_id = resolve_user_id(token, owner_upn, verbose=verbose)
        click.echo(f"Resolved owner UPN {owner_upn} -> {owner_id}")

    if not sponsor_id and not sponsor_upn:
        # Sponsor is typical but doc shows sponsors@odata.bind optional? We'll warn.
        click.echo(
            "Warning: no sponsor provided; blueprint default sponsors may apply. "
            "Provide --sponsor-id or --sponsor-upn to set explicitly.",
            err=True,
        )

    url = f"{GRAPH_BETA}/servicePrincipals/Microsoft.Graph.AgentIdentity"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "OData-Version": "4.0",
    }
    body: dict = {
        "displayName": display_name,
        "agentIdentityBlueprintId": blueprint_id,
    }
    if sponsor_id:
        body["sponsors@odata.bind"] = [
            f"https://graph.microsoft.com/v1.0/users/{sponsor_id}"
        ]
    if owner_id:
        body["owners@odata.bind"] = [
            f"https://graph.microsoft.com/v1.0/users/{owner_id}"
        ]

    if verbose:
        click.echo(f"POST {url}\n{json.dumps(body, indent=2)}")
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if not resp.ok:
        click.echo(
            f"Create agent identity failed: {resp.status_code} {resp.text}", err=True
        )
        resp.raise_for_status()
    data = resp.json()
    click.echo(
        f"Agent identity created: displayName={data.get('displayName')} id={data.get('id')} appId={data.get('appId')}"
    )
    if verbose:
        click.echo(json.dumps(data, indent=2))
    return data


def delete_agent_identity(token: str, identity_id: str, verbose: bool = False) -> None:
    """DELETE /beta/serviceprincipals/{identity_id}"""
    url = f"{GRAPH_BETA}/servicePrincipals/{identity_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "OData-Version": "4.0",
        "Content-Type": "application/json",
    }
    if verbose:
        click.echo(f"DELETE {url}")
    resp = requests.delete(url, headers=headers, timeout=30)
    if not resp.ok:
        # 204 No Content expected on success
        if resp.status_code == 204:
            click.echo(f"Agent identity {identity_id} deleted (204)")
            return
        click.echo(
            f"Delete agent identity failed: {resp.status_code} {resp.text}", err=True
        )
        resp.raise_for_status()
    # Success can be 204 with empty body or 200 with body
    if resp.status_code == 204 or not resp.text:
        click.echo(f"Agent identity {identity_id} deleted")
    else:
        try:
            data = resp.json()
            click.echo(f"Delete response: {json.dumps(data, indent=2)}")
        except (ValueError, requests.exceptions.JSONDecodeError):
            click.echo(
                f"Agent identity {identity_id} deleted (status {resp.status_code})"
            )


def _run_blueprint(
    display_name: str,
    sponsor_id: str | None,
    sponsor_upn: str | None,
    owner_id: str | None,
    owner_upn: str | None,
    credential_display_name: str,
    credential_end_date: str | None,
    skip_principal: bool,
    auth_mode: str,
    dry_run: bool,
    verbose: bool,
) -> None:
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


@click.group()
def cli():
    """Create Agent Identity Blueprints and Agent Identities (local-dev)."""


@cli.command("create-blueprint")
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
            "azure-cli",
        ],
        case_sensitive=False,
    ),
    default="auto",
    show_default=True,
    help="Authentication mode: auto (SP if ARM_CLIENT_SECRET set else azure-cli), service-principal (client credentials), azure-cli (az login)",
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
def create_blueprint_cmd(
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
    """Create an agent identity blueprint."""
    _run_blueprint(
        display_name=display_name,
        sponsor_id=sponsor_id,
        sponsor_upn=sponsor_upn,
        owner_id=owner_id,
        owner_upn=owner_upn,
        credential_display_name=credential_display_name,
        credential_end_date=credential_end_date,
        skip_principal=skip_principal,
        auth_mode=auth_mode,
        dry_run=dry_run,
        verbose=verbose,
    )


@cli.command("create-identity")
@click.option(
    "--blueprint-id",
    required=True,
    help="Agent blueprint appId (client ID) to create identity from",
)
@click.option(
    "--display-name",
    "identity_display_name",
    required=True,
    help="Agent identity displayName",
)
@click.option(
    "--sponsor-id",
    default=None,
    help="Sponsor user objectId (optional but recommended; if omitted uses blueprint defaults)",
)
@click.option(
    "--sponsor-upn",
    default=None,
    help="Sponsor UPN (resolved via Graph if --sponsor-id not set)",
)
@click.option("--owner-id", default=None, help="Owner user objectId (optional)")
@click.option("--owner-upn", default=None, help="Owner UPN (resolved via Graph)")
@click.option(
    "--blueprint-secret",
    default=None,
    help="Blueprint client secret (preferred for local dev). Falls back to BLUEPRINT_CLIENT_SECRET env var. If not set, uses --auth-mode token.",
)
@click.option(
    "--auth-mode",
    type=click.Choice(["auto", "service-principal", "azure-cli"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Auth mode if --blueprint-secret not provided (uses ARM_* or az login).",
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
def create_identity_cmd(
    blueprint_id,
    identity_display_name,
    sponsor_id,
    sponsor_upn,
    owner_id,
    owner_upn,
    blueprint_secret,
    auth_mode,
    dry_run,
    verbose,
):
    """Create an agent identity from a blueprint.

    Requires either --blueprint-secret (plus --blueprint-id) for blueprint
    client_credentials flow, or --auth-mode service-principal/azure-cli where
    the caller is owner/sponsor of the blueprint.
    """
    blueprint_secret = (
        blueprint_secret
        or os.getenv("BLUEPRINT_CLIENT_SECRET", "").strip()
        or os.getenv("AGENT_BLUEPRINT_SECRET", "").strip()
    )

    if dry_run:
        click.echo(
            f"[dry-run] Would acquire token via {'blueprint' if blueprint_secret else auth_mode} "
            f"and POST /beta/servicePrincipals/Microsoft.Graph.AgentIdentity "
            f"displayName={identity_display_name} blueprintId={blueprint_id} sponsor={sponsor_id or sponsor_upn or '<none>'}"
        )
        # Still validate sponsor resolution would happen
        if verbose:
            click.echo(
                f"[dry-run] blueprint_id={blueprint_id} auth_mode={auth_mode} blueprint_secret={'set' if blueprint_secret else 'not set'}"
            )
        return

    token = acquire_token_for_identity(
        blueprint_id, blueprint_secret, auth_mode, verbose=verbose
    )

    # Delegate UPN resolution and POST to helper (also does sponsor check)
    create_agent_identity(
        token,
        display_name=identity_display_name,
        blueprint_id=blueprint_id,
        sponsor_id=sponsor_id,
        sponsor_upn=sponsor_upn,
        owner_id=owner_id,
        owner_upn=owner_upn,
        verbose=verbose,
    )


@cli.command("delete-identity")
@click.option(
    "--identity-id",
    required=True,
    help="Object ID of the agent identity service principal to delete",
)
@click.option(
    "--blueprint-id",
    default=None,
    help="Agent blueprint appId (needed only if using --blueprint-secret flow)",
)
@click.option(
    "--blueprint-secret",
    default=None,
    help="Blueprint client secret (if provided, uses blueprint token; else uses --auth-mode). Falls back to BLUEPRINT_CLIENT_SECRET env var.",
)
@click.option(
    "--auth-mode",
    type=click.Choice(["auto", "service-principal", "azure-cli"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Auth mode if --blueprint-secret not provided.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without calling Graph",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose logging",
)
def delete_identity_cmd(
    identity_id, blueprint_id, blueprint_secret, auth_mode, dry_run, verbose
):
    """Delete an agent identity (DELETE /beta/servicePrincipals/{id})."""
    blueprint_secret = (
        blueprint_secret
        or os.getenv("BLUEPRINT_CLIENT_SECRET", "").strip()
        or os.getenv("AGENT_BLUEPRINT_SECRET", "").strip()
    )

    if dry_run:
        click.echo(
            f"[dry-run] Would acquire token via {'blueprint' if blueprint_secret else auth_mode} "
            f"and DELETE /beta/servicePrincipals/{identity_id}"
        )
        return

    # For delete, blueprint_id is optional if using auth-mode token; but required if using blueprint_secret
    if blueprint_secret and not blueprint_id:
        click.echo(
            "--blueprint-secret requires --blueprint-id for blueprint token flow; "
            "either provide both or use --auth-mode without blueprint secret",
            err=True,
        )
        sys.exit(1)

    # If blueprint_secret not set, blueprint_id may be None - acquire_token_for_identity handles fallback
    token = acquire_token_for_identity(
        blueprint_id, blueprint_secret, auth_mode, verbose=verbose
    )
    delete_agent_identity(token, identity_id, verbose=verbose)


if __name__ == "__main__":
    cli()

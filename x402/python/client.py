#!/usr/bin/env python3
"""
x402 Python client — demonstrates the full payment flow end-to-end.

Flow per request:
  1. GET /endpoint                    → 402 + PAYMENT-REQUIRED header
  2. Parse header, sign EIP-3009      (pure Python, no MetaMask)
  3. GET /endpoint + PAYMENT-SIGNATURE → 200 OK + resource

Run inside Docker:  docker compose run client
Run locally:       SERVER_URL=http://localhost:8080 python client.py
"""

import json
import os
import sys
import time
from typing import Optional

import httpx
from eth_account import Account

from x402 import b64dec, sign_transfer_with_authorization

# ─── Config ───────────────────────────────────────────────────────────────────

# Anvil account #0 — 1 000 000 USDC pre-minted at deployment
BUYER_KEY  = os.environ.get(
    "BUYER_PRIVATE_KEY",
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
)
SERVER_URL = os.environ.get("SERVER_URL", "http://server:8080")

buyer = Account.from_key(BUYER_KEY)


# ─── Demo runner ──────────────────────────────────────────────────────────────

def _wait_for_server(url: str, retries: int = 60) -> None:
    for i in range(retries):
        try:
            httpx.get(f"{url}/", timeout=2)
            return
        except Exception:
            if i < 3:
                print(f"  ⏳  Waiting for server at {url} …")
            time.sleep(2)
    print("  ❌  Server did not come up in time.")
    sys.exit(1)


def call_endpoint(path: str) -> None:
    url = f"{SERVER_URL}{path}"
    sep = "─" * 60

    print(f"\n{sep}")
    print(f"  GET {url}")
    print(sep)

    with httpx.Client(timeout=30) as client:
        # ── Step 1: first request, expecting 402 ─────────────────────────────
        r1 = client.get(url)
        print(f"  → {r1.status_code} {r1.reason_phrase}")

        if r1.status_code != 402:
            print(f"  Unexpected status. Body: {r1.text[:200]}")
            return

        pr_b64 = r1.headers.get("payment-required") or r1.headers.get("PAYMENT-REQUIRED")
        if not pr_b64:
            print("  ❌  No PAYMENT-REQUIRED header in 402 response.")
            return

        pr = b64dec(pr_b64)
        amount_usdc = int(pr["maxAmountRequired"]) / 1e6
        print(f"  → Server requests  {amount_usdc:.4f} USDC")
        print(f"     network : {pr['network']}")
        print(f"     pay to  : {pr['payTo']}")
        print(f"     asset   : {pr['asset']['address']}")

        # ── Step 2: sign EIP-712 TransferWithAuthorization ───────────────────
        print(f"  ✍   Signing EIP-712 TransferWithAuthorization …")
        sig_b64 = sign_transfer_with_authorization(BUYER_KEY, payment_required_b64=pr_b64)

        # ── Step 3: retry with PAYMENT-SIGNATURE ─────────────────────────────
        r2 = client.get(url, headers={"PAYMENT-SIGNATURE": sig_b64})
        print(f"  → {r2.status_code} {r2.reason_phrase}")

        if r2.status_code == 200:
            tx_hash = r2.headers.get("x-payment-txhash", "n/a")
            block   = r2.headers.get("x-payment-block",  "n/a")
            print(f"  ✅  Payment settled!")
            print(f"     tx    : {tx_hash}")
            print(f"     block : {block}")
            print(f"  Resource data:")
            print(json.dumps(r2.json(), indent=4))
        else:
            print(f"  ❌  Failed. Body: {r2.text[:400]}")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           x402 Payment Protocol — Python Demo           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Buyer wallet : {buyer.address}")
    print(f"  Server URL   : {SERVER_URL}")

    _wait_for_server(SERVER_URL)

    for path in ["/api/quote", "/api/portfolio", "/api/forecast"]:
        call_endpoint(path)
        time.sleep(1)

    print(f"\n{'─' * 60}")
    print("  Demo complete — all three endpoints paid and served.")
    print()


if __name__ == "__main__":
    main()

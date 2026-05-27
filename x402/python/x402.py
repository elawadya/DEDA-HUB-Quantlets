"""
Shared x402 utilities: EIP-712 / EIP-3009 signing and header encoding.

Used by both server.py (decoding) and client.py (signing).
"""

import base64
import json
import os
import secrets
import time
from typing import Any

from eth_account import Account
from eth_utils import to_checksum_address


# ─── Encoding helpers ─────────────────────────────────────────────────────────

def b64enc(obj: Any) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()

def b64dec(s: str) -> Any:
    return json.loads(base64.b64decode(s))


# ─── PAYMENT-REQUIRED header builder ─────────────────────────────────────────

def make_payment_required(
    *,
    price_units: int,        # USDC amount in micro-units (6 decimals)
    usdc_address: str,
    pay_to: str,             # server wallet address
    resource_url: str,
    chain_id: int = 8453,
) -> str:
    """Return base64-encoded PAYMENT-REQUIRED header value (x402 v2 spec)."""
    payload = {
        "version": 2,
        "scheme": "exact",
        "network": f"eip155:{chain_id}",
        "maxAmountRequired": str(price_units),
        "asset": {
            "address": to_checksum_address(usdc_address),
            "decimals": 6,
            "eip712Domain": {
                "name": "USD Coin",
                "version": "2",
                "chainId": chain_id,
                "verifyingContract": to_checksum_address(usdc_address),
            },
        },
        "payTo": to_checksum_address(pay_to),
        "resource": resource_url,
    }
    return b64enc(payload)


# ─── EIP-712 / EIP-3009 signer ────────────────────────────────────────────────

def sign_transfer_with_authorization(
    private_key: str,
    *,
    payment_required_b64: str,
    valid_seconds: int = 3600,
) -> str:
    """
    Sign a TransferWithAuthorization (EIP-3009) for the given PAYMENT-REQUIRED
    header and return a base64-encoded PAYMENT-SIGNATURE header value.

    This is pure Python — no browser, no MetaMask.
    """
    pr = b64dec(payment_required_b64)
    domain = pr["asset"]["eip712Domain"]
    amount = int(pr["maxAmountRequired"])
    to     = pr["payTo"]

    account    = Account.from_key(private_key)
    now        = int(time.time())
    valid_after  = now - 60          # allow slight clock skew
    valid_before = now + valid_seconds
    nonce_hex    = "0x" + secrets.token_hex(32)

    # Account.sign_typed_data handles EIP-712 encoding end-to-end
    signed = Account.sign_typed_data(
        private_key=private_key,
        domain_data={
            "name":              domain["name"],
            "version":           domain["version"],
            "chainId":           int(domain["chainId"]),
            "verifyingContract": domain["verifyingContract"],
        },
        message_types={
            "TransferWithAuthorization": [
                {"name": "from",        "type": "address"},
                {"name": "to",          "type": "address"},
                {"name": "value",       "type": "uint256"},
                {"name": "validAfter",  "type": "uint256"},
                {"name": "validBefore", "type": "uint256"},
                {"name": "nonce",       "type": "bytes32"},
            ]
        },
        message_data={
            "from":        account.address,
            "to":          to_checksum_address(to),
            "value":       amount,
            "validAfter":  valid_after,
            "validBefore": valid_before,
            "nonce":       bytes.fromhex(nonce_hex[2:]),
        },
    )

    payload = {
        "version": pr["version"],
        "scheme":  pr["scheme"],
        "network": pr["network"],
        "payload": {
            "from":        account.address,
            "to":          to_checksum_address(to),
            "value":       str(amount),
            "validAfter":  str(valid_after),
            "validBefore": str(valid_before),
            "nonce":       nonce_hex,
            "v":           signed.v,
            "r":           "0x" + signed.r.to_bytes(32, "big").hex(),
            "s":           "0x" + signed.s.to_bytes(32, "big").hex(),
        },
    }
    return b64enc(payload)

#!/usr/bin/env python3
"""
x402 Facilitator — verifies EIP-3009 signatures and settles payments on-chain.

Role in the protocol:
  Client → signed PAYMENT-SIGNATURE → Server → POST /verify → Facilitator
  Facilitator submits transferWithAuthorization to the blockchain and returns the tx receipt.

The facilitator cannot steal: the signed struct specifies exact from/to/value.
"""

import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from web3 import Web3
from eth_account import Account
from eth_utils import to_checksum_address
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("facilitator")

app = FastAPI(title="x402 Facilitator", version="2.0")

RPC_URL = os.environ.get("RPC_URL", "http://blockchain:8545")
# Anvil account #3 — funded with ETH to cover gas fees
FACILITATOR_KEY = os.environ.get(
    "FACILITATOR_PRIVATE_KEY",
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
)

w3 = Web3(Web3.HTTPProvider(RPC_URL))
facilitator = Account.from_key(FACILITATOR_KEY)

# Minimal ABI: only the functions we call
USDC_ABI = [
    {
        "inputs": [
            {"name": "from",        "type": "address"},
            {"name": "to",          "type": "address"},
            {"name": "value",       "type": "uint256"},
            {"name": "validAfter",  "type": "uint256"},
            {"name": "validBefore", "type": "uint256"},
            {"name": "nonce",       "type": "bytes32"},
            {"name": "v",           "type": "uint8"},
            {"name": "r",           "type": "bytes32"},
            {"name": "s",           "type": "bytes32"},
        ],
        "name": "transferWithAuthorization",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs":  [{"name": "account", "type": "address"}],
        "name":    "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


# ─── Pydantic models ───────────────────────────────────────────────────────────

class PaymentPayload(BaseModel):
    from_address: str
    to_address:   str
    value:        int
    valid_after:  int
    valid_before: int
    nonce:        str   # "0x" + 32 bytes hex
    v:            int
    r:            str   # "0x" + 32 bytes hex
    s:            str   # "0x" + 32 bytes hex


class VerifyRequest(BaseModel):
    payload:      PaymentPayload
    usdc_address: str
    network:      str = "eip155:8453"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_bytes32(h: str) -> bytes:
    raw = bytes.fromhex(h.removeprefix("0x"))
    if len(raw) < 32:
        raw = raw.rjust(32, b"\x00")
    return raw[:32]


def _wait_for_rpc(retries: int = 60) -> None:
    for _ in range(retries):
        try:
            w3.eth.block_number
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Could not connect to RPC at {RPC_URL}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    _wait_for_rpc()
    eth_bal = w3.eth.get_balance(facilitator.address) / 1e18
    log.info("Facilitator wallet : %s", facilitator.address)
    log.info("ETH balance        : %.4f ETH", eth_bal)
    log.info("Chain ID           : %d", w3.eth.chain_id)


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "block":       w3.eth.block_number,
        "facilitator": facilitator.address,
        "chain_id":    w3.eth.chain_id,
    }


@app.post("/verify")
async def verify_and_settle(req: VerifyRequest):
    """
    Verify an EIP-3009 authorization and submit transferWithAuthorization on-chain.
    Returns the transaction receipt on success.
    """
    usdc = w3.eth.contract(
        address=to_checksum_address(req.usdc_address),
        abi=USDC_ABI,
    )
    p = req.payload

    log.info(
        "Verifying payment: from=%s  to=%s  value=%d",
        p.from_address[:10], p.to_address[:10], p.value,
    )

    try:
        tx = usdc.functions.transferWithAuthorization(
            to_checksum_address(p.from_address),
            to_checksum_address(p.to_address),
            p.value,
            p.valid_after,
            p.valid_before,
            _hex_to_bytes32(p.nonce),
            p.v,
            _hex_to_bytes32(p.r),
            _hex_to_bytes32(p.s),
        ).build_transaction({
            "from":     facilitator.address,
            "nonce":    w3.eth.get_transaction_count(facilitator.address),
            "gas":      150_000,
            "gasPrice": w3.eth.gas_price,
        })

        signed  = facilitator.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

        log.info("Settled  tx=%s  block=%d  status=%d",
                 receipt.transactionHash.hex(), receipt.blockNumber, receipt.status)

        return {
            "success":    receipt.status == 1,
            "tx_hash":    receipt.transactionHash.hex(),
            "block":      receipt.blockNumber,
            "gas_used":   receipt.gasUsed,
        }

    except Exception as exc:
        log.error("Settlement failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")

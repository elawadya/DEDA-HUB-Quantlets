#!/usr/bin/env python3
"""
x402 Resource Server — three paywalled FastAPI endpoints.

Middleware pattern:
  ① Client hits endpoint without payment → 402 + PAYMENT-REQUIRED header
  ② Client signs EIP-3009 and retries    → server calls facilitator /verify
  ③ Facilitator settles on-chain          → server returns 200 + resource
"""

import json
import os
import random
import time
from functools import wraps

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from x402 import b64dec, make_payment_required

# ─── Config ───────────────────────────────────────────────────────────────────

FACILITATOR_URL = os.environ.get("FACILITATOR_URL", "http://facilitator:8090")
SERVER_WALLET   = os.environ.get(
    "SERVER_WALLET",
    "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # Anvil account #2
)
CHAIN_ID = int(os.environ.get("CHAIN_ID", "8453"))


def _usdc_address() -> str:
    path = "/shared/usdc_address.txt"
    deadline = time.time() + 120
    while time.time() < deadline:
        if os.path.exists(path):
            addr = open(path).read().strip()
            if addr:
                return addr
        time.sleep(1)
    raise RuntimeError("USDC address not available at /shared/usdc_address.txt")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="x402 Resource Server", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── x402 gate decorator ──────────────────────────────────────────────────────

def x402_gate(price_units: int):
    """
    Decorator that gates a FastAPI endpoint behind an x402 payment.

    @x402_gate(price_units=1_000)   # 0.001 USDC
    async def my_endpoint(request: Request): ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            usdc = _usdc_address()

            # ① No payment header → return 402
            sig_b64 = (
                request.headers.get("payment-signature")
                or request.headers.get("PAYMENT-SIGNATURE")
            )
            if not sig_b64:
                pr = make_payment_required(
                    price_units=price_units,
                    usdc_address=usdc,
                    pay_to=SERVER_WALLET,
                    resource_url=str(request.url),
                    chain_id=CHAIN_ID,
                )
                return Response(
                    status_code=402,
                    headers={"PAYMENT-REQUIRED": pr},
                    content=json.dumps({
                        "error":   "Payment required",
                        "price":   f"{price_units / 1e6:.4f} USDC",
                        "network": f"eip155:{CHAIN_ID}",
                    }),
                    media_type="application/json",
                )

            # ② Has payment → forward to facilitator for verification
            try:
                sig_data = b64dec(sig_b64)
                payload  = sig_data["payload"]
            except Exception:
                return Response(status_code=400, content="Malformed PAYMENT-SIGNATURE")

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{FACILITATOR_URL}/verify",
                    json={
                        "payload": {
                            "from_address": payload["from"],
                            "to_address":   payload["to"],
                            "value":        int(payload["value"]),
                            "valid_after":  int(payload["validAfter"]),
                            "valid_before": int(payload["validBefore"]),
                            "nonce":        payload["nonce"],
                            "v":            payload["v"],
                            "r":            payload["r"],
                            "s":            payload["s"],
                        },
                        "usdc_address": usdc,
                        "network":      sig_data.get("network", f"eip155:{CHAIN_ID}"),
                    },
                )

            if resp.status_code != 200 or not resp.json().get("success"):
                pr = make_payment_required(
                    price_units=price_units,
                    usdc_address=usdc,
                    pay_to=SERVER_WALLET,
                    resource_url=str(request.url),
                    chain_id=CHAIN_ID,
                )
                return Response(
                    status_code=402,
                    headers={"PAYMENT-REQUIRED": pr},
                    content=json.dumps({
                        "error":  "Payment verification failed",
                        "detail": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
                    }),
                    media_type="application/json",
                )

            # ③ Settled — serve the resource
            result = await func(request, *args, **kwargs)
            result.headers["X-Payment-TxHash"] = resp.json().get("tx_hash", "")
            result.headers["X-Payment-Block"]  = str(resp.json().get("block", ""))
            return result

        return wrapper
    return decorator


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "x402 Resource Server",
        "version": "2.0",
        "endpoints": {
            "/api/quote":     {"price_usdc": 0.001, "description": "Spot prices for BTC/ETH/SOL"},
            "/api/portfolio": {"price_usdc": 0.005, "description": "Portfolio risk analytics"},
            "/api/forecast":  {"price_usdc": 0.010, "description": "7-day ML price forecast"},
        },
    }


@app.get("/api/quote")
@x402_gate(price_units=1_000)          # 0.001 USDC
async def quote(request: Request):
    data = {
        "timestamp": int(time.time()),
        "prices_usd": {
            "BTC":  round(random.uniform(55_000, 70_000), 2),
            "ETH":  round(random.uniform(2_800,   4_200), 2),
            "SOL":  round(random.uniform(130,       200), 2),
            "USDC": 1.00,
        },
        "source": "x402-demo",
    }
    return Response(content=json.dumps(data), media_type="application/json")


@app.get("/api/portfolio")
@x402_gate(price_units=5_000)          # 0.005 USDC
async def portfolio(request: Request):
    data = {
        "timestamp": int(time.time()),
        "portfolio": {
            "assets":           ["BTC", "ETH", "SOL"],
            "weights":          [0.50,  0.30,  0.20],
            "sharpe_ratio":     round(random.uniform(1.5, 3.5), 3),
            "max_drawdown":     round(random.uniform(-0.25, -0.05), 3),
            "expected_annual_return": round(random.uniform(0.10, 0.50), 3),
            "annual_volatility":      round(random.uniform(0.20, 0.60), 3),
        },
    }
    return Response(content=json.dumps(data), media_type="application/json")


@app.get("/api/forecast")
@x402_gate(price_units=10_000)         # 0.010 USDC
async def forecast(request: Request):
    base   = random.uniform(55_000, 68_000)
    prices = [round(base * (1 + random.uniform(-0.04, 0.04)), 2) for _ in range(7)]
    data = {
        "timestamp":    int(time.time()),
        "asset":        "BTC",
        "model":        "LSTM-Transformer",
        "horizon_days": 7,
        "forecast_usd": prices,
        "confidence_95": [
            [round(p * 0.92, 2), round(p * 1.08, 2)] for p in prices
        ],
    }
    return Response(content=json.dumps(data), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

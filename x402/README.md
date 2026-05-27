# x402 Payment Protocol — Local Demo

> **HTTP 402: the dormant status code that finally woke up.**  
> Wolfgang Karl Häerdle · Humboldt-Universität zu Berlin · DEDA · May 2026

A fully self-contained, seven-service Docker stack demonstrating the x402 payment protocol: machine-native USDC micropayments over plain HTTP, no accounts, no credit cards, no settlement delay.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     x402 Protocol Flow                           │
│                                                                  │
│  Client          Resource Server       Facilitator    Blockchain │
│  (buyer)         (FastAPI)             (settler)      (Anvil)   │
│                                                                  │
│  GET /api/quote ────────────────►                               │
│                  ◄── 402 + PAYMENT-REQUIRED header ──           │
│  [sign EIP-712 TransferWithAuthorization]                        │
│  GET + PAYMENT-SIGNATURE ────────►                              │
│                  POST /verify ─────────────────────►            │
│                               transferWithAuthorization ────►   │
│                               ◄─────────────────── receipt     │
│                  ◄── {success: true} ──────────────             │
│  ◄── 200 OK + resource ──────────                               │
└──────────────────────────────────────────────────────────────────┘
```

| Service          | Port | Role |
|------------------|------|------|
| `blockchain`     | 8545 | Anvil EVM node — chain-id 8453, 2-second blocks, 5 funded wallets |
| `blockchain-init`| —    | Compiles & deploys MockUSDC (EIP-3009), writes address to shared volume |
| `facilitator`    | 8090 | Verifies EIP-3009 signatures, submits `transferWithAuthorization` on-chain |
| `server`         | 8080 | FastAPI with `@x402_gate` decorator — three paywalled endpoints |
| `client`         | —    | Pure-Python demo buyer: signs & pays autonomously (run on demand) |
| `dashboard`      | 8501 | Streamlit live dashboard — balances, events, protocol flow diagram |

---

## Quick Start

```bash
# 1. Start the stack (blockchain + facilitator + server + dashboard)
docker compose up --build

# 2. In a second terminal — run the demo buyer
docker compose run client
```

Open **http://localhost:8501** to watch the dashboard update in real time.

---

## Paywalled Endpoints

| Endpoint         | Price     | Description |
|------------------|-----------|-------------|
| `GET /api/quote`     | 0.001 USDC | Spot prices for BTC / ETH / SOL |
| `GET /api/portfolio` | 0.005 USDC | Portfolio risk analytics |
| `GET /api/forecast`  | 0.010 USDC | 7-day ML price forecast |

Try manually:
```bash
# Step 1 — get the 402 with payment requirements
curl -i http://localhost:8080/api/quote

# Step 2 — the client handles signing automatically
docker compose run client
```

---

## How It Works

### MockUSDC (EIP-3009)

`blockchain/src/MockUSDC.sol` — minimal ERC-20 + `transferWithAuthorization`:

```solidity
function transferWithAuthorization(
    address from, address to, uint256 value,
    uint256 validAfter, uint256 validBefore,
    bytes32 nonce,
    uint8 v, bytes32 r, bytes32 s
) external { ... }
```

The buyer signs off-chain; the facilitator submits and pays gas. Buyer pays zero gas.

### @x402_gate decorator

```python
@app.get("/api/quote")
@x402_gate(price_units=1_000)   # 0.001 USDC
async def quote(request: Request):
    ...  # only reached after payment is settled
```

### Pure-Python EIP-712 Signing

```python
signed = account.sign_typed_data(
    domain_data   = { "name": "USD Coin", "version": "2", "chainId": 8453, ... },
    message_types = { "TransferWithAuthorization": [...] },
    message_data  = { "from": buyer, "to": server, "value": 1000, ... },
)
```

No browser. No MetaMask. Just a private key.

---

## Cryptographic Primitives

- **EIP-712** — typed structured data signing (human-readable, wallet-safe)
- **EIP-3009** — `transferWithAuthorization`: signed, delegated, gasless ERC-20 transfer
- **USDC** (Mock) — stablecoin with native EIP-3009 support (Circle's production USDC does the same on Base)

---

## Payment Schemes (x402 v2)

| Scheme         | Status      | Use case |
|----------------|-------------|----------|
| `exact`        | Production  | Fixed price per request — articles, API metering |
| `upto`         | Proposed    | Pay actual consumed amount ≤ cap — LLM tokens |
| `batch`        | Proposed    | One signature for N requests — agent workflows |
| `subscription` | Draft       | Fixed time window — bridge to SaaS billing |

This demo implements **`exact`** — the only scheme in production (165M+ transactions on Base).

---

## Test Wallets (Anvil defaults)

| Account | Address | Role | USDC |
|---------|---------|------|------|
| 0 | `0xf39F…2266` | Buyer (client) | 1 000 000 |
| 1 | `0x7099…C8`   | Deployer       | 1 000 000 |
| 2 | `0x3C44…3BC`  | Server wallet  | 1 000 000 |
| 3 | `0x90F7…906`  | Facilitator    | 1 000 000 |
| 4 | `0x15d3…65`   | Spare          | 1 000 000 |

---

## References

- [x402 Specification](https://github.com/coinbase/x402) — Apache 2.0, Coinbase
- [EIP-3009](https://eips.ethereum.org/EIPS/eip-3009) — Transfer With Authorization
- [EIP-712](https://eips.ethereum.org/EIPS/eip-712) — Typed Structured Data Hashing and Signing
- [CAIP-2](https://github.com/ChainAgnostic/CAIPs/blob/main/CAIPs/caip-2.md) — Blockchain ID Specification
- [Foundry / Anvil](https://book.getfoundry.sh/) — Local EVM node

"""
x402 Demo Dashboard — live Streamlit view of the local stack.

Shows:
  • Chain status  (latest block, chain-id)
  • Wallet balances of buyer / server / facilitator
  • Recent x402 payment transactions (AuthorizationUsed events)
  • Live protocol flow diagram
"""

import os
import time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from web3 import Web3

# ─── Config ───────────────────────────────────────────────────────────────────

RPC_URL         = os.environ.get("RPC_URL",         "http://blockchain:8545")
FACILITATOR_URL = os.environ.get("FACILITATOR_URL", "http://facilitator:8090")
SERVER_URL      = os.environ.get("SERVER_URL",       "http://server:8080")

WALLET_LABELS = {
    "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266": "Buyer  (account 0)",
    "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC": "Server (account 2)",
    "0x90F79bf6EB2c4f870365E785982E1f101E93b906": "Facilitator (account 3)",
}

AUTHORIZATION_USED_TOPIC = Web3.keccak(
    text="AuthorizationUsed(address,bytes32)"
).hex()

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_web3() -> Web3:
    return Web3(Web3.HTTPProvider(RPC_URL))

def load_usdc_address() -> str | None:
    path = "/shared/usdc_address.txt"
    if os.path.exists(path):
        return open(path).read().strip() or None
    return None

@st.cache_data(ttl=3)
def fetch_chain_info():
    w3 = get_web3()
    try:
        return {
            "block":    w3.eth.block_number,
            "chain_id": w3.eth.chain_id,
            "connected": True,
        }
    except Exception:
        return {"block": "—", "chain_id": "—", "connected": False}

@st.cache_data(ttl=3)
def fetch_balances(usdc_address: str) -> dict:
    w3 = get_web3()
    abi = [
        {"inputs": [{"name": "a", "type": "address"}],
         "name": "balanceOf", "outputs": [{"type": "uint256"}],
         "stateMutability": "view", "type": "function"}
    ]
    try:
        usdc = w3.eth.contract(address=Web3.to_checksum_address(usdc_address), abi=abi)
        return {
            label: usdc.functions.balanceOf(addr).call() / 1e6
            for addr, label in WALLET_LABELS.items()
        }
    except Exception as e:
        return {label: 0 for label in WALLET_LABELS.values()}

@st.cache_data(ttl=3)
def fetch_events(usdc_address: str, from_block: int = 0) -> list[dict]:
    w3 = get_web3()
    try:
        logs = w3.eth.get_logs({
            "address":   Web3.to_checksum_address(usdc_address),
            "fromBlock": max(0, from_block),
            "topics":    [AUTHORIZATION_USED_TOPIC],
        })
        events = []
        for log in logs[-20:]:  # last 20
            events.append({
                "block":      log["blockNumber"],
                "tx":         log["transactionHash"].hex()[:18] + "…",
                "authorizer": "0x" + log["topics"][1].hex()[-40:],
            })
        return events[::-1]  # newest first
    except Exception:
        return []

@st.cache_data(ttl=5)
def facilitator_health() -> dict:
    try:
        r = httpx.get(f"{FACILITATOR_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable"}

# ─── Layout ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="x402 Demo Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ x402 Payment Protocol — Live Demo Dashboard")
st.caption("Local seven-service stack • Base (chain-id 8453) • USDC (EIP-3009)")

usdc_addr = load_usdc_address()

# Row 1: chain + facilitator status
col1, col2, col3, col4 = st.columns(4)
chain = fetch_chain_info()
fac   = facilitator_health()

col1.metric("Chain", f"eip155:{chain['chain_id']}", delta="Base (local)")
col2.metric("Latest Block", chain["block"])
col3.metric("Facilitator", fac.get("status", "—").upper())
col4.metric("USDC Contract", (usdc_addr[:10] + "…") if usdc_addr else "deploying…")

st.divider()

# Row 2: balances
if usdc_addr:
    st.subheader("💰 USDC Balances")
    balances = fetch_balances(usdc_addr)
    bcols = st.columns(len(balances))
    for i, (label, bal) in enumerate(balances.items()):
        bcols[i].metric(label, f"{bal:,.4f} USDC")

    # Row 3: recent events
    st.subheader("🔗 Recent AuthorizationUsed Events")
    events = fetch_events(usdc_addr)
    if events:
        st.dataframe(
            pd.DataFrame(events),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No payments yet — run `docker compose run client` to trigger the demo.")
else:
    st.warning("⏳ Waiting for MockUSDC to be deployed … (run `docker compose up` first)")

# Row 4: protocol flow diagram
st.divider()
st.subheader("🔄 x402 Protocol Flow")

fig = go.Figure()
nodes = ["Client\n(buyer)", "Resource\nServer", "Facilitator\n(Coinbase CDP)", "Blockchain\n(Base)"]
x_pos = [0, 1, 2, 3]
y_pos = [0, 0, 0, 0]

# Nodes
fig.add_trace(go.Scatter(
    x=x_pos, y=y_pos,
    mode="markers+text",
    marker=dict(size=60, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"]),
    text=nodes,
    textposition="bottom center",
    hoverinfo="none",
))

# Arrows (annotated)
arrows = [
    (0, 1, "① GET /resource"),
    (1, 0, "② 402 + PAYMENT-REQUIRED"),
    (0, 1, "③ GET + PAYMENT-SIGNATURE"),
    (1, 2, "④ POST /verify"),
    (2, 3, "⑤ transferWithAuthorization"),
    (3, 2, "⑥ receipt"),
    (2, 1, "⑦ {success: true}"),
    (1, 0, "⑧ 200 OK + resource"),
]
for i, (src, dst, label) in enumerate(arrows):
    yoff = 0.08 * (1 if i % 2 == 0 else -1)
    fig.add_annotation(
        x=x_pos[dst], y=y_pos[dst] + yoff,
        ax=x_pos[src], ay=y_pos[src] + yoff,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowcolor="#888",
        text=label, font=dict(size=10),
        bgcolor="white", opacity=0.85,
    )

fig.update_layout(
    height=320,
    xaxis=dict(range=[-0.5, 3.5], visible=False),
    yaxis=dict(range=[-0.5, 0.5], visible=False),
    showlegend=False,
    margin=dict(t=20, b=60, l=20, r=20),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

# Auto-refresh
time.sleep(0.1)
st.button("🔄 Refresh")

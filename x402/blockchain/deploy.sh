#!/usr/bin/env bash
# Deploy MockUSDC to the local Anvil node and write address to /shared/
set -euo pipefail

RPC_URL="${RPC_URL:-http://blockchain:8545}"
# Anvil account #1 — used only for deployment
DEPLOY_KEY="0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

echo "⏳  Waiting for Anvil at $RPC_URL …"
until cast chain-id --rpc-url "$RPC_URL" >/dev/null 2>&1; do sleep 1; done
echo "✅  Anvil ready  (chain-id: $(cast chain-id --rpc-url "$RPC_URL"))"

echo "🔨  Compiling & deploying MockUSDC …"
OUTPUT=$(forge create /app/src/MockUSDC.sol:MockUSDC \
    --rpc-url   "$RPC_URL" \
    --private-key "$DEPLOY_KEY" \
    --broadcast \
    --json 2>&1 | tail -1)

ADDRESS=$(echo "$OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['deployedTo'])")
mkdir -p /shared
echo "$ADDRESS" > /shared/usdc_address.txt
echo "✅  MockUSDC deployed at $ADDRESS"

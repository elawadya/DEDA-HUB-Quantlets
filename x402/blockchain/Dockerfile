FROM ghcr.io/foundry-rs/foundry:latest

EXPOSE 8545

# 2-second blocks, chain-id 8453 (Base), 5 pre-funded test wallets, silent output
CMD ["anvil", \
     "--host", "0.0.0.0", \
     "--chain-id", "8453", \
     "--block-time", "2", \
     "--accounts", "5", \
     "--silent"]

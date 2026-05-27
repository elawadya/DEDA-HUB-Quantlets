// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice Minimal ERC-20 + EIP-3009 (transferWithAuthorization) for local demo.
///         Mints 1 000 000 USDC (6 decimals) to the first 5 Anvil test wallets.
contract MockUSDC {
    string  public constant name     = "USD Coin";
    string  public constant symbol   = "USDC";
    uint8   public constant decimals = 6;

    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => mapping(bytes32 => bool))    public authorizationState;

    bytes32 public immutable DOMAIN_SEPARATOR;

    bytes32 public constant TRANSFER_WITH_AUTHORIZATION_TYPEHASH = keccak256(
        "TransferWithAuthorization(address from,address to,uint256 value,"
        "uint256 validAfter,uint256 validBefore,bytes32 nonce)"
    );

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event AuthorizationUsed(address indexed authorizer, bytes32 indexed nonce);

    constructor() {
        DOMAIN_SEPARATOR = keccak256(abi.encode(
            keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
            keccak256(bytes(name)),
            keccak256(bytes("2")),
            block.chainid,
            address(this)
        ));

        // Anvil test wallets (mnemonic: "test test test … junk")
        _mint(0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266, 1_000_000 * 1e6); // account 0 — buyer
        _mint(0x70997970C51812dc3A010C7d01b50e0d17dc79C8, 1_000_000 * 1e6); // account 1
        _mint(0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC, 1_000_000 * 1e6); // account 2 — server
        _mint(0x90F79bf6EB2c4f870365E785982E1f101E93b906, 1_000_000 * 1e6); // account 3 — facilitator
        _mint(0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65, 1_000_000 * 1e6); // account 4
    }

    // ─── ERC-20 ────────────────────────────────────────────────────────────────

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to]         += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        allowance[from][msg.sender] -= amount;
        balanceOf[from]             -= amount;
        balanceOf[to]               += amount;
        emit Transfer(from, to, amount);
        return true;
    }

    // ─── EIP-3009 ──────────────────────────────────────────────────────────────

    function transferWithAuthorization(
        address from,
        address to,
        uint256 value,
        uint256 validAfter,
        uint256 validBefore,
        bytes32 nonce,
        uint8   v,
        bytes32 r,
        bytes32 s
    ) external {
        require(block.timestamp >  validAfter,  "Auth: not yet valid");
        require(block.timestamp <  validBefore,  "Auth: expired");
        require(!authorizationState[from][nonce], "Auth: nonce used");

        bytes32 digest = keccak256(abi.encodePacked(
            "\x19\x01",
            DOMAIN_SEPARATOR,
            keccak256(abi.encode(
                TRANSFER_WITH_AUTHORIZATION_TYPEHASH,
                from, to, value, validAfter, validBefore, nonce
            ))
        ));

        require(ecrecover(digest, v, r, s) == from, "Auth: invalid signature");

        authorizationState[from][nonce] = true;
        emit AuthorizationUsed(from, nonce);

        balanceOf[from] -= value;
        balanceOf[to]   += value;
        emit Transfer(from, to, value);
    }

    // ─── internal ──────────────────────────────────────────────────────────────

    function _mint(address to, uint256 amount) internal {
        totalSupply  += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }
}

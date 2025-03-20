#!/bin/bash

# Install Rust via rustup (non-interactive)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ensure the new cargo is on PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Optionally print versions
rustc --version
cargo --version

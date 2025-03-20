#!/usr/bin/env bash
set -e  # Stop if anything fails

# Install Rust non-interactively via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Cargo (Rust’s package manager) to PATH for the current shell
export PATH="$HOME/.cargo/bin:$PATH"

# Print the installed versions (for debugging)
rustc --version
cargo --version

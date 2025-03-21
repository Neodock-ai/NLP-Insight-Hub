#!/bin/bash
# Install the latest stable Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Ensure Cargo’s bin directory is on PATH
export PATH="$HOME/.cargo/bin:$PATH"

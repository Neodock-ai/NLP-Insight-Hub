#!/bin/bash
# Install the latest stable Rust toolchain
curl --proto 'https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Ensure Cargo's bin directory is on PATH
export PATH="$HOME/.cargo/bin:$PATH"
# Install Python dependencies with the no-build-isolation flag
pip install -r requirements.txt --no-build-isolation

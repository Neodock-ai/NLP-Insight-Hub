#!/bin/bash
# Install rustup (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Ensure the cargo bin directory is on PATH
source $HOME/.cargo/env
# Update to the latest stable version
rustup update stable

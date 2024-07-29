### Noah dockerfile
# FROM mcr.microsoft.com/devcontainers/rust:0-1-bullseye

# # Include lld linker to improve build times either by using environment variable
# # RUSTFLAGS="-C link-arg=-fuse-ld=lld" or with Cargo's configuration file (i.e see .cargo/config.toml).
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get -y install clang lld \
#    && apt-get autoremove -y && apt-get clean -y


### From Rust official docker
# FROM debian:buster

# ENV RUSTUP_HOME=/usr/local/rustup \
#     CARGO_HOME=/usr/local/cargo \
#     PATH=/usr/local/cargo/bin:$PATH \
#     RUST_VERSION=1.76.0

# RUN set -eux; \
#     apt-get update; \
#     apt-get install -y --no-install-recommends \
#         ca-certificates \
#         gcc \
#         libc6-dev \
#         wget \
#         ; \
#     dpkgArch="$(dpkg --print-architecture)"; \
#     case "${dpkgArch##*-}" in \
#         amd64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='a3d541a5484c8fa2f1c21478a6f6c505a778d473c21d60a18a4df5185d320ef8' ;; \
#         armhf) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='7cff34808434a28d5a697593cd7a46cefdf59c4670021debccd4c86afde0ff76' ;; \
#         arm64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='76cd420cb8a82e540025c5f97bda3c65ceb0b0661d5843e6ef177479813b0367' ;; \
#         i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='cacdd10eb5ec58498cd95dbb7191fdab5fa4343e05daaf0fb7cdcae63be0a272' ;; \
#         *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
#     esac; \
#     url="https://static.rust-lang.org/rustup/archive/1.27.0/${rustArch}/rustup-init"; \
#     wget "$url"; \
#     echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
#     chmod +x rustup-init; \
#     ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${rustArch}; \
#     rm rustup-init; \
#     chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
#     rustup --version; \
#     cargo --version; \
#     rustc --version; \
#     apt-get remove -y --auto-remove \
#         wget \
#         ; \
#     rm -rf /var/lib/apt/lists/*;


### Also from official Rust
# Use the official Rust image as a base
FROM rust:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    # liblapack-dev \
    # libblas-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-depgraph
RUN cargo install cargo-depgraph

# Set TERM environment variable
ENV TERM xterm

# Create a new user to avoid running as root
RUN useradd -m rustuser
USER rustuser

# Set the working directory
WORKDIR /home/rustuser/app

# Copy the current directory contents into the container at /home/rustuser/app
COPY . .

# Run the formatting and testing commands
CMD ["make", "prep"] && ["make", "test"]


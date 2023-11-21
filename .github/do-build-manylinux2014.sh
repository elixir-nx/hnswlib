#!/bin/sh

set -x

MIX_ENV=$1
OTP_VERSION=$2
ELIXIR_VERSION=$3
OPENSSL_VERSION=$4
ARCH=$5
HNSWLIB_CI_PRECOMPILE=$6

OTP_MAJOR_VERSION=$(cut -d "." -f 1 <<< "$OTP_VERSION")
OPENSSL_VERSION=${OPENSSL_VERSION:-3.1.1}
PERFIX_DIR="/openssl-${ARCH}"
OPENSSL_ARCHIVE="openssl-${ARCH}.tar.gz"
OTP_ARCHIVE="otp-${ARCH}.tar.gz"

yum install -y openssl-devel ncurses-devel && \
    cd / && \
    curl -fSL "https://github.com/cocoa-xu/elixir_make-manylinux-openssl-precompiled/releases/download/v${OPENSSL_VERSION}/${OPENSSL_ARCHIVE}" -o "${OPENSSL_ARCHIVE}" && \
    tar -xf "${OPENSSL_ARCHIVE}" && \
    curl -fSL "https://github.com/cocoa-xu/elixir_make-manylinux-otp-precompiled/releases/download/v${OTP_VERSION}/${OTP_ARCHIVE}" -o "${OTP_ARCHIVE}" && \
    tar -xf "${OTP_ARCHIVE}" && \
    export PATH="/${OTP_VERSION}/bin:${PATH}" && \
    mkdir -p "elixir-${ELIXIR_VERSION}" && \
    cd "elixir-${ELIXIR_VERSION}" && \
    curl -fSL "https://github.com/elixir-lang/elixir/releases/download/${ELIXIR_VERSION}/elixir-otp-${OTP_MAJOR_VERSION}.zip" -o "elixir-otp-${OTP_MAJOR_VERSION}.zip" && \
    unzip -q "elixir-otp-${OTP_MAJOR_VERSION}.zip" && \
    export PATH="/elixir-${ELIXIR_VERSION}/bin:${PATH}" && \
    export CMAKE_HNSWLIB_OPTIONS="-D CMAKE_C_FLAGS=\"-static-libgcc -static-libstdc++\" -D CMAKE_CXX_FLAGS=\"-static-libgcc -static-libstdc++\"" && \
    cd /work && \
    mix local.hex --force && \
    mix deps.get

# Mix compile
cd /work
export MIX_ENV="${MIX_ENV}"
export ELIXIR_MAKE_CACHE_DIR=$(pwd)/cache
export HNSWLIB_CI_PRECOMPILE="${HNSWLIB_CI_PRECOMPILE}"

mkdir -p "${ELIXIR_MAKE_CACHE_DIR}"
mix elixir_make.precompile

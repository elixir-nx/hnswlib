#!/bin/sh

set -x

MIX_ENV=$1
OTP_VERSION=$2
ELIXIR_VERSION=$3

OPENSSL_VERSION="3.1.1"
ASDF_VERSION="v0.12.0"

yum install -y openssl-devel ncurses-devel perl-IPC-Cmd python3 && \
    cd / && \
    curl -fSL https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz -o openssl-${OPENSSL_VERSION}.tar.gz && \
    tar xf openssl-${OPENSSL_VERSION}.tar.gz && \
    cd openssl-${OPENSSL_VERSION} && \
    ./Configure && make && \
    make DESTDIR="/openssl-${OPENSSL_VERSION}/openssl-build" install &&
    cd / && \
    git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch ${ASDF_VERSION} && \
    . "/root/.asdf/asdf.sh" && \
    export KERL_CONFIGURE_OPTIONS="--without-javac --with-ssl=/openssl-${OPENSSL_VERSION}/openssl-build/usr/local" && \
    asdf plugin add erlang && \
    asdf plugin add elixir && \
    asdf install erlang ${OTP_VERSION} && \
    asdf install elixir ${ELIXIR_VERSION} && \
    asdf global erlang ${OTP_VERSION} && \
    asdf global elixir ${ELIXIR_VERSION} && \
    export CMAKE_HNSWLIB_OPTIONS="-D CMAKE_C_FLAGS=\"-static-libgcc -static-libstdc++\" -D CMAKE_CXX_FLAGS=\"-static-libgcc -static-libstdc++\"" && \
    cd /work && \
    mix deps.get

# Mix compile
cd /work
export MIX_ENV="${MIX_ENV}"

mix compile

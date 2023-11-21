#!/bin/sh

set -x

MIX_ENV=$1
OTP_VERSION=$2
ELIXIR_VERSION=$3
OPENSSL_VERSION=$4
ARCH=$5
HNSWLIB_CI_PRECOMPILE=$6

sudo docker run --privileged --network=host --rm -v `pwd`:/work "quay.io/pypa/manylinux2014_$ARCH:latest" \
    sh -c "chmod a+x /work/do-build-manylinux2014.sh && /work/do-build-manylinux2014.sh ${MIX_ENV} ${OTP_VERSION} ${ELIXIR_VERSION} ${OPENSSL_VERSION} ${ARCH} ${HNSWLIB_CI_PRECOMPILE}"

if [ -d "`pwd`/cache" ]; then
  sudo chmod -R a+wr `pwd`/cache ;
fi

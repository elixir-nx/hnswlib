ifndef MIX_APP_PATH
	MIX_APP_PATH=$(shell pwd)
endif

PRIV_DIR = $(MIX_APP_PATH)/priv
NIF_SO = $(PRIV_DIR)/hnswlib_nif.so
HNSWLIB_SRC = $(shell pwd)/3rd_party/hnswlib
C_SRC = $(shell pwd)/c_src

ifdef CC_PRECOMPILER_CURRENT_TARGET
	ifeq ($(findstring darwin, $(CC_PRECOMPILER_CURRENT_TARGET)), darwin)
		ifeq ($(findstring aarch64, $(CC_PRECOMPILER_CURRENT_TARGET)), aarch64)
			CMAKE_CONFIGURE_FLAGS=-D CMAKE_OSX_ARCHITECTURES=arm64
		else
			CMAKE_CONFIGURE_FLAGS=-D CMAKE_OSX_ARCHITECTURES=x86_64
		endif
	endif
endif

ifdef CMAKE_TOOLCHAIN_FILE
	CMAKE_CONFIGURE_FLAGS=-D CMAKE_TOOLCHAIN_FILE="$(CMAKE_TOOLCHAIN_FILE)"
endif

CMAKE_BUILD_TYPE ?= Release
DEFAULT_JOBS ?= 1
CMAKE_HNSWLIB_BUILD_DIR = $(MIX_APP_PATH)/cmake_hnswlib
CMAKE_HNSWLIB_OPTIONS ?= ""
MAKE_BUILD_FLAGS ?= -j$(DEFAULT_JOBS)

.DEFAULT_GLOBAL := build

build: $(NIF_SO)
	@echo > /dev/null

$(NIF_SO):
	@ mkdir -p "$(PRIV_DIR)"
	@ if [ ! -f "${NIF_SO}" ]; then \
		mkdir -p "$(CMAKE_HNSWLIB_BUILD_DIR)" && \
		cd "$(CMAKE_HNSWLIB_BUILD_DIR)" && \
			{ cmake --no-warn-unused-cli \
			-D CMAKE_BUILD_TYPE="$(CMAKE_BUILD_TYPE)" \
			-D C_SRC="$(C_SRC)" \
			-D HNSWLIB_SRC="$(HNSWLIB_SRC)" \
			-D MIX_APP_PATH="$(MIX_APP_PATH)" \
			-D PRIV_DIR="$(PRIV_DIR)" \
			-D ERTS_INCLUDE_DIR="$(ERTS_INCLUDE_DIR)" \
			$(CMAKE_CONFIGURE_FLAGS) $(CMAKE_HNSWLIB_OPTIONS) "$(shell pwd)" && \
			make "$(MAKE_BUILD_FLAGS)" \
			|| { echo "\033[0;31mincomplete build of hnswlib found in '$(CMAKE_HNSWLIB_BUILD_DIR)', please delete that directory and retry\033[0m" && exit 1 ; } ; } \
			&& if [ "$(EVISION_PREFER_PRECOMPILED)" != "true" ]; then \
				cp "$(CMAKE_HNSWLIB_BUILD_DIR)/hnswlib_nif.so" "$(NIF_SO)" ; \
			fi ; \
		fi

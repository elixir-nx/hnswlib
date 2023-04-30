#include <erl_nif.h>
#include <stdbool.h>
#include <stdio.h>
#include "nif_utils.hpp"
#include "hnswlib_helper.hpp"
#include "hnswlib_index.hpp"
#include "hnswlib_nif_resource.h"

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

ErlNifResourceType * NifResHNSWLibIndex::type = nullptr;

static ERL_NIF_TERM hnswlib_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments: space and dim");
    }

    std::string space;
    size_t dim;
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if (!erlang::nif::get_atom(env, argv[0], space)) {
        return erlang::nif::error(env, "expect parameter `space` to be an atom");
    }
    if (!erlang::nif::get(env, argv[1], &dim)) {
        return erlang::nif::error(env, "expect parameter `space` to be a non-negative integer");
    }

    if ((index = NifResHNSWLibIndex::allocate_resource(env, error)) == nullptr) {
        return error;
    }

    index->val = new Index<float>(space, dim);

    ret = enif_make_resource(env, index);
    enif_release_resource(index);
    return erlang::nif::ok(env, ret);
}

static int on_load(ErlNifEnv *env, void **_sth1, ERL_NIF_TERM _sth2) {
    ErlNifResourceType *rt;

    rt = enif_open_resource_type(env, "Elixir.HNSWLib.Nif", "NifResHNSWLibIndex", NifResHNSWLibIndex::destruct_resource, ERL_NIF_RT_CREATE, NULL);
    if (!rt) return -1;
    NifResHNSWLibIndex::type = rt;

    return 0;
}

static int on_reload(ErlNifEnv *_sth0, void **_sth1, ERL_NIF_TERM _sth2) {
    return 0;
}

static int on_upgrade(ErlNifEnv *_sth0, void **_sth1, void **_sth2, ERL_NIF_TERM _sth3) {
    return 0;
}

#define F_CPU(NAME, ARITY) {#NAME, ARITY, NAME, ERL_NIF_DIRTY_JOB_CPU_BOUND}
#define F_IO(NAME, ARITY) {#NAME, ARITY, NAME, ERL_NIF_DIRTY_JOB_IO_BOUND}

static ErlNifFunc nif_functions[] = {
    {"new", 2, hnswlib_new, 0}
};

ERL_NIF_INIT(Elixir.HNSWLib.Nif, nif_functions, on_load, on_reload, on_upgrade, NULL);

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

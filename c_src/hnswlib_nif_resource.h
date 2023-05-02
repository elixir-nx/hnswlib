#ifndef HNSWLIB_NIF_RESOURCE_H
#define HNSWLIB_NIF_RESOURCE_H

#pragma once

#include <atomic>
#include <string>
#include <erl_nif.h>
#include "hnswlib_index.hpp"

struct NifResHNSWLibIndex {
    Index<float> * val;

    static ErlNifResourceType * type;
    static NifResHNSWLibIndex * allocate_resource(ErlNifEnv * env, ERL_NIF_TERM &error);
    static NifResHNSWLibIndex * get_resource(ErlNifEnv * env, ERL_NIF_TERM term, ERL_NIF_TERM &error);
    static void destruct_resource(ErlNifEnv *env, void *args);
};

struct NifResHNSWLibBFIndex {
    BFIndex<float> * val;

    static ErlNifResourceType * type;
    static NifResHNSWLibBFIndex * allocate_resource(ErlNifEnv * env, ERL_NIF_TERM &error);
    static NifResHNSWLibBFIndex * get_resource(ErlNifEnv * env, ERL_NIF_TERM term, ERL_NIF_TERM &error);
    static void destruct_resource(ErlNifEnv *env, void *args);
};

#endif  /* HNSWLIB_NIF_RESOURCE_H */

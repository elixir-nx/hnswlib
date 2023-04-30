#include <erl_nif.h>
#include "nif_utils.hpp"

#include "hnswlib_nif_resource.h"

NifResHNSWLibIndex * NifResHNSWLibIndex::allocate_resource(ErlNifEnv * env, ERL_NIF_TERM &error) {
    NifResHNSWLibIndex * res = (NifResHNSWLibIndex *)enif_alloc_resource(NifResHNSWLibIndex::type, sizeof(NifResHNSWLibIndex));
    if (res == nullptr) {
        error = erlang::nif::error(env, "cannot allocate NifResHNSWLibIndex resource");
        return res;
    }

    return res;
}

NifResHNSWLibIndex * NifResHNSWLibIndex::get_resource(ErlNifEnv * env, ERL_NIF_TERM term, ERL_NIF_TERM &error) {
    NifResHNSWLibIndex * self_res = nullptr;
    if (!enif_get_resource(env, term, NifResHNSWLibIndex::type, (void **)&self_res) || self_res == nullptr || self_res->val == nullptr) {
        error = erlang::nif::error(env, "cannot access NifResHNSWLibIndex resource");
    }
    return self_res;
}

void NifResHNSWLibIndex::destruct_resource(ErlNifEnv *env, void *args) {
    auto res = (NifResHNSWLibIndex *)args;
    if (res) {
        if (res->val) {
            delete res->val;
            res->val = nullptr;
        }
    }
}

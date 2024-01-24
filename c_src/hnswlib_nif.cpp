#include <erl_nif.h>
#include <stdbool.h>
#include <stdio.h>
#include <climits>
#include "nif_utils.hpp"
#include "hnswlib_index.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

ErlNifResourceType * NifResHNSWLibIndex::type = nullptr;
ErlNifResourceType * NifResHNSWLibBFIndex::type = nullptr;

static ERL_NIF_TERM hnswlib_index_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    std::string space;
    size_t dim;
    size_t max_elements;
    size_t m = 16;
    size_t ef_construction = 200;
    size_t random_seed = 100;
    bool allow_replace_deleted = false;
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if (!erlang::nif::get_atom(env, argv[0], space)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &dim)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[2], &max_elements)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[3], &m)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[4], &ef_construction)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[5], &random_seed)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[6], &allow_replace_deleted)) {
        return enif_make_badarg(env);
    }

    if ((index = NifResHNSWLibIndex::allocate_resource(env, error)) == nullptr) {
        return error;
    }

    index->val = nullptr;
    try {
        index->val = new Index<float>(space, dim);
        index->val->init_new_index(max_elements, m, ef_construction, random_seed, allow_replace_deleted);
    } catch (std::runtime_error &err) {
        if (index->val) {
            delete index->val;
            index->val = nullptr;
        }
        enif_release_resource(index);
        return erlang::nif::error(env, err.what());
    }

    ret = enif_make_resource(env, index);
    enif_release_resource(index);
    return erlang::nif::ok(env, ret);
}

static ERL_NIF_TERM hnswlib_index_knn_query(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ErlNifBinary data;
    size_t k;
    long long num_threads;
    ERL_NIF_TERM filter;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &data)) {
        return enif_make_badarg(env);
    }
    if (data.size % sizeof(float) != 0) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[2], &k) || k == 0) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[3], &num_threads)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_fun(env, argv[4]) && !erlang::nif::check_nil(env, argv[4])) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[5], &rows)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[6], &features)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rlock(index->rwlock);
    index->val->knnQuery(env, (float *)data.data, rows, features, k, num_threads, ret);
    enif_rwlock_runlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_add_items(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ErlNifBinary f32_data;
    ErlNifBinary ids_binary;
    size_t ids_count = 0;
    long long num_threads;
    bool replace_deleted;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &f32_data)) {
        return enif_make_badarg(env);
    }
    if (f32_data.size % sizeof(float) != 0) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[2], &ids_binary)) {
        if (!erlang::nif::check_nil(env, argv[2])) {
            return enif_make_badarg(env);
        } else {
            ids_binary.data = nullptr;
            ids_binary.size = 0;
        }
    } else {
        if (ids_binary.size % sizeof(uint64_t) != 0) {
            return enif_make_badarg(env);
        } else {
            ids_count = ids_binary.size / sizeof(uint64_t);
        }
    }
    if (!erlang::nif::get(env, argv[3], &num_threads)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[4], &replace_deleted)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[5], &rows)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[6], &features)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rwlock(index->rwlock);
    try {
        index->val->addItems((float *)f32_data.data, rows, features, (const uint64_t *)ids_binary.data, ids_count, num_threads, replace_deleted);
        ret = erlang::nif::ok(env);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
    }
    enif_rwlock_rwunlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_save_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    std::string path;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], path)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rlock(index->rwlock);
    try {
        index->val->saveIndex(path);
        ret = erlang::nif::ok(env);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
    } catch (...) {
        ret = erlang::nif::error(env, "cannot save index: unknown reason");
    }
    enif_rwlock_runlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_load_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    std::string space;
    size_t dim;
    std::string path;
    size_t max_elements;
    bool allow_replace_deleted;
    ERL_NIF_TERM ret, error;

    if (!erlang::nif::get_atom(env, argv[0], space)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &dim)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[2], path)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[3], &max_elements)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[4], &allow_replace_deleted)) {
        return enif_make_badarg(env);
    }

    if ((index = NifResHNSWLibIndex::allocate_resource(env, error)) == nullptr) {
        return error;
    }

    enif_rwlock_rwlock(index->rwlock);
    try {
        index->val = new Index<float>(space, dim);
        index->val->loadIndex(path, max_elements, allow_replace_deleted);

        ret = erlang::nif::ok(env, enif_make_resource(env, index));
    } catch (std::runtime_error &err) {
        if (index->val) {
            delete index->val;
            index->val = nullptr;
        }
        ret = erlang::nif::error(env, err.what());
    }
    enif_rwlock_rwunlock(index->rwlock);
    enif_release_resource(index);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_mark_deleted(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    size_t label;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &label)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rwlock(index->rwlock);
    try {
        index->val->markDeleted(label);
        ret = erlang::nif::ok(env);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
    }
    enif_rwlock_rwunlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_unmark_deleted(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    size_t label;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &label)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rwlock(index->rwlock);
    try {
        index->val->unmarkDeleted(label);
        ret = erlang::nif::ok(env);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
    }
    enif_rwlock_rwunlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_resize_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    size_t new_size;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &new_size)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rwlock(index->rwlock);
    try {
        index->val->resizeIndex(new_size);
        ret = erlang::nif::ok(env);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
    } catch (std::bad_alloc&) {
        ret = erlang::nif::error(env, "no enough memory available to resize the index");
    }
    enif_rwlock_rwunlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_get_max_elements(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }

    size_t max_elements = index->val->getMaxElements();
    return erlang::nif::ok(env, erlang::nif::make(env, (unsigned long long)max_elements));
}

static ERL_NIF_TERM hnswlib_index_get_current_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }

    size_t count = index->val->getCurrentCount();
    return erlang::nif::ok(env, erlang::nif::make(env, (unsigned long long)count));
}

static ERL_NIF_TERM hnswlib_index_get_ef(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (unsigned long long)(index->val->index_inited ? index->val->appr_alg->ef_ : index->val->default_ef)));
}

static ERL_NIF_TERM hnswlib_index_set_ef(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    size_t new_ef;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &new_ef)) {
        return enif_make_badarg(env);
    }

    index->val->set_ef(new_ef);
    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_get_num_threads(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (int64_t)index->val->num_threads_default));
}

static ERL_NIF_TERM hnswlib_index_set_num_threads(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    int new_num_threads;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &new_num_threads)) {
        return enif_make_badarg(env);
    }
    index->val->num_threads_default = new_num_threads;

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_get_items(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ErlNifBinary ids_binary;
    size_t ids_count = 0;
    std::string return_type;
    ERL_NIF_TERM ret = 0, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &ids_binary)) {
        if (!erlang::nif::check_nil(env, argv[1])) {
            return enif_make_badarg(env);
        } else {
            ids_binary.data = nullptr;
            ids_binary.size = 0;
        }
    } else {
        if (ids_binary.size % sizeof(uint64_t) != 0) {
            return enif_make_badarg(env);
        } else {
            ids_count = ids_binary.size / sizeof(uint64_t);
        }
    }

    std::vector<std::vector<float>> data;
    bool has_error = false;
    enif_rwlock_rwlock(index->rwlock);
    try {
        data = index->val->getDataReturnList((const uint64_t *)ids_binary.data, ids_count);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
        has_error = true;
    }
    enif_rwlock_rwunlock(index->rwlock);
    if (has_error) return ret;

    std::vector<ERL_NIF_TERM> ret_list;
    for (auto& d : data) {
        ErlNifBinary bin;
        size_t bin_size = d.size() * sizeof(float);
        if (!enif_alloc_binary(bin_size, &bin)) {
            return erlang::nif::error(env, "cannot allocate enough memory to hold the list");
        }
        memcpy(bin.data, d.data(), bin_size);
        ret_list.push_back(enif_make_binary(env, &bin));
    }

    return erlang::nif::ok(env, enif_make_list_from_array(env, ret_list.data(), (unsigned)ret_list.size()));
}

static ERL_NIF_TERM hnswlib_index_get_ids_list(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rlock(index->rwlock);
    std::vector<hnswlib::labeltype> ids = index->val->getIdsList();
    enif_rwlock_runlock(index->rwlock);
    if (erlang::nif::make(env, ids, ret)) {
        return erlang::nif::error(env, "cannot allocate enough memory to hold the list");
    }

    return erlang::nif::ok(env, ret);
}

static ERL_NIF_TERM hnswlib_index_get_ef_construction(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (unsigned long long)(index->val->index_inited ? index->val->appr_alg->ef_construction_ : 0)));
}

static ERL_NIF_TERM hnswlib_index_get_m(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (unsigned long long)(index->val->index_inited ? index->val->appr_alg->M_ : 0)));
}

static ERL_NIF_TERM hnswlib_bfindex_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    std::string space;
    size_t dim;
    size_t max_elements;
    NifResHNSWLibBFIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if (!erlang::nif::get_atom(env, argv[0], space)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &dim)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[2], &max_elements)) {
        return enif_make_badarg(env);
    }

    if ((index = NifResHNSWLibBFIndex::allocate_resource(env, error)) == nullptr) {
        return error;
    }

    index->val = nullptr;
    try {
        index->val = new BFIndex<float>(space, dim);
        index->val->init_new_index(max_elements);
    } catch (std::runtime_error &err) {
        if (index->val) {
            delete index->val;
            index->val = nullptr;
        }
        enif_release_resource(index);
        return erlang::nif::error(env, err.what());
    }

    ret = enif_make_resource(env, index);
    enif_release_resource(index);
    return erlang::nif::ok(env, ret);
}

static ERL_NIF_TERM hnswlib_bfindex_knn_query(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibBFIndex * index = nullptr;
    ErlNifBinary data;
    size_t k;
    ERL_NIF_TERM filter;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &data)) {
        return enif_make_badarg(env);
    }
    if (data.size % sizeof(float) != 0) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[2], &k) || k == 0) {
        return enif_make_badarg(env);
    }
    if (!enif_is_fun(env, argv[3]) && !erlang::nif::check_nil(env, argv[3])) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[4], &rows)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[5], &features)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rlock(index->rwlock);
    index->val->knnQuery(env, (float *)data.data, rows, features, k, ret);
    enif_rwlock_runlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_bfindex_add_items(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibBFIndex * index = nullptr;
    ErlNifBinary f32_data;
    ErlNifBinary ids_binary;
    size_t ids_count;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &f32_data)) {
        return enif_make_badarg(env);
    }
    if (f32_data.size % sizeof(float) != 0) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[2], &ids_binary)) {
        if (!erlang::nif::check_nil(env, argv[2])) {
            return enif_make_badarg(env);
        } else {
            ids_binary.data = nullptr;
            ids_binary.size = 0;
        }
    } else {
        if (ids_binary.size % sizeof(uint64_t) != 0) {
            return enif_make_badarg(env);
        } else {
            ids_count = ids_binary.size / sizeof(uint64_t);
        }
    }
    if (!erlang::nif::get(env, argv[3], &rows)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[4], &features)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rwlock(index->rwlock);
    try {
        index->val->addItems((float *)f32_data.data, rows, features, (const uint64_t *)ids_binary.data, ids_count);
        ret = erlang::nif::ok(env);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
    }
    enif_rwlock_rwunlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_bfindex_delete_vector(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibBFIndex * index = nullptr;
    size_t label;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &label)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rwlock(index->rwlock);
    index->val->deleteVector(label);
    enif_rwlock_rwunlock(index->rwlock);

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_bfindex_save_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibBFIndex * index = nullptr;
    std::string path;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], path)) {
        return enif_make_badarg(env);
    }

    enif_rwlock_rlock(index->rwlock);
    try {
        index->val->saveIndex(path);
        ret = erlang::nif::ok(env);
    } catch (std::runtime_error &err) {
        ret = erlang::nif::error(env, err.what());
    } catch (...) {
        ret = erlang::nif::error(env, "cannot save index: unknown reason");
    }
    enif_rwlock_runlock(index->rwlock);

    return ret;
}

static ERL_NIF_TERM hnswlib_bfindex_load_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NifResHNSWLibBFIndex * index = nullptr;
    std::string space;
    size_t dim;
    std::string path;
    size_t max_elements;
    ERL_NIF_TERM ret, error;

    if (!erlang::nif::get_atom(env, argv[0], space)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[1], &dim)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[2], path)) {
        return enif_make_badarg(env);
    }
    if (!erlang::nif::get(env, argv[3], &max_elements)) {
        return enif_make_badarg(env);
    }

    if ((index = NifResHNSWLibBFIndex::allocate_resource(env, error)) == nullptr) {
        return error;
    }

    enif_rwlock_rlock(index->rwlock);
    try {
        index->val = new BFIndex<float>(space, dim);
        index->val->loadIndex(path, max_elements);
        
        ret = erlang::nif::ok(env, enif_make_resource(env, index));
    } catch (std::runtime_error &err) {
        if (index->val) {
            delete index->val;
            index->val = nullptr;
        }
        ret = erlang::nif::error(env, err.what());
    }
    enif_rwlock_runlock(index->rwlock);
    enif_release_resource(index);

    return ret;
}

static ERL_NIF_TERM hnswlib_bfindex_get_max_elements(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ERL_NIF_TERM error;
    NifResHNSWLibBFIndex * index = nullptr;
    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return enif_make_badarg(env);
    }
    
    size_t max_elements = index->val->getMaxElements();
    return erlang::nif::ok(env, erlang::nif::make(env, (unsigned long long)max_elements));
}

static ERL_NIF_TERM hnswlib_float_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    return enif_make_uint(env, sizeof(float));
}

static int on_load(ErlNifEnv *env, void **, ERL_NIF_TERM) {
    ErlNifResourceType *rt;

    rt = enif_open_resource_type(env, "Elixir.HNSWLib.Nif", "NifResHNSWLibIndex", NifResHNSWLibIndex::destruct_resource, ERL_NIF_RT_CREATE, NULL);
    if (!rt) return -1;
    NifResHNSWLibIndex::type = rt;

    rt = enif_open_resource_type(env, "Elixir.HNSWLib.Nif", "NifResHNSWLibBFIndex", NifResHNSWLibBFIndex::destruct_resource, ERL_NIF_RT_CREATE, NULL);
    if (!rt) return -1;
    NifResHNSWLibBFIndex::type = rt;

    return 0;
}

static int on_reload(ErlNifEnv *, void **, ERL_NIF_TERM) {
    return 0;
}

static int on_upgrade(ErlNifEnv *, void **, void **, ERL_NIF_TERM) {
    return 0;
}

static ErlNifFunc nif_functions[] = {
    {"index_new", 7, hnswlib_index_new, 0},
    {"index_knn_query", 7, hnswlib_index_knn_query, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"index_add_items", 7, hnswlib_index_add_items, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"index_get_items", 2, hnswlib_index_get_items, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"index_get_ids_list", 1, hnswlib_index_get_ids_list, 0},
    {"index_get_ef", 1, hnswlib_index_get_ef, 0},
    {"index_set_ef", 2, hnswlib_index_set_ef, 0},
    {"index_get_num_threads", 1, hnswlib_index_get_num_threads, 0},
    {"index_set_num_threads", 2, hnswlib_index_set_num_threads, 0},
    {"index_save_index", 2, hnswlib_index_save_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"index_load_index", 5, hnswlib_index_load_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"index_mark_deleted", 2, hnswlib_index_mark_deleted, 0},
    {"index_unmark_deleted", 2, hnswlib_index_unmark_deleted, 0},
    {"index_resize_index", 2, hnswlib_index_resize_index, 0},
    {"index_get_max_elements", 1, hnswlib_index_get_max_elements, 0},
    {"index_get_current_count", 1, hnswlib_index_get_current_count, 0},
    {"index_get_ef_construction", 1, hnswlib_index_get_ef_construction, 0},
    {"index_get_m", 1, hnswlib_index_get_m, 0},

    {"bfindex_new", 3, hnswlib_bfindex_new, 0},
    {"bfindex_knn_query", 6, hnswlib_bfindex_knn_query, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"bfindex_add_items", 5, hnswlib_bfindex_add_items, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"bfindex_delete_vector", 2, hnswlib_bfindex_delete_vector, 0},
    {"bfindex_save_index", 2, hnswlib_bfindex_save_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"bfindex_load_index", 4, hnswlib_bfindex_load_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"bfindex_get_max_elements", 1, hnswlib_bfindex_get_max_elements, 0},

    {"float_size", 0, hnswlib_float_size, 0}
};

ERL_NIF_INIT(Elixir.HNSWLib.Nif, nif_functions, on_load, on_reload, on_upgrade, NULL);

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

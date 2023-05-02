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
ErlNifResourceType * NifResHNSWLibBFIndex::type = nullptr;

static ERL_NIF_TERM hnswlib_index_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 7) {
        return erlang::nif::error(env, "expecting 7 arguments");
    }

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
        return erlang::nif::error(env, "expect parameter `space` to be an atom");
    }
    if (!erlang::nif::get(env, argv[1], &dim)) {
        return erlang::nif::error(env, "expect parameter `space` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[2], &max_elements)) {
        return erlang::nif::error(env, "expect parameter `max_elements` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[3], &m)) {
        return erlang::nif::error(env, "expect parameter `m` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[4], &ef_construction)) {
        return erlang::nif::error(env, "expect parameter `ef_construction` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[5], &random_seed)) {
        return erlang::nif::error(env, "expect parameter `random_seed` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[6], &allow_replace_deleted)) {
        return erlang::nif::error(env, "expect parameter `allow_replace_deleted` to be a boolean");
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
        }
        enif_release_resource(index);
        return erlang::nif::error(env, err.what());
    }

    ret = enif_make_resource(env, index);
    enif_release_resource(index);
    return erlang::nif::ok(env, ret);
}

static ERL_NIF_TERM hnswlib_index_knn_query(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 7) {
        return erlang::nif::error(env, "expecting 7 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    ErlNifBinary data;
    size_t k;
    ssize_t num_threads;
    ERL_NIF_TERM filter;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!enif_inspect_binary(env, argv[1], &data)) {
        return erlang::nif::error(env, "expect `data` to be a binary");
    }
    if (data.size % sizeof(float) != 0) {
        return erlang::nif::error(env, (
            std::string{"expect `data`'s size to be a multiple of "} + std::to_string(sizeof(float)) + " (sizeof(float)), got `" + std::to_string(data.size) + "` bytes").c_str());
    }
    if (!erlang::nif::get(env, argv[2], &k) || k == 0) {
        return erlang::nif::error(env, "expect parameter `k` to be a positive integer");
    }
    if (!erlang::nif::get(env, argv[3], &num_threads)) {
        return erlang::nif::error(env, "expect parameter `num_threads` to be an integer");
    }
    if (!enif_is_fun(env, argv[4]) && !erlang::nif::check_nil(env, argv[4])) {
        return erlang::nif::error(env, "expect parameter `filter` to be a function or `nil`");
    }
    if (!erlang::nif::get(env, argv[5], &rows)) {
        return erlang::nif::error(env, "expect parameter `rows` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[6], &features)) {
        return erlang::nif::error(env, "expect parameter `features` to be a non-negative integer");
    }

    index->val->knnQuery(env, (float *)data.data, rows, features, k, num_threads, ret);

    return ret;
}

static ERL_NIF_TERM hnswlib_index_add_items(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 7) {
        return erlang::nif::error(env, "expecting 7 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    ErlNifBinary f32_data;
    ErlNifBinary ids_binary;
    std::vector<uint64_t> ids;
    ssize_t num_threads;
    bool replace_deleted;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!enif_inspect_binary(env, argv[1], &f32_data)) {
        return erlang::nif::error(env, "expect `f32_data` to be a binary");
    }
    if (f32_data.size % sizeof(float) != 0) {
        return erlang::nif::error(env, (
            std::string{"expect `f32_data`'s size to be a multiple of "} + std::to_string(sizeof(float)) + " (sizeof(float)), got `" + std::to_string(f32_data.size) + "` bytes").c_str());
    }
    if (!enif_inspect_binary(env, argv[2], &ids_binary)) {
        if (!erlang::nif::check_nil(env, argv[2])) {
            if (!erlang::nif::get_list(env, argv[2], ids)) {
                return erlang::nif::error(env, "expect `ids` to be either a binary, `nil`, or a list of non-negative integers.");
            }
        }
    } else {
        if (ids_binary.size % sizeof(uint64_t) != 0) {
            return erlang::nif::error(env, (
                std::string{"expect `ids`'s size to be a multiple of "} + std::to_string(sizeof(uint64_t)) + " (sizeof(uint64_t)), got `" + std::to_string(ids_binary.size) + "` bytes").c_str());
        } else {
            uint64_t * ptr = (uint64_t *)ids_binary.data;
            size_t count = ids_binary.size / sizeof(uint64_t);
            ids = std::vector<uint64_t>{ptr, ptr + count};
        }
    }
    if (!erlang::nif::get(env, argv[3], &num_threads)) {
        return erlang::nif::error(env, "expect parameter `num_threads` to be an integer");
    }
    if (!erlang::nif::get(env, argv[4], &replace_deleted)) {
        return erlang::nif::error(env, "expect parameter `replace_deleted` to be a boolean");
    }
    if (!erlang::nif::get(env, argv[5], &rows)) {
        return erlang::nif::error(env, "expect parameter `rows` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[6], &features)) {
        return erlang::nif::error(env, "expect parameter `features` to be a non-negative integer");
    }

    try {
        index->val->addItems((float *)f32_data.data, rows, features, ids, num_threads, replace_deleted);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_save_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    std::string path;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], path)) {
        return erlang::nif::error(env, "expect parameter `path` to be a string");
    }

    try {
        index->val->saveIndex(path);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    } catch (...) {
        return erlang::nif::error(env, "cannot save index: unknown reason");
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_load_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 4) {
        return erlang::nif::error(env, "expecting 4 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    std::string path;
    size_t max_elements;
    bool allow_replace_deleted;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], path)) {
        return erlang::nif::error(env, "expect parameter `path` to be a string");
    }
    if (!erlang::nif::get(env, argv[2], &max_elements)) {
        return erlang::nif::error(env, "expect parameter `max_elements` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[3], &allow_replace_deleted)) {
        return erlang::nif::error(env, "expect parameter `allow_replace_deleted` to be a boolean");
    }

    try {
        index->val->loadIndex(path, max_elements, allow_replace_deleted);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_mark_deleted(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    size_t label;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], &label)) {
        return erlang::nif::error(env, "expect parameter `label` to be a non-negative integer");
    }

    try {
        index->val->markDeleted(label);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_unmark_deleted(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    size_t label;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], &label)) {
        return erlang::nif::error(env, "expect parameter `label` to be a non-negative integer");
    }

    try {
        index->val->unmarkDeleted(label);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_resize_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    size_t new_size;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], &new_size)) {
        return erlang::nif::error(env, "expect parameter `new_size` to be a non-negative integer");
    }

    try {
        index->val->resizeIndex(new_size);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    } catch (std::bad_alloc&) {
        return erlang::nif::error(env, "no enough memory available to resize the index");
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_get_max_elements(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return erlang::nif::error(env, "expecting 1 argument");
    }

    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }

    size_t max_elements = index->val->getMaxElements();
    return erlang::nif::ok(env, erlang::nif::make(env, (uint64_t)max_elements));
}

static ERL_NIF_TERM hnswlib_index_get_current_count(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return erlang::nif::error(env, "expecting 1 argument");
    }

    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }

    size_t count = index->val->getCurrentCount();
    return erlang::nif::ok(env, erlang::nif::make(env, (uint64_t)count));
}

static ERL_NIF_TERM hnswlib_index_get_ef(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return erlang::nif::error(env, "expecting 1 argument");
    }

    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (uint64_t)(index->val->index_inited ? index->val->appr_alg->ef_ : index->val->default_ef)));
}

static ERL_NIF_TERM hnswlib_index_set_ef(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    size_t new_ef;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], &new_ef)) {
        return erlang::nif::error(env, "expect parameter `new_ef` to be a non-negative integer");
    }

    index->val->set_ef(new_ef);
    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_get_num_threads(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return erlang::nif::error(env, "expecting 1 argument");
    }

    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (int64_t)index->val->num_threads_default));
}

static ERL_NIF_TERM hnswlib_index_set_num_threads(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    int new_num_threads;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], &new_num_threads)) {
        return erlang::nif::error(env, (std::string{"expect parameter `num_threads` to be an integer between " + std::to_string(INT_MIN) + " and " + std::to_string(INT_MAX) +  "."}).c_str());
    }
    index->val->num_threads_default = new_num_threads;

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_index_get_items(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 3) {
        return erlang::nif::error(env, "expecting 3 arguments");
    }

    NifResHNSWLibIndex * index = nullptr;
    ErlNifBinary ids_binary;
    std::vector<uint64_t> ids;
    std::string return_type;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!enif_inspect_binary(env, argv[1], &ids_binary)) {
        if (!erlang::nif::check_nil(env, argv[1])) {
            if (!erlang::nif::get_list(env, argv[1], ids)) {
                return erlang::nif::error(env, "expect `ids` to be either a binary, `nil`, or a list of non-negative integers.");
            }
        }
    } else {
        if (ids_binary.size % sizeof(uint64_t) != 0) {
            return erlang::nif::error(env, (
                std::string{"expect `ids`'s size to be a multiple of "} + std::to_string(sizeof(uint64_t)) + " (sizeof(uint64_t)), got `" + std::to_string(ids_binary.size) + "` bytes").c_str());
        } else {
            uint64_t * ptr = (uint64_t *)ids_binary.data;
            size_t count = ids_binary.size / sizeof(uint64_t);
            ids = std::vector<uint64_t>{ptr, ptr + count};
        }
    }
    if (!erlang::nif::get_atom(env, argv[2], return_type)) {
        return erlang::nif::error(env, "expect `return` to be an atom");
    }
    if (!(return_type == "tensor" || return_type == "binary" || return_type == "list")) {
        return erlang::nif::error(env, "expect `return` to be an atom and is one of `:tensor`, `:binary` or `:list`.");
    }

    std::vector<std::vector<float>> data;
    try {
        data = index->val->getDataReturnList(ids);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    }

    std::vector<ERL_NIF_TERM> ret_list;
    if (return_type == "list") {
        for (auto& d : data) {
            ERL_NIF_TERM cur;
            if (erlang::nif::make(env, d, cur)) {
                return erlang::nif::error(env, "cannot allocate enough memory to hold the list");
            }
            ret_list.push_back(cur);
        }

        return erlang::nif::ok(env, enif_make_list_from_array(env, ret_list.data(), (unsigned)ret_list.size()));
    } else {
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
}

static ERL_NIF_TERM hnswlib_index_get_ids_list(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return erlang::nif::error(env, "expecting 1 argument");
    }

    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }

    std::vector<hnswlib::labeltype> ids = index->val->getIdsList();
    if (erlang::nif::make(env, ids, ret)) {
        return erlang::nif::error(env, "cannot allocate enough memory to hold the list");
    }

    return erlang::nif::ok(env, ret);
}

static ERL_NIF_TERM hnswlib_index_get_ef_construction(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return erlang::nif::error(env, "expecting 1 argument");
    }

    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (uint64_t)(index->val->index_inited ? index->val->appr_alg->ef_construction_ : 0)));
}

static ERL_NIF_TERM hnswlib_index_get_m(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return erlang::nif::error(env, "expecting 1 argument");
    }

    NifResHNSWLibIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }

    return erlang::nif::ok(env, erlang::nif::make(env, (uint64_t)(index->val->index_inited ? index->val->appr_alg->M_ : 0)));
}

static ERL_NIF_TERM hnswlib_bfindex_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 3) {
        return erlang::nif::error(env, "expecting 3 arguments");
    }

    std::string space;
    size_t dim;
    size_t max_elements;
    NifResHNSWLibBFIndex * index = nullptr;
    ERL_NIF_TERM ret, error;

    if (!erlang::nif::get_atom(env, argv[0], space)) {
        return erlang::nif::error(env, "expect parameter `space` to be an atom");
    }
    if (!erlang::nif::get(env, argv[1], &dim)) {
        return erlang::nif::error(env, "expect parameter `space` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[2], &max_elements)) {
        return erlang::nif::error(env, "expect parameter `max_elements` to be a non-negative integer");
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
        }
        enif_release_resource(index);
        return erlang::nif::error(env, err.what());
    }

    ret = enif_make_resource(env, index);
    enif_release_resource(index);
    return erlang::nif::ok(env, ret);
}

static ERL_NIF_TERM hnswlib_bfindex_knn_query(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 6) {
        return erlang::nif::error(env, "expecting 6 arguments");
    }

    NifResHNSWLibBFIndex * index = nullptr;
    ErlNifBinary data;
    size_t k;
    ERL_NIF_TERM filter;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!enif_inspect_binary(env, argv[1], &data)) {
        return erlang::nif::error(env, "expect `data` to be a binary");
    }
    if (data.size % sizeof(float) != 0) {
        return erlang::nif::error(env, (
            std::string{"expect `data`'s size to be a multiple of "} + std::to_string(sizeof(float)) + " (sizeof(float)), got `" + std::to_string(data.size) + "` bytes").c_str());
    }
    if (!erlang::nif::get(env, argv[2], &k) || k == 0) {
        return erlang::nif::error(env, "expect parameter `k` to be a positive integer");
    }
    if (!enif_is_fun(env, argv[3]) && !erlang::nif::check_nil(env, argv[3])) {
        return erlang::nif::error(env, "expect parameter `filter` to be a function or `nil`");
    }
    if (!erlang::nif::get(env, argv[4], &rows)) {
        return erlang::nif::error(env, "expect parameter `rows` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[5], &features)) {
        return erlang::nif::error(env, "expect parameter `features` to be a non-negative integer");
    }

    index->val->knnQuery(env, (float *)data.data, rows, features, k, ret);

    return ret;
}

static ERL_NIF_TERM hnswlib_bfindex_add_items(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 5) {
        return erlang::nif::error(env, "expecting 5 arguments");
    }

    NifResHNSWLibBFIndex * index = nullptr;
    ErlNifBinary f32_data;
    ErlNifBinary ids_binary;
    std::vector<uint64_t> ids;
    size_t rows, features;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!enif_inspect_binary(env, argv[1], &f32_data)) {
        return erlang::nif::error(env, "expect `f32_data` to be a binary");
    }
    if (f32_data.size % sizeof(float) != 0) {
        return erlang::nif::error(env, (
            std::string{"expect `f32_data`'s size to be a multiple of "} + std::to_string(sizeof(float)) + " (sizeof(float)), got `" + std::to_string(f32_data.size) + "` bytes").c_str());
    }
    if (!enif_inspect_binary(env, argv[2], &ids_binary)) {
        if (!erlang::nif::check_nil(env, argv[2])) {
            if (!erlang::nif::get_list(env, argv[2], ids)) {
                return erlang::nif::error(env, "expect `ids` to be either a binary, `nil`, or a list of non-negative integers.");
            }
        }
    } else {
        if (ids_binary.size % sizeof(uint64_t) != 0) {
            return erlang::nif::error(env, (
                std::string{"expect `ids`'s size to be a multiple of "} + std::to_string(sizeof(uint64_t)) + " (sizeof(uint64_t)), got `" + std::to_string(ids_binary.size) + "` bytes").c_str());
        } else {
            uint64_t * ptr = (uint64_t *)ids_binary.data;
            size_t count = ids_binary.size / sizeof(uint64_t);
            ids = std::vector<uint64_t>{ptr, ptr + count};
        }
    }
    if (!erlang::nif::get(env, argv[3], &rows)) {
        return erlang::nif::error(env, "expect parameter `rows` to be a non-negative integer");
    }
    if (!erlang::nif::get(env, argv[4], &features)) {
        return erlang::nif::error(env, "expect parameter `features` to be a non-negative integer");
    }

    try {
        index->val->addItems((float *)f32_data.data, rows, features, ids);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_bfindex_delete_vector(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibBFIndex * index = nullptr;
    size_t label;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], &label)) {
        return erlang::nif::error(env, "expect parameter `label` to be a non-negative integer");
    }

    index->val->deleteVector(label);

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_bfindex_save_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return erlang::nif::error(env, "expecting 2 arguments");
    }

    NifResHNSWLibBFIndex * index = nullptr;
    std::string path;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], path)) {
        return erlang::nif::error(env, "expect parameter `path` to be a string");
    }

    try {
        index->val->saveIndex(path);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    } catch (...) {
        return erlang::nif::error(env, "cannot save index: unknown reason");
    }

    return erlang::nif::ok(env);
}

static ERL_NIF_TERM hnswlib_bfindex_load_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 3) {
        return erlang::nif::error(env, "expecting 3 arguments");
    }

    NifResHNSWLibBFIndex * index = nullptr;
    std::string path;
    size_t max_elements;
    ERL_NIF_TERM ret, error;

    if ((index = NifResHNSWLibBFIndex::get_resource(env, argv[0], error)) == nullptr) {
        return error;
    }
    if (!erlang::nif::get(env, argv[1], path)) {
        return erlang::nif::error(env, "expect parameter `path` to be a string");
    }
    if (!erlang::nif::get(env, argv[2], &max_elements)) {
        return erlang::nif::error(env, "expect parameter `max_elements` to be a non-negative integer");
    }

    try {
        index->val->loadIndex(path, max_elements);
    } catch (std::runtime_error &err) {
        return erlang::nif::error(env, err.what());
    }

    return erlang::nif::ok(env);
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
    {"index_get_items", 3, hnswlib_index_get_items, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"index_get_ids_list", 1, hnswlib_index_get_ids_list, 0},
    {"index_get_ef", 1, hnswlib_index_get_ef, 0},
    {"index_set_ef", 2, hnswlib_index_set_ef, 0},
    {"index_get_num_threads", 1, hnswlib_index_get_num_threads, 0},
    {"index_set_num_threads", 2, hnswlib_index_set_num_threads, 0},
    {"index_save_index", 2, hnswlib_index_save_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"index_load_index", 4, hnswlib_index_load_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
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
    {"bfindex_load_index", 3, hnswlib_bfindex_load_index, ERL_NIF_DIRTY_JOB_IO_BOUND},

    {"float_size", 0, hnswlib_float_size, 0}
};

ERL_NIF_INIT(Elixir.HNSWLib.Nif, nif_functions, on_load, on_reload, on_upgrade, NULL);

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

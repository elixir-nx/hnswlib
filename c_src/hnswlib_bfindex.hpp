#ifndef HNSWLIB_BFINDEX_HPP
#define HNSWLIB_BFINDEX_HPP

#pragma once

#include <iostream>
#include <hnswlib.h>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>
#include <erl_nif.h>
#include "hnswlib_helper.hpp"
#include "nif_utils.hpp"

template<typename dist_t, typename data_t = float>
class BFIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    bool index_inited;
    bool normalize;

    hnswlib::labeltype cur_l;
    hnswlib::BruteforceSearch<dist_t>* alg;
    hnswlib::SpaceInterface<float>* space;


    BFIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        alg = NULL;
        index_inited = false;
    }


    ~BFIndex() {
        delete space;
        if (alg)
            delete alg;
    }


    void init_new_index(const size_t maxElements) {
        if (alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        alg = new hnswlib::BruteforceSearch<dist_t>(space, maxElements);
        index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(float * input, size_t rows, size_t features, const std::vector<uint64_t>& ids) {
        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        for (size_t row = 0; row < rows; row++) {
            uint64_t id = ids.size() ? ids.at(row) : cur_l + row;
            if (!normalize) {
                alg->addPoint((void *)(input + row * features), (size_t)id);
            } else {
                std::vector<float> normalized_vector(dim);
                normalize_vector((float *)(input + row * features), normalized_vector.data());
                alg->addPoint((void *)normalized_vector.data(), (size_t)id);
            }
        }
        cur_l+=rows;
    }


    void deleteVector(size_t label) {
        alg->removePoint(label);
    }


    void saveIndex(const std::string &path_to_index) {
        alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (alg) {
            fprintf(stderr, "Warning: Calling load_index for an already inited index. Old index is being deallocated.\r\n");
            delete alg;
        }
        alg = new hnswlib::BruteforceSearch<dist_t>(space, path_to_index);
        cur_l = alg->cur_element_count;
        index_inited = true;
    }


    bool knnQuery(
        ErlNifEnv * env,
        float* input,
        size_t rows,
        size_t features,
        size_t k,
        // const std::function<bool(hnswlib::labeltype)>& filter,
        ERL_NIF_TERM& out) {
        ErlNifBinary data_l_bin;
        ErlNifBinary data_d_bin;

        hnswlib::labeltype* data_l;
        dist_t* data_d;

        try {
            if (!enif_alloc_binary(sizeof(hnswlib::labeltype) * rows * k, &data_l_bin)) {
                out = hnswlib_error(env, "out of memory for storing labels");
                return false;
            }
            data_l = (hnswlib::labeltype *)data_l_bin.data;

            if (!enif_alloc_binary(sizeof(dist_t) * rows * k, &data_d_bin)) {
                enif_release_binary(&data_l_bin);
                out = hnswlib_error(env, "out of memory for storing distances");
                return false;
            }
            data_d = (dist_t *)data_d_bin.data;

            // CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = nullptr;

            for (size_t row = 0; row < rows; row++) {
                std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
                        (void *)(input + row * features), k, p_idFilter);
                for (int i = k - 1; i >= 0; i--) {
                    auto &result_tuple = result.top();
                    data_d[row * k + i] = result_tuple.first;
                    data_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            }

            ERL_NIF_TERM labels_out = enif_make_binary(env, &data_l_bin);
            ERL_NIF_TERM dists_out = enif_make_binary(env, &data_d_bin);

            ERL_NIF_TERM label_size = enif_make_uint(env, sizeof(hnswlib::labeltype) * 8);
            ERL_NIF_TERM dist_size = enif_make_uint(env, sizeof(dist_t) * 8);
            out = enif_make_tuple7(env,
                hnswlib_atom(env, "ok"),
                labels_out,
                dists_out, 
                enif_make_uint64(env, rows), 
                enif_make_uint64(env, k),
                label_size,
                dist_size);
        } catch (std::runtime_error &err) {
            out = hnswlib_error(env, err.what());

            enif_release_binary(&data_l_bin);
            enif_release_binary(&data_d_bin);
        }

        return true;
    }

    ERL_NIF_TERM hnswlib_atom(ErlNifEnv *env, const char *msg) {
        ERL_NIF_TERM a;
        if (enif_make_existing_atom(env, msg, &a, ERL_NIF_LATIN1)) {
            return a;
        } else {
            return enif_make_atom(env, msg);
        }
    }

    // Helper for returning `{:error, msg}` from NIF.
    ERL_NIF_TERM hnswlib_error(ErlNifEnv *env, const char *msg) {
        ERL_NIF_TERM error_atom = hnswlib_atom(env, "error");
        ERL_NIF_TERM reason;
        unsigned char *ptr;
        size_t len = strlen(msg);
        if ((ptr = enif_make_new_binary(env, len, &reason)) != nullptr) {
            strcpy((char *) ptr, msg);
            return enif_make_tuple2(env, error_atom, reason);
        } else {
            ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
            return enif_make_tuple2(env, error_atom, msg_term);
        }
    }
};

#endif  /* HNSWLIB_BFINDEX_HPP */

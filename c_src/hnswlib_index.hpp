#ifndef HNSWLIB_INDEX_HPP
#define HNSWLIB_INDEX_HPP

#pragma once

#include <iostream>
#include <hnswlib.h>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>
#include <erl_nif.h>
#include <functional>
#include "nif_utils.hpp"

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


inline void assert_true(bool expr, const std::string & msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}


class CustomFilterFunctor: public hnswlib::BaseFilterFunctor {
    std::function<bool(hnswlib::labeltype)> filter;

 public:
    explicit CustomFilterFunctor(const std::function<bool(hnswlib::labeltype)>& f) {
        filter = f;
    }

    bool operator()(hnswlib::labeltype id) {
        return filter(id);
    }
};

template<typename dist_t, typename data_t = float>
class Index {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;


    Index(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }

    size_t indexFileSize() const {
        return appr_alg->indexFileSize();
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
      if (appr_alg) {
          fprintf(stderr, "Warning: Calling load_index for an already inited index. Old index is being deallocated.\r\n");
          delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
      cur_l = appr_alg->cur_element_count;
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


    void addItems(float * input, size_t rows, size_t features, const uint64_t * ids, size_t ids_count, int num_threads = -1, bool replace_deleted = false) {
        if (num_threads <= 0)
            num_threads = num_threads_default;

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        {
            int start = 0;
            if (!ep_added) {
                uint64_t id = ids_count ? ids[0] : (cur_l);
                float* vector_data = input;
                std::vector<float> norm_array(dim);
                if (normalize) {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void *)vector_data, (size_t)id, replace_deleted);
                start = 1;
                ep_added = true;
            }

            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    uint64_t id = ids_count ? ids[row] : (cur_l + row);
                    appr_alg->addPoint((void *)(input + row * dim), (size_t)id, replace_deleted);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float *)(input + row * dim), (norm_array.data() + start_idx));

                    uint64_t id = ids_count ? ids[row] : (cur_l + row);
                    appr_alg->addPoint((void *)(norm_array.data() + start_idx), (size_t)id, replace_deleted);
                    });
            }
            cur_l += rows;
        }
    }


    std::vector<std::vector<data_t>> getDataReturnList(const uint64_t* ids, size_t ids_count) {
        std::vector<std::vector<data_t>> data;
        for (size_t i = 0; i < ids_count; i++) {
            data.push_back(appr_alg->template getDataByLabel<data_t>((size_t)ids[i]));
        }
        return data;
    }


    std::vector<hnswlib::labeltype> getIdsList() {
        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }


    // return true if no error, false otherwise (the `{:error, reason}`-tuple will be saved in `out`)
    bool knnQuery(
        ErlNifEnv * env,
        float * input,
        size_t rows,
        size_t features,
        size_t k,
        int num_threads,
        // const std::function<bool(hnswlib::labeltype)>& filter,
        ERL_NIF_TERM& out) {
        ErlNifBinary data_l_bin;
        ErlNifBinary data_d_bin;

        hnswlib::labeltype* data_l;
        dist_t* data_d;

        if (num_threads <= 0) {
            num_threads = num_threads_default;
        }

        // avoid using threads when the number of searches is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

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

        // CustomFilterFunctor idFilter;
        CustomFilterFunctor* p_idFilter = nullptr;

        try {
            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void *)(input + row * features), k, p_idFilter);
                    if (result.size() != k) {
                        throw std::runtime_error(
                            "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
                    }

                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_d[row * k + i] = result_tuple.first;
                        data_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    float* data = input + row * features;

                    size_t start_idx = threadId * dim;
                    normalize_vector(data, (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    if (result.size() != k) {
                        throw std::runtime_error(
                            "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
                    }

                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_d[row * k + i] = result_tuple.first;
                        data_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
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


    void markDeleted(size_t label) {
        appr_alg->markDelete(label);
    }


    void unmarkDeleted(size_t label) {
        appr_alg->unmarkDelete(label);
    }


    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }


    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
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

template<typename dist_t, typename data_t = float>
class BFIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    bool index_inited;
    bool normalize;
    int num_threads_default;

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

        num_threads_default = std::thread::hardware_concurrency();
    }


    ~BFIndex() {
        delete space;
        if (alg)
            delete alg;
    }


    size_t getMaxElements() const {
        return alg->maxelements_;
    }


    size_t getCurrentCount() const {
        return alg->cur_element_count;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
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


    void addItems(float * input, size_t rows, size_t features, const uint64_t* ids, size_t ids_count) {
        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        for (size_t row = 0; row < rows; row++) {
            uint64_t id = ids_count ? ids[row] : cur_l + row;
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

struct NifResHNSWLibIndex {
    Index<float> * val;
    ErlNifRWLock * rwlock;

    static ErlNifResourceType * type;
    static NifResHNSWLibIndex * allocate_resource(ErlNifEnv * env, ERL_NIF_TERM &error) {
        NifResHNSWLibIndex * res = (NifResHNSWLibIndex *)enif_alloc_resource(NifResHNSWLibIndex::type, sizeof(NifResHNSWLibIndex));
        if (res == nullptr) {
            error = erlang::nif::error(env, "cannot allocate NifResHNSWLibIndex resource");
            return res;
        }
        
        res->rwlock = enif_rwlock_create((char *)"hnswlib.index");
        if (res->rwlock == nullptr) {
            error = erlang::nif::error(env, "cannot allocate rwlock for the NifResHNSWLibIndex resource");
            return res;
        }

        return res;
    }

    static NifResHNSWLibIndex * get_resource(ErlNifEnv * env, ERL_NIF_TERM term, ERL_NIF_TERM &error) {
        NifResHNSWLibIndex * self_res = nullptr;
        if (!enif_get_resource(env, term, NifResHNSWLibIndex::type, (void **)&self_res) || self_res == nullptr || self_res->val == nullptr) {
            error = erlang::nif::error(env, "cannot access NifResHNSWLibIndex resource");
        }
        return self_res;
    }

    static void destruct_resource(ErlNifEnv *env, void *args) {
        auto res = (NifResHNSWLibIndex *)args;
        if (res) {
            if (res->val) {
                delete res->val;
                res->val = nullptr;
            }

            if (res->rwlock) {
                enif_rwlock_destroy(res->rwlock);
                res->rwlock = nullptr;
            }
        }
    }
};

struct NifResHNSWLibBFIndex {
    BFIndex<float> * val;
    ErlNifRWLock * rwlock;

    static ErlNifResourceType * type;
    static NifResHNSWLibBFIndex * allocate_resource(ErlNifEnv * env, ERL_NIF_TERM &error) {
        NifResHNSWLibBFIndex * res = (NifResHNSWLibBFIndex *)enif_alloc_resource(NifResHNSWLibBFIndex::type, sizeof(NifResHNSWLibBFIndex));
        if (res == nullptr) {
            error = erlang::nif::error(env, "cannot allocate NifResHNSWLibBFIndex resource");
            return res;
        }
        
        res->rwlock = enif_rwlock_create((char *)"hnswlib.bfindex");
        if (res->rwlock == nullptr) {
            error = erlang::nif::error(env, "cannot allocate rwlock for the NifResHNSWLibBFIndex resource");
            return res;
        }

        return res;
    }

    static NifResHNSWLibBFIndex * get_resource(ErlNifEnv * env, ERL_NIF_TERM term, ERL_NIF_TERM &error) {
        NifResHNSWLibBFIndex * self_res = nullptr;
        if (!enif_get_resource(env, term, NifResHNSWLibBFIndex::type, (void **)&self_res) || self_res == nullptr || self_res->val == nullptr) {
            error = erlang::nif::error(env, "cannot access NifResHNSWLibBFIndex resource");
        }
        return self_res;
    }

    static void destruct_resource(ErlNifEnv *env, void *args) {
        auto res = (NifResHNSWLibBFIndex *)args;
        if (res) {
            if (res->val) {
                delete res->val;
                res->val = nullptr;
            }

            if (res->rwlock) {
                enif_rwlock_destroy(res->rwlock);
                res->rwlock = nullptr;
            }
        }
    }
};

#endif  /* HNSWLIB_INDEX_HPP */

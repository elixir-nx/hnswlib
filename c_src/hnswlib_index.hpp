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


    void addItems(float * input, size_t rows, size_t features, const std::vector<uint64_t>& ids, int num_threads = -1, bool replace_deleted = false) {
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
                uint64_t id = ids.size() ? ids.at(0) : (cur_l);
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
                    uint64_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void *)(input + row * dim), (size_t)id, replace_deleted);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float *)(input + row * dim), (norm_array.data() + start_idx));

                    uint64_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void *)(norm_array.data() + start_idx), (size_t)id, replace_deleted);
                    });
            }
            cur_l += rows;
        }
    }


    std::vector<std::vector<data_t>> getDataReturnList(const std::vector<uint64_t>& ids) {
        std::vector<std::vector<data_t>> data;
        for (auto id : ids) {
            data.push_back(appr_alg->template getDataByLabel<data_t>((size_t)id));
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


    // py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
    //     std::unique_lock <std::mutex> templock(appr_alg->global);

    //     size_t level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
    //     size_t link_npy_size = 0;
    //     std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

    //     for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
    //         size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
    //         link_npy_offsets[i] = link_npy_size;
    //         if (linkListSize)
    //             link_npy_size += linkListSize;
    //     }

    //     char* data_level0_npy = (char*)malloc(level0_npy_size);
    //     char* link_list_npy = (char*)malloc(link_npy_size);
    //     int* element_levels_npy = (int*)malloc(appr_alg->element_levels_.size() * sizeof(int));

    //     hnswlib::labeltype* label_lookup_key_npy = (hnswlib::labeltype*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
    //     hnswlib::tableint* label_lookup_val_npy = (hnswlib::tableint*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

    //     memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
    //     memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

    //     size_t idx = 0;
    //     for (auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it) {
    //         label_lookup_key_npy[idx] = it->first;
    //         label_lookup_val_npy[idx] = it->second;
    //         idx++;
    //     }

    //     memset(link_list_npy, 0, link_npy_size);

    //     memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
    //     memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

    //     for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
    //         size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
    //         if (linkListSize) {
    //             memcpy(link_list_npy + link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
    //         }
    //     }

    //     py::capsule free_when_done_l0(data_level0_npy, [](void* f) {
    //         delete[] f;
    //         });
    //     py::capsule free_when_done_lvl(element_levels_npy, [](void* f) {
    //         delete[] f;
    //         });
    //     py::capsule free_when_done_lb(label_lookup_key_npy, [](void* f) {
    //         delete[] f;
    //         });
    //     py::capsule free_when_done_id(label_lookup_val_npy, [](void* f) {
    //         delete[] f;
    //         });
    //     py::capsule free_when_done_ll(link_list_npy, [](void* f) {
    //         delete[] f;
    //         });

    //     /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
    //     /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */

    //     return py::dict(
    //         "offset_level0"_a = appr_alg->offsetLevel0_,
    //         "max_elements"_a = appr_alg->max_elements_,
    //         "cur_element_count"_a = (size_t)appr_alg->cur_element_count,
    //         "size_data_per_element"_a = appr_alg->size_data_per_element_,
    //         "label_offset"_a = appr_alg->label_offset_,
    //         "offset_data"_a = appr_alg->offsetData_,
    //         "max_level"_a = appr_alg->maxlevel_,
    //         "enterpoint_node"_a = appr_alg->enterpoint_node_,
    //         "max_M"_a = appr_alg->maxM_,
    //         "max_M0"_a = appr_alg->maxM0_,
    //         "M"_a = appr_alg->M_,
    //         "mult"_a = appr_alg->mult_,
    //         "ef_construction"_a = appr_alg->ef_construction_,
    //         "ef"_a = appr_alg->ef_,
    //         "has_deletions"_a = (bool)appr_alg->num_deleted_,
    //         "size_links_per_element"_a = appr_alg->size_links_per_element_,
    //         "allow_replace_deleted"_a = appr_alg->allow_replace_deleted_,

    //         "label_lookup_external"_a = py::array_t<hnswlib::labeltype>(
    //             { appr_alg->label_lookup_.size() },  // shape
    //             { sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
    //             label_lookup_key_npy,  // the data pointer
    //             free_when_done_lb),

    //         "label_lookup_internal"_a = py::array_t<hnswlib::tableint>(
    //             { appr_alg->label_lookup_.size() },  // shape
    //             { sizeof(hnswlib::tableint) },  // C-style contiguous strides for each index
    //             label_lookup_val_npy,  // the data pointer
    //             free_when_done_id),

    //         "element_levels"_a = py::array_t<int>(
    //             { appr_alg->element_levels_.size() },  // shape
    //             { sizeof(int) },  // C-style contiguous strides for each index
    //             element_levels_npy,  // the data pointer
    //             free_when_done_lvl),

    //         // linkLists_,element_levels_,data_level0_memory_
    //         "data_level0"_a = py::array_t<char>(
    //             { level0_npy_size },  // shape
    //             { sizeof(char) },  // C-style contiguous strides for each index
    //             data_level0_npy,  // the data pointer
    //             free_when_done_l0),

    //         "link_lists"_a = py::array_t<char>(
    //             { link_npy_size },  // shape
    //             { sizeof(char) },  // C-style contiguous strides for each index
    //             link_list_npy,  // the data pointer
    //             free_when_done_ll));
    // }


    // py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
    //     auto params = py::dict(
    //         "ser_version"_a = py::int_(Index<float>::ser_version),  // serialization version
    //         "space"_a = space_name,
    //         "dim"_a = dim,
    //         "index_inited"_a = index_inited,
    //         "ep_added"_a = ep_added,
    //         "normalize"_a = normalize,
    //         "num_threads"_a = num_threads_default,
    //         "seed"_a = seed);

    //     if (index_inited == false)
    //         return py::dict(**params, "ef"_a = default_ef);

    //     auto ann_params = getAnnData();

    //     return py::dict(**params, **ann_params);
    // }


    // static Index<float>* createFromParams(const py::dict d) {
    //     // check serialization version
    //     assert_true(((int)py::int_(Index<float>::ser_version)) >= d["ser_version"].cast<int>(), "Invalid serialization version!");

    //     auto space_name_ = d["space"].cast<std::string>();
    //     auto dim_ = d["dim"].cast<int>();
    //     auto index_inited_ = d["index_inited"].cast<bool>();

    //     Index<float>* new_index = new Index<float>(space_name_, dim_);

    //     /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
    //     /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
    //     new_index->seed = d["seed"].cast<size_t>();

    //     if (index_inited_) {
    //         new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
    //             new_index->l2space,
    //             d["max_elements"].cast<size_t>(),
    //             d["M"].cast<size_t>(),
    //             d["ef_construction"].cast<size_t>(),
    //             new_index->seed);
    //         new_index->cur_l = d["cur_element_count"].cast<size_t>();
    //     }

    //     new_index->index_inited = index_inited_;
    //     new_index->ep_added = d["ep_added"].cast<bool>();
    //     new_index->num_threads_default = d["num_threads"].cast<int>();
    //     new_index->default_ef = d["ef"].cast<size_t>();

    //     if (index_inited_)
    //         new_index->setAnnData(d);

    //     return new_index;
    // }


    // static Index<float> * createFromIndex(const Index<float> & index) {
    //     return createFromParams(index.getIndexParams());
    // }


    // void setAnnData(const py::dict d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
    //     std::unique_lock <std::mutex> templock(appr_alg->global);

    //     assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
    //     assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

    //     appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

    //     assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
    //     assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
    //     assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

    //     appr_alg->maxlevel_ = d["max_level"].cast<int>();
    //     appr_alg->enterpoint_node_ = d["enterpoint_node"].cast<hnswlib::tableint>();

    //     assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
    //     assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
    //     assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
    //     assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
    //     assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

    //     appr_alg->ef_ = d["ef"].cast<size_t>();

    //     assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

    //     auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t < hnswlib::labeltype, py::array::c_style | py::array::forcecast > >();
    //     auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t < hnswlib::tableint, py::array::c_style | py::array::forcecast > >();
    //     auto element_levels_npy = d["element_levels"].cast<py::array_t < int, py::array::c_style | py::array::forcecast > >();
    //     auto data_level0_npy = d["data_level0"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();
    //     auto link_list_npy = d["link_lists"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();

    //     for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
    //         if (label_lookup_val_npy.data()[i] < 0) {
    //             throw std::runtime_error("Internal id cannot be negative!");
    //         } else {
    //             appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
    //         }
    //     }

    //     memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

    //     size_t link_npy_size = 0;
    //     std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

    //     for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
    //         size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
    //         link_npy_offsets[i] = link_npy_size;
    //         if (linkListSize)
    //             link_npy_size += linkListSize;
    //     }

    //     memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

    //     for (size_t i = 0; i < appr_alg->max_elements_; i++) {
    //         size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
    //         if (linkListSize == 0) {
    //             appr_alg->linkLists_[i] = nullptr;
    //         } else {
    //             appr_alg->linkLists_[i] = (char*)malloc(linkListSize);
    //             if (appr_alg->linkLists_[i] == nullptr)
    //                 throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

    //             memcpy(appr_alg->linkLists_[i], link_list_npy.data() + link_npy_offsets[i], linkListSize);
    //         }
    //     }

    //     // process deleted elements
    //     bool allow_replace_deleted = false;
    //     if (d.contains("allow_replace_deleted")) {
    //         allow_replace_deleted = d["allow_replace_deleted"].cast<bool>();
    //     }
    //     appr_alg->allow_replace_deleted_= allow_replace_deleted;

    //     appr_alg->num_deleted_ = 0;
    //     bool has_deletions = d["has_deletions"].cast<bool>();
    //     if (has_deletions) {
    //         for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
    //             if (appr_alg->isMarkedDeleted(i)) {
    //                 appr_alg->num_deleted_ += 1;
    //                 if (allow_replace_deleted) appr_alg->deleted_elements.insert(i);
    //             }
    //         }
    //     }
    // }


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

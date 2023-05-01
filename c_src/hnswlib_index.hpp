#ifndef HNSWLIB_INDEX_HPP
#define HNSWLIB_INDEX_HPP

#include <iostream>
#include <hnswlib.h>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>

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


    // void set_ef(size_t ef) {
    //   default_ef = ef;
    //   if (appr_alg)
    //       appr_alg->ef_ = ef;
    // }


    // void set_num_threads(int num_threads) {
    //     this->num_threads_default = num_threads;
    // }


    // void saveIndex(const std::string &path_to_index) {
    //     appr_alg->saveIndex(path_to_index);
    // }


    // void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
    //   if (appr_alg) {
    //       std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
    //       delete appr_alg;
    //   }
    //   appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
    //   cur_l = appr_alg->cur_element_count;
    //   index_inited = true;
    // }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(float * input, size_t rows, size_t features, uint64_t * ids_ = nullptr, int num_threads = -1, bool replace_deleted = false) {
        if (num_threads <= 0)
            num_threads = num_threads_default;

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<uint64_t> ids;
        if (ids_) {
            ids = std::vector<uint64_t>(ids_, ids_ + rows);
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


    // std::vector<std::vector<data_t>> getDataReturnList(py::object ids_ = py::none()) {
    //     std::vector<size_t> ids;
    //     if (!ids_.is_none()) {
    //         py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
    //         auto ids_numpy = items.request();

    //         if (ids_numpy.ndim == 0) {
    //             throw std::invalid_argument("get_items accepts a list of indices and returns a list of vectors");
    //         } else {
    //             std::vector<size_t> ids1(ids_numpy.shape[0]);
    //             for (size_t i = 0; i < ids1.size(); i++) {
    //                 ids1[i] = items.data()[i];
    //             }
    //             ids.swap(ids1);
    //         }
    //     }

    //     std::vector<std::vector<data_t>> data;
    //     for (auto id : ids) {
    //         data.push_back(appr_alg->template getDataByLabel<data_t>(id));
    //     }
    //     return data;
    // }


    std::vector<hnswlib::labeltype> getIdsList() {
        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
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


    // py::object knnQuery_return_numpy(
    //     py::object input,
    //     size_t k = 1,
    //     int num_threads = -1,
    //     const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
    //     py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
    //     auto buffer = items.request();
    //     hnswlib::labeltype* data_numpy_l;
    //     dist_t* data_numpy_d;
    //     size_t rows, features;

    //     if (num_threads <= 0)
    //         num_threads = num_threads_default;

    //     {
    //         py::gil_scoped_release l;
    //         get_input_array_shapes(buffer, &rows, &features);

    //         // avoid using threads when the number of searches is small:
    //         if (rows <= num_threads * 4) {
    //             num_threads = 1;
    //         }

    //         data_numpy_l = new hnswlib::labeltype[rows * k];
    //         data_numpy_d = new dist_t[rows * k];

    //         // Warning: search with a filter works slow in python in multithreaded mode. For best performance set num_threads=1
    //         CustomFilterFunctor idFilter(filter);
    //         CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

    //         if (normalize == false) {
    //             ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
    //                 std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
    //                     (void*)items.data(row), k, p_idFilter);
    //                 if (result.size() != k)
    //                     throw std::runtime_error(
    //                         "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
    //                 for (int i = k - 1; i >= 0; i--) {
    //                     auto& result_tuple = result.top();
    //                     data_numpy_d[row * k + i] = result_tuple.first;
    //                     data_numpy_l[row * k + i] = result_tuple.second;
    //                     result.pop();
    //                 }
    //             });
    //         } else {
    //             std::vector<float> norm_array(num_threads * features);
    //             ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
    //                 float* data = (float*)items.data(row);

    //                 size_t start_idx = threadId * dim;
    //                 normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

    //                 std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
    //                     (void*)(norm_array.data() + start_idx), k, p_idFilter);
    //                 if (result.size() != k)
    //                     throw std::runtime_error(
    //                         "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
    //                 for (int i = k - 1; i >= 0; i--) {
    //                     auto& result_tuple = result.top();
    //                     data_numpy_d[row * k + i] = result_tuple.first;
    //                     data_numpy_l[row * k + i] = result_tuple.second;
    //                     result.pop();
    //                 }
    //             });
    //         }
    //     }
    //     // py::capsule free_when_done_l(data_numpy_l, [](void* f) {
    //     //     delete[] f;
    //     //     });
    //     // py::capsule free_when_done_d(data_numpy_d, [](void* f) {
    //     //     delete[] f;
    //     //     });

    //     return py::make_tuple(
    //         py::array_t<hnswlib::labeltype>(
    //             { rows, k },  // shape
    //             { k * sizeof(hnswlib::labeltype),
    //               sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
    //             data_numpy_l,  // the data pointer
    //             free_when_done_l),
    //         py::array_t<dist_t>(
    //             { rows, k },  // shape
    //             { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
    //             data_numpy_d,  // the data pointer
    //             free_when_done_d));
    // }


    // void markDeleted(size_t label) {
    //     appr_alg->markDelete(label);
    // }


    // void unmarkDeleted(size_t label) {
    //     appr_alg->unmarkDelete(label);
    // }


    // void resizeIndex(size_t new_size) {
    //     appr_alg->resizeIndex(new_size);
    // }


    // size_t getMaxElements() const {
    //     return appr_alg->max_elements_;
    // }


    // size_t getCurrentCount() const {
    //     return appr_alg->cur_element_count;
    // }
};

#endif  /* HNSWLIB_INDEX_HPP */

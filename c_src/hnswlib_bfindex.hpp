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


    // void addItems(py::object input, py::object ids_ = py::none()) {
    //     py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
    //     auto buffer = items.request();
    //     size_t rows, features;
    //     get_input_array_shapes(buffer, &rows, &features);

    //     if (features != dim)
    //         throw std::runtime_error("Wrong dimensionality of the vectors");

    //     std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

    //     {
    //         for (size_t row = 0; row < rows; row++) {
    //             size_t id = ids.size() ? ids.at(row) : cur_l + row;
    //             if (!normalize) {
    //                 alg->addPoint((void *) items.data(row), (size_t) id);
    //             } else {
    //                 std::vector<float> normalized_vector(dim);
    //                 normalize_vector((float *)items.data(row), normalized_vector.data());
    //                 alg->addPoint((void *) normalized_vector.data(), (size_t) id);
    //             }
    //         }
    //         cur_l+=rows;
    //     }
    // }


    void deleteVector(size_t label) {
        alg->removePoint(label);
    }


    void saveIndex(const std::string &path_to_index) {
        alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
            delete alg;
        }
        alg = new hnswlib::BruteforceSearch<dist_t>(space, path_to_index);
        cur_l = alg->cur_element_count;
        index_inited = true;
    }


    // py::object knnQuery_return_numpy(
    //     py::object input,
    //     size_t k = 1,
    //     const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
    //     py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
    //     auto buffer = items.request();
    //     hnswlib::labeltype *data_numpy_l;
    //     dist_t *data_numpy_d;
    //     size_t rows, features;
    //     {
    //         py::gil_scoped_release l;

    //         get_input_array_shapes(buffer, &rows, &features);

    //         data_numpy_l = new hnswlib::labeltype[rows * k];
    //         data_numpy_d = new dist_t[rows * k];

    //         CustomFilterFunctor idFilter(filter);
    //         CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

    //         for (size_t row = 0; row < rows; row++) {
    //             std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
    //                     (void *) items.data(row), k, p_idFilter);
    //             for (int i = k - 1; i >= 0; i--) {
    //                 auto &result_tuple = result.top();
    //                 data_numpy_d[row * k + i] = result_tuple.first;
    //                 data_numpy_l[row * k + i] = result_tuple.second;
    //                 result.pop();
    //             }
    //         }
    //     }

    //     py::capsule free_when_done_l(data_numpy_l, [](void *f) {
    //         delete[] f;
    //     });
    //     py::capsule free_when_done_d(data_numpy_d, [](void *f) {
    //         delete[] f;
    //     });


    //     return py::make_tuple(
    //             py::array_t<hnswlib::labeltype>(
    //                     { rows, k },  // shape
    //                     { k * sizeof(hnswlib::labeltype),
    //                       sizeof(hnswlib::labeltype)},  // C-style contiguous strides for each index
    //                     data_numpy_l,  // the data pointer
    //                     free_when_done_l),
    //             py::array_t<dist_t>(
    //                     { rows, k },  // shape
    //                     { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
    //                     data_numpy_d,  // the data pointer
    //                     free_when_done_d));
    // }
};

#endif  /* HNSWLIB_BFINDEX_HPP */

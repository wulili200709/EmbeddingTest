#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

FloatArray as_float_array(const py::object& obj, const char* name) {
    try {
        return FloatArray(obj);
    } catch (const std::exception& exc) {
        throw std::invalid_argument(std::string(name) + " must be convertible to a float32 ndarray: " + exc.what());
    }
}

void require_vector(const FloatArray& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D vector");
    }
}

void require_matrix(const FloatArray& arr, const char* name) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D matrix");
    }
}

py::array_t<float> compute_one_proto(const FloatArray& embeddings, const char* name) {
    require_matrix(embeddings, name);
    const py::ssize_t rows = embeddings.shape(0);
    const py::ssize_t cols = embeddings.shape(1);
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }

    py::array_t<float> proto(std::vector<py::ssize_t>{1, cols});
    auto in = embeddings.unchecked<2>();
    auto out = proto.mutable_unchecked<2>();

    double norm_sq = 0.0;
    for (py::ssize_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (py::ssize_t row = 0; row < rows; ++row) {
            sum += static_cast<double>(in(row, col));
        }
        const float mean = static_cast<float>(sum / static_cast<double>(rows));
        out(0, col) = mean;
        norm_sq += static_cast<double>(mean) * static_cast<double>(mean);
    }

    const float norm = static_cast<float>(std::sqrt(norm_sq));
    for (py::ssize_t col = 0; col < cols; ++col) {
        out(0, col) = out(0, col) / norm;
    }
    return proto;
}

float dot_row_vector(const FloatArray& matrix, py::ssize_t row, const FloatArray& vector) {
    require_matrix(matrix, "matrix");
    require_vector(vector, "vector");
    if (matrix.shape(1) != vector.shape(0)) {
        throw std::invalid_argument("matrix/vector dimensions do not match");
    }

    auto mat = matrix.unchecked<2>();
    auto vec = vector.unchecked<1>();
    double sum = 0.0;
    for (py::ssize_t col = 0; col < matrix.shape(1); ++col) {
        sum += static_cast<double>(mat(row, col)) * static_cast<double>(vec(col));
    }
    return static_cast<float>(sum);
}

std::vector<float> similarity_scores(const FloatArray& e, const FloatArray& bank) {
    require_vector(e, "e");
    require_matrix(bank, "bank");
    if (bank.shape(1) != e.shape(0)) {
        throw std::invalid_argument("bank/e dimensions do not match");
    }

    std::vector<float> scores;
    scores.reserve(static_cast<std::size_t>(bank.shape(0)));
    auto emb = e.unchecked<1>();
    auto mat = bank.unchecked<2>();
    for (py::ssize_t row = 0; row < bank.shape(0); ++row) {
        double sum = 0.0;
        for (py::ssize_t col = 0; col < bank.shape(1); ++col) {
            sum += static_cast<double>(mat(row, col)) * static_cast<double>(emb(col));
        }
        scores.push_back(static_cast<float>(sum));
    }
    return scores;
}

float score_topk_arrays(const FloatArray& e, const FloatArray& bank, int k) {
    std::vector<float> scores = similarity_scores(e, bank);
    if (scores.empty()) {
        return -std::numeric_limits<float>::infinity();
    }
    if (k <= 0) {
        throw std::invalid_argument("k must be >= 1");
    }

    const std::size_t count = scores.size();
    const std::size_t keep = static_cast<std::size_t>(k);
    if (keep < count) {
        std::nth_element(scores.begin(), scores.end() - static_cast<std::ptrdiff_t>(keep), scores.end());
        scores.erase(scores.begin(), scores.end() - static_cast<std::ptrdiff_t>(keep));
    }

    double total = 0.0;
    for (float value : scores) {
        total += static_cast<double>(value);
    }
    return static_cast<float>(total / static_cast<double>(scores.size()));
}

}  // namespace

py::tuple compute_prototypes(const py::object& ok_embeddings_obj, const py::object& ng_embeddings_obj) {
    FloatArray ok_embeddings = as_float_array(ok_embeddings_obj, "ok_embeddings");
    FloatArray ng_embeddings = as_float_array(ng_embeddings_obj, "ng_embeddings");
    return py::make_tuple(
        compute_one_proto(ok_embeddings, "ok_embeddings"),
        compute_one_proto(ng_embeddings, "ng_embeddings")
    );
}

float score_topk(const py::object& e_obj, const py::object& bank_obj, int k = 3) {
    FloatArray e = as_float_array(e_obj, "e");
    FloatArray bank = as_float_array(bank_obj, "bank");
    return score_topk_arrays(e, bank, k);
}

py::tuple predict_one(
    const py::object& e_obj,
    const py::object& ok_proto_obj,
    const py::object& ng_proto_obj,
    const py::object& ok_bank_obj,
    const py::object& ng_bank_obj,
    const std::string& mode,
    float margin,
    int topk
) {
    FloatArray e = as_float_array(e_obj, "e");
    FloatArray ok_proto = as_float_array(ok_proto_obj, "ok_proto");
    FloatArray ng_proto = as_float_array(ng_proto_obj, "ng_proto");
    FloatArray ok_bank = as_float_array(ok_bank_obj, "ok_bank");
    FloatArray ng_bank = as_float_array(ng_bank_obj, "ng_bank");

    require_vector(e, "e");
    require_matrix(ok_proto, "ok_proto");
    require_matrix(ng_proto, "ng_proto");
    if (ok_proto.shape(0) <= 0 || ng_proto.shape(0) <= 0) {
        throw std::invalid_argument("prototype arrays must be non-empty");
    }

    float sim_ok = 0.0f;
    float sim_ng = 0.0f;
    if (mode == "proto") {
        sim_ok = dot_row_vector(ok_proto, 0, e);
        sim_ng = dot_row_vector(ng_proto, 0, e);
    } else if (mode == "topk") {
        sim_ok = score_topk_arrays(e, ok_bank, std::min(topk, static_cast<int>(ok_bank.shape(0))));
        sim_ng = score_topk_arrays(e, ng_bank, std::min(topk, static_cast<int>(ng_bank.shape(0))));
    } else {
        throw std::invalid_argument("Unknown score mode: " + mode);
    }

    const float diff = sim_ok - sim_ng;
    const char* pred = diff >= margin ? "OK" : "NG";
    return py::make_tuple(pred, static_cast<double>(diff), static_cast<double>(sim_ok), static_cast<double>(sim_ng));
}

PYBIND11_MODULE(embedding_core, module) {
    module.doc() = "Native embedding prototype and scoring core.";
    module.def("compute_prototypes", &compute_prototypes, py::arg("ok_embeddings"), py::arg("ng_embeddings"));
    module.def("score_topk", &score_topk, py::arg("e"), py::arg("bank"), py::arg("k") = 3);
    module.def(
        "predict_one",
        &predict_one,
        py::arg("e"),
        py::arg("ok_proto"),
        py::arg("ng_proto"),
        py::arg("ok_bank"),
        py::arg("ng_bank"),
        py::arg("mode"),
        py::arg("margin"),
        py::arg("topk")
    );
}

#include "line2Dup.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

class ScopedCoutSilencer {
public:
    ScopedCoutSilencer() : old_buf_(std::cout.rdbuf(sink_.rdbuf())) {}

    ~ScopedCoutSilencer() {
        std::cout.rdbuf(old_buf_);
    }

private:
    std::ostringstream sink_;
    std::streambuf* old_buf_;
};

struct MatHolder {
    py::object owner = py::none();
    cv::Mat mat;
};

class NativeDetector : public line2Dup::Detector {
public:
    NativeDetector(int num_features, const std::vector<int>& T_levels, float weak_threshold, float strong_threshold)
        : line2Dup::Detector(num_features, T_levels, weak_threshold, strong_threshold) {}

    void clear_classes() {
        class_templates.clear();
    }

    void replace_class_templates(
        const std::string& class_id,
        const std::vector<std::vector<line2Dup::Template>>& template_pyramids
    ) {
        class_templates[class_id] = template_pyramids;
    }

    std::vector<std::vector<line2Dup::Template>> export_class_templates(const std::string& class_id) const {
        auto it = class_templates.find(class_id);
        if (it == class_templates.end()) {
            return {};
        }
        return it->second;
    }

    std::vector<line2Dup::Template> export_template_pyramid(const std::string& class_id, int template_id) const {
        auto it = class_templates.find(class_id);
        if (it == class_templates.end()) {
            return {};
        }
        if (template_id < 0 || static_cast<size_t>(template_id) >= it->second.size()) {
            return {};
        }
        return it->second[static_cast<size_t>(template_id)];
    }

    bool has_class(const std::string& class_id) const {
        return class_templates.find(class_id) != class_templates.end();
    }
};

static MatHolder numpy_to_mat(const py::object& obj) {
    MatHolder holder;
    if (obj.is_none()) {
        return holder;
    }

    auto array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!array) {
        throw py::value_error("expected a uint8 numpy array");
    }

    py::buffer_info info = array.request();
    if (info.ndim == 2) {
        holder.owner = py::reinterpret_borrow<py::object>(array);
        holder.mat = cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC1, info.ptr);
        return holder;
    }

    if (info.ndim == 3) {
        const int channels = static_cast<int>(info.shape[2]);
        if (channels == 1 || channels == 3 || channels == 4) {
            holder.owner = py::reinterpret_borrow<py::object>(array);
            holder.mat = cv::Mat(
                static_cast<int>(info.shape[0]),
                static_cast<int>(info.shape[1]),
                CV_MAKETYPE(CV_8U, channels),
                info.ptr
            );
            return holder;
        }
    }

    throw py::value_error("expected an image array with shape HxW or HxWx{1,3,4}");
}

static cv::Mat ensure_source_image(const MatHolder& holder) {
    if (holder.mat.empty()) {
        throw py::value_error("source image is required");
    }
    if (holder.mat.channels() == 1 || holder.mat.channels() == 3) {
        return holder.mat;
    }
    if (holder.mat.channels() == 4) {
        cv::Mat converted;
        cv::cvtColor(holder.mat, converted, cv::COLOR_BGRA2BGR);
        return converted;
    }
    throw py::value_error("source image must be grayscale or BGR");
}

static cv::Mat ensure_mask_image(const MatHolder& holder) {
    if (holder.mat.empty()) {
        return {};
    }
    if (holder.mat.channels() == 1) {
        return holder.mat;
    }
    cv::Mat converted;
    if (holder.mat.channels() == 3) {
        cv::cvtColor(holder.mat, converted, cv::COLOR_BGR2GRAY);
        return converted;
    }
    if (holder.mat.channels() == 4) {
        cv::cvtColor(holder.mat, converted, cv::COLOR_BGRA2GRAY);
        return converted;
    }
    throw py::value_error("mask image must be single-channel or convertible to grayscale");
}

static line2Dup::Feature feature_from_dict(const py::handle& handle) {
    py::dict data = py::reinterpret_borrow<py::dict>(handle);
    line2Dup::Feature feature;
    feature.x = py::cast<int>(data["x"]);
    feature.y = py::cast<int>(data["y"]);
    feature.label = py::cast<int>(data["label"]);
    feature.theta = data.contains("theta") ? py::cast<float>(data["theta"]) : 0.0f;
    return feature;
}

static py::dict feature_to_dict(const line2Dup::Feature& feature) {
    py::dict out;
    out["x"] = feature.x;
    out["y"] = feature.y;
    out["label"] = feature.label;
    out["theta"] = feature.theta;
    return out;
}

static line2Dup::Template template_from_dict(const py::handle& handle) {
    py::dict data = py::reinterpret_borrow<py::dict>(handle);
    line2Dup::Template templ;
    templ.width = py::cast<int>(data["width"]);
    templ.height = py::cast<int>(data["height"]);
    templ.tl_x = py::cast<int>(data["tl_x"]);
    templ.tl_y = py::cast<int>(data["tl_y"]);
    templ.pyramid_level = py::cast<int>(data["pyramid_level"]);

    py::list features = py::cast<py::list>(data["features"]);
    templ.features.reserve(py::len(features));
    for (const py::handle& item : features) {
        templ.features.push_back(feature_from_dict(item));
    }
    return templ;
}

static py::dict template_to_dict(const line2Dup::Template& templ) {
    py::dict out;
    out["width"] = templ.width;
    out["height"] = templ.height;
    out["tl_x"] = templ.tl_x;
    out["tl_y"] = templ.tl_y;
    out["pyramid_level"] = templ.pyramid_level;

    py::list features;
    for (const auto& feature : templ.features) {
        features.append(feature_to_dict(feature));
    }
    out["features"] = features;
    return out;
}

static std::vector<line2Dup::Template> template_pyramid_from_python(const py::handle& handle) {
    py::list levels = py::cast<py::list>(handle);
    std::vector<line2Dup::Template> pyramid;
    pyramid.reserve(py::len(levels));
    for (const py::handle& level : levels) {
        pyramid.push_back(template_from_dict(level));
    }
    return pyramid;
}

static py::list template_pyramid_to_python(const std::vector<line2Dup::Template>& pyramid) {
    py::list out;
    for (const auto& templ : pyramid) {
        out.append(template_to_dict(templ));
    }
    return out;
}

static py::list class_templates_to_python(const std::vector<std::vector<line2Dup::Template>>& template_pyramids) {
    py::list out;
    for (const auto& pyramid : template_pyramids) {
        out.append(template_pyramid_to_python(pyramid));
    }
    return out;
}

static void replace_class_templates_py(NativeDetector& detector, const std::string& class_id, const py::list& template_pyramids) {
    std::vector<std::vector<line2Dup::Template>> parsed;
    parsed.reserve(py::len(template_pyramids));
    for (const py::handle& item : template_pyramids) {
        auto pyramid = template_pyramid_from_python(item);
        if (pyramid.size() != static_cast<size_t>(detector.pyramidLevels())) {
            throw py::value_error("template pyramid level count does not match detector pyramid levels");
        }
        parsed.push_back(std::move(pyramid));
    }
    detector.replace_class_templates(class_id, parsed);
}

static py::list export_class_templates_py(const NativeDetector& detector, const std::string& class_id) {
    return class_templates_to_python(detector.export_class_templates(class_id));
}

static py::list export_template_pyramid_py(const NativeDetector& detector, const std::string& class_id, int template_id) {
    return template_pyramid_to_python(detector.export_template_pyramid(class_id, template_id));
}

static int add_template_py(
    NativeDetector& detector,
    const py::object& source_obj,
    const std::string& class_id,
    const py::object& mask_obj,
    int num_features
) {
    MatHolder source_holder = numpy_to_mat(source_obj);
    MatHolder mask_holder = numpy_to_mat(mask_obj);
    cv::Mat source = ensure_source_image(source_holder);
    cv::Mat mask = ensure_mask_image(mask_holder);

    int template_id = -1;
    {
        py::gil_scoped_release release;
        ScopedCoutSilencer silence;
        template_id = detector.addTemplate(source, class_id, mask, num_features);
    }
    return template_id;
}

static int add_template_rotate_py(
    NativeDetector& detector,
    const std::string& class_id,
    int zero_id,
    float theta_deg,
    float center_x,
    float center_y
) {
    if (!detector.has_class(class_id)) {
        return -1;
    }
    if (zero_id < 0 || zero_id >= detector.numTemplates(class_id)) {
        return -1;
    }

    int template_id = -1;
    {
        py::gil_scoped_release release;
        ScopedCoutSilencer silence;
        template_id = detector.addTemplate_rotate(class_id, zero_id, theta_deg, cv::Point2f(center_x, center_y));
    }
    return template_id;
}

static py::list match_py(
    NativeDetector& detector,
    const py::object& source_obj,
    float threshold,
    const py::object& class_ids_obj,
    const py::object& mask_obj
) {
    MatHolder source_holder = numpy_to_mat(source_obj);
    MatHolder mask_holder = numpy_to_mat(mask_obj);
    cv::Mat source = ensure_source_image(source_holder);
    cv::Mat mask = ensure_mask_image(mask_holder);

    std::vector<std::string> class_ids;
    if (!class_ids_obj.is_none()) {
        class_ids = py::cast<std::vector<std::string>>(class_ids_obj);
    }

    std::vector<line2Dup::Match> matches;
    {
        py::gil_scoped_release release;
        ScopedCoutSilencer silence;
        matches = detector.match(source, threshold, class_ids, mask);
    }

    py::list out;
    for (const auto& match : matches) {
        out.append(py::make_tuple(match.x, match.y, match.similarity, match.class_id, match.template_id));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(line2dup_native, module) {
    module.doc() = "OpenCV-backed native wrapper around the original line2Dup detector";

    py::class_<NativeDetector>(module, "NativeDetector")
        .def(
            py::init<int, const std::vector<int>&, float, float>(),
            py::arg("num_features"),
            py::arg("T_levels"),
            py::arg("weak_threshold") = 30.0f,
            py::arg("strong_threshold") = 60.0f
        )
        .def("clear_classes", &NativeDetector::clear_classes)
        .def("replace_class_templates", &replace_class_templates_py, py::arg("class_id"), py::arg("template_pyramids"))
        .def("export_class_templates", &export_class_templates_py, py::arg("class_id"))
        .def("export_template_pyramid", &export_template_pyramid_py, py::arg("class_id"), py::arg("template_id"))
        .def(
            "add_template",
            &add_template_py,
            py::arg("source"),
            py::arg("class_id"),
            py::arg("object_mask") = py::none(),
            py::arg("num_features") = 0
        )
        .def(
            "add_template_rotate",
            &add_template_rotate_py,
            py::arg("class_id"),
            py::arg("zero_id"),
            py::arg("theta_deg"),
            py::arg("center_x"),
            py::arg("center_y")
        )
        .def(
            "match",
            &match_py,
            py::arg("source"),
            py::arg("threshold"),
            py::arg("class_ids") = py::none(),
            py::arg("mask") = py::none()
        )
        .def("class_ids", &NativeDetector::classIds)
        .def(
            "num_templates",
            [](const NativeDetector& detector, const std::string& class_id) {
                if (class_id.empty()) {
                    return detector.numTemplates();
                }
                return detector.numTemplates(class_id);
            },
            py::arg("class_id") = ""
        );
}

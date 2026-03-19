#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <emmintrin.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using cv::Mat;
using cv::Point;
using cv::Point2d;
using cv::Point2f;
using cv::Rect;
using cv::RotatedRect;
using cv::Scalar;
using cv::Size;
using std::string;
using std::vector;

namespace {

constexpr double VISION_TOLERANCE = 1e-7;
constexpr double D2R = CV_PI / 180.0;
constexpr double R2D = 180.0 / CV_PI;
constexpr int MATCH_CANDIDATE_NUM = 5;

struct TemplateData {
    vector<Mat> pyramid;
    vector<Scalar> templ_mean;
    vector<double> templ_norm;
    vector<double> inv_area;
    vector<bool> result_equal_one;
    bool is_pattern_learned = false;
    int border_color = 0;

    void clear() {
        vector<Mat>().swap(pyramid);
        vector<Scalar>().swap(templ_mean);
        vector<double>().swap(templ_norm);
        vector<double>().swap(inv_area);
        vector<bool>().swap(result_equal_one);
        is_pattern_learned = false;
        border_color = 0;
    }

    void resize(int size) {
        templ_mean.resize(size);
        templ_norm.resize(size, 0.0);
        inv_area.resize(size, 1.0);
        result_equal_one.resize(size, false);
    }
};

struct MatchParameter {
    Point2d pt;
    double match_score = 0.0;
    double match_angle = 0.0;
    double angle_start = 0.0;
    double angle_end = 0.0;
    RotatedRect rect_r;
    bool deleted = false;
    double result_patch[3][3] = {};
    bool pos_on_border = false;

    MatchParameter() = default;
    MatchParameter(Point2f p, double score, double angle) : pt(p), match_score(score), match_angle(angle) {}
};

struct SingleTargetMatch {
    Point2d pt_lt;
    Point2d pt_rt;
    Point2d pt_rb;
    Point2d pt_lb;
    Point2d pt_center;
    double matched_angle = 0.0;
    double match_score = 0.0;
};

bool compare_score_big_to_small(const MatchParameter& lhs, const MatchParameter& rhs) {
    return lhs.match_score > rhs.match_score;
}

bool compare_angle(const std::pair<Point2f, double>& lhs, const std::pair<Point2f, double>& rhs) {
    return lhs.second < rhs.second;
}

struct BlockMax {
    struct Block {
        Rect rect;
        double max_value = 0.0;
        Point max_loc;
    };

    vector<Block> blocks;
    Mat source;

    BlockMax() = default;

    BlockMax(Mat src, Size template_size) : source(std::move(src)) {
        const int block_w = template_size.width * 2;
        const int block_h = template_size.height * 2;
        const int cols = block_w > 0 ? source.cols / block_w : 0;
        const bool has_h_residue = block_w > 0 && (source.cols % block_w != 0);
        const int rows = block_h > 0 ? source.rows / block_h : 0;
        const bool has_v_residue = block_h > 0 && (source.rows % block_h != 0);

        if (cols == 0 || rows == 0) {
            return;
        }

        blocks.resize(cols * rows);
        int count = 0;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                Rect block_rect(x * block_w, y * block_h, block_w, block_h);
                blocks[count].rect = block_rect;
                cv::minMaxLoc(source(block_rect), nullptr, &blocks[count].max_value, nullptr, &blocks[count].max_loc);
                blocks[count].max_loc += block_rect.tl();
                ++count;
            }
        }

        if (has_h_residue && has_v_residue) {
            Rect right_rect(cols * block_w, 0, source.cols - cols * block_w, source.rows);
            Block right_block;
            right_block.rect = right_rect;
            cv::minMaxLoc(source(right_rect), nullptr, &right_block.max_value, nullptr, &right_block.max_loc);
            right_block.max_loc += right_rect.tl();
            blocks.push_back(right_block);

            Rect bottom_rect(0, rows * block_h, cols * block_w, source.rows - rows * block_h);
            Block bottom_block;
            bottom_block.rect = bottom_rect;
            cv::minMaxLoc(source(bottom_rect), nullptr, &bottom_block.max_value, nullptr, &bottom_block.max_loc);
            bottom_block.max_loc += bottom_rect.tl();
            blocks.push_back(bottom_block);
        } else if (has_h_residue) {
            Rect right_rect(cols * block_w, 0, source.cols - cols * block_w, source.rows);
            Block right_block;
            right_block.rect = right_rect;
            cv::minMaxLoc(source(right_rect), nullptr, &right_block.max_value, nullptr, &right_block.max_loc);
            right_block.max_loc += right_rect.tl();
            blocks.push_back(right_block);
        } else if (has_v_residue) {
            Rect bottom_rect(0, rows * block_h, source.cols, source.rows - rows * block_h);
            Block bottom_block;
            bottom_block.rect = bottom_rect;
            cv::minMaxLoc(source(bottom_rect), nullptr, &bottom_block.max_value, nullptr, &bottom_block.max_loc);
            bottom_block.max_loc += bottom_rect.tl();
            blocks.push_back(bottom_block);
        }
    }

    void update_max(const Rect& ignore_rect) {
        for (auto& block : blocks) {
            Rect intersection = ignore_rect & block.rect;
            if (intersection.width == 0 && intersection.height == 0) {
                continue;
            }
            cv::minMaxLoc(source(block.rect), nullptr, &block.max_value, nullptr, &block.max_loc);
            block.max_loc += block.rect.tl();
        }
    }

    void get_max_value_loc(double& max_value, Point& max_loc) const {
        if (blocks.empty()) {
            cv::minMaxLoc(source, nullptr, &max_value, nullptr, &max_loc);
            return;
        }

        int best_index = 0;
        max_value = blocks[0].max_value;
        for (int i = 1; i < static_cast<int>(blocks.size()); ++i) {
            if (blocks[i].max_value >= max_value) {
                best_index = i;
                max_value = blocks[i].max_value;
            }
        }
        max_loc = blocks[best_index].max_loc;
    }
};

inline int hsum_epi32(__m128i value) {
    __m128i tmp = _mm_add_epi32(value, _mm_srli_si128(value, 8));
    tmp = _mm_add_epi32(tmp, _mm_srli_si128(tmp, 4));
    return _mm_cvtsi128_si32(tmp);
}

inline int conv_simd(const unsigned char* kernel, const unsigned char* conv, int length) {
    const int block_size = 16;
    const int block_count = length / block_size;
    __m128i sum_v = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < block_count * block_size; y += block_size) {
        __m128i src_k = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kernel + y));
        __m128i src_c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(conv + y));
        __m128i src_k_l = _mm_unpacklo_epi8(src_k, zero);
        __m128i src_k_h = _mm_unpackhi_epi8(src_k, zero);
        __m128i src_c_l = _mm_unpacklo_epi8(src_c, zero);
        __m128i src_c_h = _mm_unpackhi_epi8(src_c, zero);
        __m128i sum_t = _mm_add_epi32(_mm_madd_epi16(src_k_l, src_c_l), _mm_madd_epi16(src_k_h, src_c_h));
        sum_v = _mm_add_epi32(sum_v, sum_t);
    }

    int sum = hsum_epi32(sum_v);
    for (int y = block_count * block_size; y < length; ++y) {
        sum += kernel[y] * conv[y];
    }
    return sum;
}

class FastPatternMatcher {
public:
    int max_pos = 5;
    double max_overlap = 0.0;
    double score = 0.8;
    double tolerance_angle = 0.0;
    int min_reduced_area = 256;
    bool use_simd = true;
    bool subpixel = false;
    bool stop_layer1 = false;
    bool bitwise_not = false;

    bool learn_pattern(const Mat& templ_gray);
    vector<SingleTargetMatch> match(const Mat& src_gray, double* elapsed_ms = nullptr) const;
    static Mat draw_results(const Mat& src_gray, const vector<SingleTargetMatch>& matches);

private:
    int get_top_layer(const Mat& templ, int min_dst_length) const;
    static bool subpix_estimation(
        const vector<MatchParameter>& vec,
        double* new_x,
        double* new_y,
        double* new_angle,
        double angle_step,
        int max_score_index);
    void match_template(Mat& src, const TemplateData& templ_data, Mat& result, int layer, bool use_simd_path) const;
    static void ccoeff_denominator(Mat& src, const TemplateData& templ_data, Mat& result, int layer);
    static void get_rotated_roi(Mat& src, Size size, Point2f pt_lt, double angle, Mat& roi);
    static Size get_best_rotation_size(Size src_size, Size dst_size, double angle);
    static Point2f rotate_point(Point2f input, Point2f origin, double angle);
    static void filter_with_score(vector<MatchParameter>& matches, double threshold);
    static void sort_pt_with_center(vector<Point2f>& points);
    static void filter_with_rotated_rect(vector<MatchParameter>& matches, int method, double max_overlap_ratio);
    static Point get_next_max_loc(Mat& result, Point max_loc, Size template_size, double& max_value, double max_overlap_ratio);
    static Point get_next_max_loc(
        Mat& result,
        Point max_loc,
        Size template_size,
        double& max_value,
        double max_overlap_ratio,
        BlockMax& block_max);

    TemplateData templ_data_;
    Mat template_image_;
};

bool FastPatternMatcher::learn_pattern(const Mat& templ_gray) {
    if (templ_gray.empty()) {
        return false;
    }

    template_image_ = templ_gray.clone();
    templ_data_.clear();

    const int top_layer = get_top_layer(template_image_, static_cast<int>(std::sqrt(static_cast<double>(min_reduced_area))));
    cv::buildPyramid(template_image_, templ_data_.pyramid, top_layer);
    templ_data_.border_color = cv::mean(template_image_).val[0] < 128 ? 255 : 0;

    const int size = static_cast<int>(templ_data_.pyramid.size());
    templ_data_.resize(size);
    for (int i = 0; i < size; ++i) {
        const double inv_area = 1.0 / static_cast<double>(templ_data_.pyramid[i].rows * templ_data_.pyramid[i].cols);
        Scalar templ_mean;
        Scalar templ_sdv;
        cv::meanStdDev(templ_data_.pyramid[i], templ_mean, templ_sdv);

        double templ_norm = templ_sdv[0] * templ_sdv[0] + templ_sdv[1] * templ_sdv[1] +
                            templ_sdv[2] * templ_sdv[2] + templ_sdv[3] * templ_sdv[3];
        if (templ_norm < DBL_EPSILON) {
            templ_data_.result_equal_one[i] = true;
        }

        templ_norm = std::sqrt(templ_norm);
        templ_norm /= std::sqrt(inv_area);

        templ_data_.inv_area[i] = inv_area;
        templ_data_.templ_mean[i] = templ_mean;
        templ_data_.templ_norm[i] = templ_norm;
    }

    templ_data_.is_pattern_learned = true;
    return true;
}

vector<SingleTargetMatch> FastPatternMatcher::match(const Mat& src_gray, double* elapsed_ms) const {
    vector<SingleTargetMatch> targets;
    if (src_gray.empty() || template_image_.empty() || !templ_data_.is_pattern_learned) {
        return targets;
    }

    if ((template_image_.cols < src_gray.cols && template_image_.rows > src_gray.rows) ||
        (template_image_.cols > src_gray.cols && template_image_.rows < src_gray.rows) ||
        template_image_.size().area() > src_gray.size().area()) {
        return targets;
    }

    auto start = std::chrono::steady_clock::now();

    const int top_layer = get_top_layer(template_image_, static_cast<int>(std::sqrt(static_cast<double>(min_reduced_area))));
    vector<Mat> src_pyr;
    if (bitwise_not) {
        Mat inverted = 255 - src_gray;
        cv::buildPyramid(inverted, src_pyr, top_layer);
    } else {
        cv::buildPyramid(src_gray, src_pyr, top_layer);
    }

    const TemplateData* templ_data = &templ_data_;
    double angle_step = std::atan(2.0 / std::max(templ_data->pyramid[top_layer].cols, templ_data->pyramid[top_layer].rows)) * R2D;
    vector<double> angles;
    if (tolerance_angle < VISION_TOLERANCE) {
        angles.push_back(0.0);
    } else {
        for (double angle = 0.0; angle < tolerance_angle + angle_step; angle += angle_step) {
            angles.push_back(angle);
        }
        for (double angle = -angle_step; angle > -tolerance_angle - angle_step; angle -= angle_step) {
            angles.push_back(angle);
        }
    }

    const int top_src_w = src_pyr[top_layer].cols;
    const int top_src_h = src_pyr[top_layer].rows;
    Point2f center((top_src_w - 1) / 2.0f, (top_src_h - 1) / 2.0f);

    vector<MatchParameter> match_parameters;
    vector<double> layer_score(top_layer + 1, score);
    for (int layer = 1; layer <= top_layer; ++layer) {
        layer_score[layer] = layer_score[layer - 1] * 0.9;
    }

    Size top_template_size = templ_data->pyramid[top_layer].size();
    bool calc_max_by_block =
        (src_pyr[top_layer].size().area() / std::max(1, top_template_size.area()) > 500) && max_pos > 10;

    for (double current_angle : angles) {
        Mat rotated_src;
        Mat rotation_matrix = cv::getRotationMatrix2D(center, current_angle, 1.0);
        Mat result;
        Point max_loc;
        double value = 0.0;
        double max_val = 0.0;
        Size best_size = get_best_rotation_size(src_pyr[top_layer].size(), templ_data->pyramid[top_layer].size(), current_angle);

        float translation_x = (best_size.width - 1) / 2.0f - center.x;
        float translation_y = (best_size.height - 1) / 2.0f - center.y;
        rotation_matrix.at<double>(0, 2) += translation_x;
        rotation_matrix.at<double>(1, 2) += translation_y;

        cv::warpAffine(
            src_pyr[top_layer],
            rotated_src,
            rotation_matrix,
            best_size,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            Scalar(templ_data->border_color));

        match_template(rotated_src, *templ_data, result, top_layer, false);

        if (calc_max_by_block) {
            BlockMax block_max(result, templ_data->pyramid[top_layer].size());
            block_max.get_max_value_loc(max_val, max_loc);
            if (max_val < layer_score[top_layer]) {
                continue;
            }

            match_parameters.emplace_back(Point2f(max_loc.x - translation_x, max_loc.y - translation_y), max_val, current_angle);
            for (int j = 0; j < max_pos + MATCH_CANDIDATE_NUM - 1; ++j) {
                max_loc = get_next_max_loc(result, max_loc, templ_data->pyramid[top_layer].size(), value, max_overlap, block_max);
                if (value < layer_score[top_layer]) {
                    break;
                }
                match_parameters.emplace_back(Point2f(max_loc.x - translation_x, max_loc.y - translation_y), value, current_angle);
            }
        } else {
            cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);
            if (max_val < layer_score[top_layer]) {
                continue;
            }

            match_parameters.emplace_back(Point2f(max_loc.x - translation_x, max_loc.y - translation_y), max_val, current_angle);
            for (int j = 0; j < max_pos + MATCH_CANDIDATE_NUM - 1; ++j) {
                max_loc = get_next_max_loc(result, max_loc, templ_data->pyramid[top_layer].size(), value, max_overlap);
                if (value < layer_score[top_layer]) {
                    break;
                }
                match_parameters.emplace_back(Point2f(max_loc.x - translation_x, max_loc.y - translation_y), value, current_angle);
            }
        }
    }

    std::sort(match_parameters.begin(), match_parameters.end(), compare_score_big_to_small);

    int dst_w = templ_data->pyramid[top_layer].cols;
    int dst_h = templ_data->pyramid[top_layer].rows;
    const bool subpixel_estimation = subpixel;
    const int stop_layer = stop_layer1 ? 1 : 0;

    vector<MatchParameter> all_results;
    for (auto& initial_match : match_parameters) {
        double rotation_angle = -initial_match.match_angle * D2R;
        Point2f pt_lt = rotate_point(initial_match.pt, center, rotation_angle);
        double refine_angle_step = std::atan(2.0 / std::max(dst_w, dst_h)) * R2D;
        initial_match.angle_start = initial_match.match_angle - refine_angle_step;
        initial_match.angle_end = initial_match.match_angle + refine_angle_step;

        if (top_layer <= stop_layer) {
            initial_match.pt = Point2d(pt_lt * ((top_layer == 0) ? 1 : 2));
            all_results.push_back(initial_match);
            continue;
        }

        for (int layer = top_layer - 1; layer >= stop_layer; --layer) {
            refine_angle_step = std::atan(2.0 / std::max(templ_data->pyramid[layer].cols, templ_data->pyramid[layer].rows)) * R2D;
            vector<double> refine_angles;
            double matched_angle = initial_match.match_angle;
            if (tolerance_angle < VISION_TOLERANCE) {
                refine_angles.push_back(0.0);
            } else {
                for (int offset = -1; offset <= 1; ++offset) {
                    refine_angles.push_back(matched_angle + refine_angle_step * offset);
                }
            }

            Point2f src_center((src_pyr[layer].cols - 1) / 2.0f, (src_pyr[layer].rows - 1) / 2.0f);
            vector<MatchParameter> new_matches(refine_angles.size());
            int max_score_index = 0;
            double best_value = -1.0;

            for (int j = 0; j < static_cast<int>(refine_angles.size()); ++j) {
                Mat result;
                Mat rotated_roi;
                double max_value = 0.0;
                Point max_loc;
                get_rotated_roi(src_pyr[layer], templ_data->pyramid[layer].size(), pt_lt * 2, refine_angles[j], rotated_roi);
                match_template(rotated_roi, *templ_data, result, layer, true);
                cv::minMaxLoc(result, nullptr, &max_value, nullptr, &max_loc);
                new_matches[j] = MatchParameter(max_loc, max_value, refine_angles[j]);

                if (new_matches[j].match_score > best_value) {
                    max_score_index = j;
                    best_value = new_matches[j].match_score;
                }

                if (max_loc.x == 0 || max_loc.y == 0 || max_loc.x == result.cols - 1 || max_loc.y == result.rows - 1) {
                    new_matches[j].pos_on_border = true;
                }
                if (!new_matches[j].pos_on_border) {
                    for (int y = -1; y <= 1; ++y) {
                        for (int x = -1; x <= 1; ++x) {
                            new_matches[j].result_patch[x + 1][y + 1] = result.at<float>(max_loc + Point(x, y));
                        }
                    }
                }
            }

            if (new_matches[max_score_index].match_score < layer_score[layer]) {
                break;
            }

            if (subpixel_estimation && layer == 0 && !new_matches[max_score_index].pos_on_border &&
                max_score_index != 0 && max_score_index != 2) {
                double new_x = 0.0;
                double new_y = 0.0;
                double new_angle = 0.0;
                subpix_estimation(new_matches, &new_x, &new_y, &new_angle, refine_angle_step, max_score_index);
                new_matches[max_score_index].pt = Point2d(new_x, new_y);
                new_matches[max_score_index].match_angle = new_angle;
            }

            const double new_match_angle = new_matches[max_score_index].match_angle;
            Point2f padding_lt = rotate_point(pt_lt * 2, src_center, new_match_angle * D2R) - Point2f(3.0f, 3.0f);
            Point2f refined_pt(
                static_cast<float>(new_matches[max_score_index].pt.x + padding_lt.x),
                static_cast<float>(new_matches[max_score_index].pt.y + padding_lt.y));
            refined_pt = rotate_point(refined_pt, src_center, -new_match_angle * D2R);

            if (layer == stop_layer) {
                new_matches[max_score_index].pt = refined_pt * ((stop_layer == 0) ? 1 : 2);
                all_results.push_back(new_matches[max_score_index]);
            } else {
                initial_match.match_angle = new_match_angle;
                initial_match.angle_start = initial_match.match_angle - refine_angle_step / 2.0;
                initial_match.angle_end = initial_match.match_angle + refine_angle_step / 2.0;
                pt_lt = refined_pt;
            }
        }
    }

    filter_with_score(all_results, score);
    dst_w = templ_data->pyramid[stop_layer].cols * ((stop_layer == 0) ? 1 : 2);
    dst_h = templ_data->pyramid[stop_layer].rows * ((stop_layer == 0) ? 1 : 2);

    for (auto& result : all_results) {
        Point2f pt_lt(result.pt);
        double rotation_angle = -result.match_angle * D2R;
        Point2f pt_rt(
            pt_lt.x + dst_w * static_cast<float>(std::cos(rotation_angle)),
            pt_lt.y - dst_w * static_cast<float>(std::sin(rotation_angle)));
        Point2f pt_lb(
            pt_lt.x + dst_h * static_cast<float>(std::sin(rotation_angle)),
            pt_lt.y + dst_h * static_cast<float>(std::cos(rotation_angle)));
        Point2f pt_rb(
            pt_rt.x + dst_h * static_cast<float>(std::sin(rotation_angle)),
            pt_rt.y + dst_h * static_cast<float>(std::cos(rotation_angle)));
        vector<Point2f> corners = {pt_lt, pt_rt, pt_rb, pt_lb};
        result.rect_r = cv::minAreaRect(corners);
    }

    filter_with_rotated_rect(all_results, cv::TM_CCOEFF_NORMED, max_overlap);
    std::sort(all_results.begin(), all_results.end(), compare_score_big_to_small);

    const int width0 = templ_data->pyramid[0].cols;
    const int height0 = templ_data->pyramid[0].rows;
    for (const auto& result : all_results) {
        SingleTargetMatch target;
        double rotation_angle = -result.match_angle * D2R;
        target.pt_lt = result.pt;
        target.pt_rt = Point2d(target.pt_lt.x + width0 * std::cos(rotation_angle), target.pt_lt.y - width0 * std::sin(rotation_angle));
        target.pt_lb = Point2d(target.pt_lt.x + height0 * std::sin(rotation_angle), target.pt_lt.y + height0 * std::cos(rotation_angle));
        target.pt_rb = Point2d(target.pt_rt.x + height0 * std::sin(rotation_angle), target.pt_rt.y + height0 * std::cos(rotation_angle));
        target.pt_center = Point2d(
            (target.pt_lt.x + target.pt_rt.x + target.pt_rb.x + target.pt_lb.x) / 4.0,
            (target.pt_lt.y + target.pt_rt.y + target.pt_rb.y + target.pt_lb.y) / 4.0);
        target.matched_angle = -result.match_angle;
        target.match_score = result.match_score;

        if (target.matched_angle < -180.0) {
            target.matched_angle += 360.0;
        }
        if (target.matched_angle > 180.0) {
            target.matched_angle -= 360.0;
        }

        targets.push_back(target);
        if (static_cast<int>(targets.size()) == max_pos) {
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    if (elapsed_ms != nullptr) {
        *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }
    return targets;
}

Mat FastPatternMatcher::draw_results(const Mat& src_gray, const vector<SingleTargetMatch>& matches) {
    Mat color;
    cv::cvtColor(src_gray, color, cv::COLOR_GRAY2BGR);
    const vector<Scalar> palette = {
        Scalar(0, 255, 0),
        Scalar(0, 255, 255),
        Scalar(255, 0, 0),
        Scalar(255, 0, 255),
        Scalar(255, 255, 0),
    };

    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        const auto& match = matches[i];
        Scalar color_i = palette[i % palette.size()];
        vector<Point> poly = {
            Point(cvRound(match.pt_lt.x), cvRound(match.pt_lt.y)),
            Point(cvRound(match.pt_rt.x), cvRound(match.pt_rt.y)),
            Point(cvRound(match.pt_rb.x), cvRound(match.pt_rb.y)),
            Point(cvRound(match.pt_lb.x), cvRound(match.pt_lb.y)),
        };
        cv::polylines(color, poly, true, color_i, 2, cv::LINE_AA);
        cv::circle(color, Point(cvRound(match.pt_center.x), cvRound(match.pt_center.y)), 3, color_i, -1, cv::LINE_AA);
        char label[128];
        std::snprintf(label, sizeof(label), "#%d s=%.3f a=%.3f", i, match.match_score, match.matched_angle);
        cv::putText(
            color,
            label,
            Point(cvRound(match.pt_lt.x), std::max(12, cvRound(match.pt_lt.y) - 4)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            color_i,
            1,
            cv::LINE_AA);
    }
    return color;
}

int FastPatternMatcher::get_top_layer(const Mat& templ, int min_dst_length) const {
    int top_layer = 0;
    int min_reduce_area_local = min_dst_length * min_dst_length;
    int area = templ.cols * templ.rows;
    while (area > min_reduce_area_local) {
        area /= 4;
        ++top_layer;
    }
    return top_layer;
}

bool FastPatternMatcher::subpix_estimation(
    const vector<MatchParameter>& vec,
    double* new_x,
    double* new_y,
    double* new_angle,
    double angle_step,
    int max_score_index) {
    Mat mat_a(27, 10, CV_64F);
    Mat mat_s(27, 1, CV_64F);

    double x_max = vec[max_score_index].pt.x;
    double y_max = vec[max_score_index].pt.y;
    double theta_max = vec[max_score_index].match_angle;
    int row = 0;
    for (int theta = 0; theta <= 2; ++theta) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                double x_val = x_max + x;
                double y_val = y_max + y;
                double t_val = (theta_max + (theta - 1) * angle_step) * D2R;
                mat_a.at<double>(row, 0) = x_val * x_val;
                mat_a.at<double>(row, 1) = y_val * y_val;
                mat_a.at<double>(row, 2) = t_val * t_val;
                mat_a.at<double>(row, 3) = x_val * y_val;
                mat_a.at<double>(row, 4) = x_val * t_val;
                mat_a.at<double>(row, 5) = y_val * t_val;
                mat_a.at<double>(row, 6) = x_val;
                mat_a.at<double>(row, 7) = y_val;
                mat_a.at<double>(row, 8) = t_val;
                mat_a.at<double>(row, 9) = 1.0;
                mat_s.at<double>(row, 0) = vec[max_score_index + (theta - 1)].result_patch[x + 1][y + 1];
                ++row;
            }
        }
    }

    Mat mat_z = (mat_a.t() * mat_a).inv() * mat_a.t() * mat_s;
    Mat mat_z_t;
    cv::transpose(mat_z, mat_z_t);
    const double* coeff = mat_z_t.ptr<double>(0);
    Mat mat_k1 = (cv::Mat_<double>(3, 3) << (2 * coeff[0]), coeff[3], coeff[4], coeff[3], (2 * coeff[1]), coeff[5], coeff[4], coeff[5], (2 * coeff[2]));
    Mat mat_k2 = (cv::Mat_<double>(3, 1) << -coeff[6], -coeff[7], -coeff[8]);
    Mat delta = mat_k1.inv() * mat_k2;

    *new_x = delta.at<double>(0, 0);
    *new_y = delta.at<double>(1, 0);
    *new_angle = delta.at<double>(2, 0) * R2D;
    return true;
}

void FastPatternMatcher::match_template(Mat& src, const TemplateData& templ_data, Mat& result, int layer, bool use_simd_path) const {
    if (use_simd && use_simd_path) {
        result.create(src.rows - templ_data.pyramid[layer].rows + 1, src.cols - templ_data.pyramid[layer].cols + 1, CV_32FC1);
        result.setTo(0);
        Mat templ = templ_data.pyramid[layer];

        int template_rows = templ.rows;
        for (int r = 0; r < result.rows; ++r) {
            float* result_row = result.ptr<float>(r);
            unsigned char* source_row = src.ptr<unsigned char>(r);
            for (int c = 0; c < result.cols; ++c, ++result_row, ++source_row) {
                unsigned char* template_ptr = templ.ptr<unsigned char>();
                unsigned char* sub_source = source_row;
                for (int t_r = 0; t_r < template_rows; ++t_r, sub_source += src.cols, template_ptr += templ.cols) {
                    *result_row += static_cast<float>(conv_simd(template_ptr, sub_source, templ.cols));
                }
            }
        }
    } else {
        cv::matchTemplate(src, templ_data.pyramid[layer], result, cv::TM_CCORR);
    }

    ccoeff_denominator(src, templ_data, result, layer);
}

void FastPatternMatcher::ccoeff_denominator(Mat& src, const TemplateData& templ_data, Mat& result, int layer) {
    if (templ_data.result_equal_one[layer]) {
        result = Scalar::all(1);
        return;
    }

    Mat sum;
    Mat sqsum;
    cv::integral(src, sum, sqsum, CV_64F);

    double* q0 = reinterpret_cast<double*>(sqsum.data);
    double* q1 = q0 + templ_data.pyramid[layer].cols;
    double* q2 = reinterpret_cast<double*>(sqsum.data + templ_data.pyramid[layer].rows * sqsum.step);
    double* q3 = q2 + templ_data.pyramid[layer].cols;

    double* p0 = reinterpret_cast<double*>(sum.data);
    double* p1 = p0 + templ_data.pyramid[layer].cols;
    double* p2 = reinterpret_cast<double*>(sum.data + templ_data.pyramid[layer].rows * sum.step);
    double* p3 = p2 + templ_data.pyramid[layer].cols;

    int sum_step = sum.data ? static_cast<int>(sum.step / sizeof(double)) : 0;
    int sq_step = sqsum.data ? static_cast<int>(sqsum.step / sizeof(double)) : 0;

    double templ_mean0 = templ_data.templ_mean[layer][0];
    double templ_norm = templ_data.templ_norm[layer];
    double inv_area = templ_data.inv_area[layer];

    for (int i = 0; i < result.rows; ++i) {
        float* result_row = result.ptr<float>(i);
        int idx = i * sum_step;
        int idx2 = i * sq_step;
        for (int j = 0; j < result.cols; ++j, ++idx, ++idx2) {
            double num = result_row[j];
            double t = 0.0;
            double wnd_mean2 = 0.0;
            double wnd_sum2 = 0.0;

            t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
            wnd_mean2 += t * t;
            num -= t * templ_mean0;
            wnd_mean2 *= inv_area;

            t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
            wnd_sum2 += t;

            double diff2 = std::max(wnd_sum2 - wnd_mean2, 0.0);
            if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * wnd_sum2)) {
                t = 0.0;
            } else {
                t = std::sqrt(diff2) * templ_norm;
            }

            if (std::fabs(num) < t) {
                num /= t;
            } else if (std::fabs(num) < t * 1.125) {
                num = num > 0 ? 1.0 : -1.0;
            } else {
                num = 0.0;
            }
            result_row[j] = static_cast<float>(num);
        }
    }
}

void FastPatternMatcher::get_rotated_roi(Mat& src, Size size, Point2f pt_lt, double angle, Mat& roi) {
    double angle_radian = angle * D2R;
    Point2f center((src.cols - 1) / 2.0f, (src.rows - 1) / 2.0f);
    Point2f pt_lt_rotate = rotate_point(pt_lt, center, angle_radian);
    Size padding_size(size.width + 6, size.height + 6);

    Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1);
    rotation_matrix.at<double>(0, 2) -= pt_lt_rotate.x - 3;
    rotation_matrix.at<double>(1, 2) -= pt_lt_rotate.y - 3;
    cv::warpAffine(src, roi, rotation_matrix, padding_size);
}

Size FastPatternMatcher::get_best_rotation_size(Size src_size, Size dst_size, double angle) {
    double angle_radian = angle * D2R;
    Point pt_lt(0, 0);
    Point pt_lb(0, src_size.height - 1);
    Point pt_rb(src_size.width - 1, src_size.height - 1);
    Point pt_rt(src_size.width - 1, 0);
    Point2f center((src_size.width - 1) / 2.0f, (src_size.height - 1) / 2.0f);

    Point2f pt_lt_r = rotate_point(Point2f(pt_lt), center, angle_radian);
    Point2f pt_lb_r = rotate_point(Point2f(pt_lb), center, angle_radian);
    Point2f pt_rb_r = rotate_point(Point2f(pt_rb), center, angle_radian);
    Point2f pt_rt_r = rotate_point(Point2f(pt_rt), center, angle_radian);

    float top_y = std::max(std::max(pt_lt_r.y, pt_lb_r.y), std::max(pt_rb_r.y, pt_rt_r.y));
    float bottom_y = std::min(std::min(pt_lt_r.y, pt_lb_r.y), std::min(pt_rb_r.y, pt_rt_r.y));
    float right_x = std::max(std::max(pt_lt_r.x, pt_lb_r.x), std::max(pt_rb_r.x, pt_rt_r.x));
    float left_x = std::min(std::min(pt_lt_r.x, pt_lb_r.x), std::min(pt_rb_r.x, pt_rt_r.x));

    if (angle > 360.0) {
        angle -= 360.0;
    } else if (angle < 0.0) {
        angle += 360.0;
    }

    if (std::fabs(std::fabs(angle) - 90.0) < VISION_TOLERANCE || std::fabs(std::fabs(angle) - 270.0) < VISION_TOLERANCE) {
        return Size(src_size.height, src_size.width);
    }
    if (std::fabs(angle) < VISION_TOLERANCE || std::fabs(std::fabs(angle) - 180.0) < VISION_TOLERANCE) {
        return src_size;
    }

    double normalized_angle = angle;
    if (normalized_angle > 90.0 && normalized_angle < 180.0) {
        normalized_angle -= 90.0;
    } else if (normalized_angle > 180.0 && normalized_angle < 270.0) {
        normalized_angle -= 180.0;
    } else if (normalized_angle > 270.0 && normalized_angle < 360.0) {
        normalized_angle -= 270.0;
    }

    float h1 = dst_size.width * static_cast<float>(std::sin(normalized_angle * D2R) * std::cos(normalized_angle * D2R));
    float h2 = dst_size.height * static_cast<float>(std::sin(normalized_angle * D2R) * std::cos(normalized_angle * D2R));

    int half_height = static_cast<int>(std::ceil(top_y - center.y - h1));
    int half_width = static_cast<int>(std::ceil(right_x - center.x - h2));
    Size best_size(half_width * 2, half_height * 2);

    bool wrong_size = (dst_size.width < best_size.width && dst_size.height > best_size.height) ||
                      (dst_size.width > best_size.width && dst_size.height < best_size.height) ||
                      (dst_size.area() > best_size.area());
    if (wrong_size) {
        best_size = Size(static_cast<int>(right_x - left_x + 0.5f), static_cast<int>(top_y - bottom_y + 0.5f));
    }
    return best_size;
}

Point2f FastPatternMatcher::rotate_point(Point2f input, Point2f origin, double angle) {
    double width = origin.x * 2.0;
    double height = origin.y * 2.0;
    double y1 = height - input.y;
    double y2 = height - origin.y;

    double x = (input.x - origin.x) * std::cos(angle) - (y1 - origin.y) * std::sin(angle) + origin.x;
    double y = (input.x - origin.x) * std::sin(angle) + (y1 - origin.y) * std::cos(angle) + y2;
    y = -y + height;
    return Point2f(static_cast<float>(x), static_cast<float>(y));
}

void FastPatternMatcher::filter_with_score(vector<MatchParameter>& matches, double threshold) {
    std::sort(matches.begin(), matches.end(), compare_score_big_to_small);
    auto it = std::find_if(matches.begin(), matches.end(), [threshold](const MatchParameter& m) {
        return m.match_score < threshold;
    });
    if (it != matches.end()) {
        matches.erase(it, matches.end());
    }
}

void FastPatternMatcher::sort_pt_with_center(vector<Point2f>& points) {
    Point2f center(0.0f, 0.0f);
    for (const auto& pt : points) {
        center += pt;
    }
    center *= (1.0f / static_cast<float>(points.size()));

    vector<std::pair<Point2f, double>> with_angle(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        with_angle[i].first = points[i];
        with_angle[i].second = std::atan2(points[i].y - center.y, points[i].x - center.x);
    }
    std::sort(with_angle.begin(), with_angle.end(), compare_angle);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        points[i] = with_angle[i].first;
    }
}

void FastPatternMatcher::filter_with_rotated_rect(vector<MatchParameter>& matches, int method, double max_overlap_ratio) {
    int match_size = static_cast<int>(matches.size());
    for (int i = 0; i < match_size - 1; ++i) {
        if (matches[i].deleted) {
            continue;
        }
        for (int j = i + 1; j < match_size; ++j) {
            if (matches[j].deleted) {
                continue;
            }

            vector<Point2f> intersections;
            int intersection_type = cv::rotatedRectangleIntersection(matches[i].rect_r, matches[j].rect_r, intersections);
            if (intersection_type == cv::INTERSECT_NONE) {
                continue;
            }

            int delete_index = -1;
            if (intersection_type == cv::INTERSECT_FULL) {
                if (method == cv::TM_SQDIFF) {
                    delete_index = (matches[i].match_score <= matches[j].match_score) ? j : i;
                } else {
                    delete_index = (matches[i].match_score >= matches[j].match_score) ? j : i;
                }
                matches[delete_index].deleted = true;
                continue;
            }

            if (intersections.size() < 3) {
                continue;
            }

            sort_pt_with_center(intersections);
            double area = cv::contourArea(intersections);
            double ratio = area / std::max(1.0f, matches[i].rect_r.size.area());
            if (ratio > max_overlap_ratio) {
                if (method == cv::TM_SQDIFF) {
                    delete_index = (matches[i].match_score <= matches[j].match_score) ? j : i;
                } else {
                    delete_index = (matches[i].match_score >= matches[j].match_score) ? j : i;
                }
                matches[delete_index].deleted = true;
            }
        }
    }

    matches.erase(std::remove_if(matches.begin(), matches.end(), [](const MatchParameter& m) { return m.deleted; }), matches.end());
}

Point FastPatternMatcher::get_next_max_loc(Mat& result, Point max_loc, Size template_size, double& max_value, double max_overlap_ratio) {
    int start_x = static_cast<int>(max_loc.x - template_size.width * (1 - max_overlap_ratio));
    int start_y = static_cast<int>(max_loc.y - template_size.height * (1 - max_overlap_ratio));
    cv::rectangle(
        result,
        Rect(
            start_x,
            start_y,
            static_cast<int>(2 * template_size.width * (1 - max_overlap_ratio)),
            static_cast<int>(2 * template_size.height * (1 - max_overlap_ratio))),
        Scalar(-1),
        cv::FILLED);
    Point new_max_loc;
    cv::minMaxLoc(result, nullptr, &max_value, nullptr, &new_max_loc);
    return new_max_loc;
}

Point FastPatternMatcher::get_next_max_loc(
    Mat& result,
    Point max_loc,
    Size template_size,
    double& max_value,
    double max_overlap_ratio,
    BlockMax& block_max) {
    int start_x = static_cast<int>(max_loc.x - template_size.width * (1 - max_overlap_ratio));
    int start_y = static_cast<int>(max_loc.y - template_size.height * (1 - max_overlap_ratio));
    Rect ignore_rect(
        start_x,
        start_y,
        static_cast<int>(2 * template_size.width * (1 - max_overlap_ratio)),
        static_cast<int>(2 * template_size.height * (1 - max_overlap_ratio)));
    cv::rectangle(result, ignore_rect, Scalar(-1), cv::FILLED);
    block_max.update_max(ignore_rect);
    Point next_max_loc;
    block_max.get_max_value_loc(max_value, next_max_loc);
    return next_max_loc;
}

string get_string(const cv::CommandLineParser& parser, const char* key) {
    return parser.get<string>(key);
}

int get_int(const cv::CommandLineParser& parser, const char* key) {
    return parser.get<int>(key);
}

double get_double(const cv::CommandLineParser& parser, const char* key) {
    return parser.get<double>(key);
}

bool get_bool(const cv::CommandLineParser& parser, const char* key) {
    return parser.get<bool>(key);
}

}  // namespace

int main(int argc, char** argv) {
    const char* keys =
        "{help h usage ? |      | Show help }"
        "{source s       |      | Source image path }"
        "{template t     |      | Template image path }"
        "{out o          |      | Output overlay path }"
        "{target-num     |5     | Max objects to keep }"
        "{max-overlap    |0.0   | Max overlap ratio }"
        "{score          |0.8   | Similarity threshold }"
        "{tolerance-angle|0.0   | Search angle range in degrees }"
        "{min-reduced-area|256  | Min top pyramid area }"
        "{repeat         |1     | Benchmark repeat count }"
        "{use-simd       |true  | Enable SIMD for refinement }"
        "{subpixel       |false | Enable subpixel estimation }"
        "{bitwise-not    |false | Invert source before search }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || !parser.has("source") || !parser.has("template") || !parser.has("out")) {
        parser.printMessage();
        return 0;
    }

    try {
        string source_path = get_string(parser, "source");
        string template_path = get_string(parser, "template");
        string out_path = get_string(parser, "out");

        Mat src = cv::imread(source_path, cv::IMREAD_GRAYSCALE);
        Mat templ = cv::imread(template_path, cv::IMREAD_GRAYSCALE);
        if (src.empty()) {
            std::cerr << "failed to read source: " << source_path << "\n";
            return 2;
        }
        if (templ.empty()) {
            std::cerr << "failed to read template: " << template_path << "\n";
            return 2;
        }

        FastPatternMatcher matcher;
        matcher.max_pos = get_int(parser, "target-num");
        matcher.max_overlap = get_double(parser, "max-overlap");
        matcher.score = get_double(parser, "score");
        matcher.tolerance_angle = get_double(parser, "tolerance-angle");
        matcher.min_reduced_area = get_int(parser, "min-reduced-area");
        matcher.use_simd = get_bool(parser, "use-simd");
        matcher.subpixel = get_bool(parser, "subpixel");
        matcher.bitwise_not = get_bool(parser, "bitwise-not");

        if (!parser.check()) {
            parser.printErrors();
            return 1;
        }

        if (!matcher.learn_pattern(templ)) {
            std::cerr << "failed to learn template\n";
            return 3;
        }

        int repeat = std::max(1, get_int(parser, "repeat"));
        double total_ms = 0.0;
        vector<SingleTargetMatch> matches;
        for (int i = 0; i < repeat; ++i) {
            double elapsed_ms = 0.0;
            matches = matcher.match(src, &elapsed_ms);
            total_ms += elapsed_ms;
        }

        double avg_ms = total_ms / static_cast<double>(repeat);
        std::cout << "repeat=" << repeat << "\n";
        std::cout << "avg_match_ms=" << avg_ms << "\n";
        std::cout << "matches=" << matches.size() << "\n";
        for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
            const auto& match = matches[i];
            std::cout << "[" << i << "] "
                      << "score=" << match.match_score
                      << " angle=" << match.matched_angle
                      << " center_x=" << match.pt_center.x
                      << " center_y=" << match.pt_center.y
                      << "\n";
        }

        Mat overlay = FastPatternMatcher::draw_results(src, matches);
        if (!cv::imwrite(out_path, overlay)) {
            std::cerr << "failed to write output: " << out_path << "\n";
            return 4;
        }
        std::cout << "saved=" << out_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}

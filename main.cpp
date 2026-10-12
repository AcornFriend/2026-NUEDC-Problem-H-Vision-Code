#include "maix_basic.hpp"
#include "maix_peripheral.hpp"
#include "maix_uart.hpp"
#include "maix_pinmap.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image.hpp"
#include "maix_image_cv.hpp"
#include "maix_touchscreen.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

using namespace maix;
/**
 * @file    My_Project.cpp
 * @brief   基于红色检测的小球位置识别
 * @author  坚果盆友
 * @date    上辈子
 * @version 1145.14
 *
 * 功能概述：
 *   1. 利用串口发送格式为“[num]”的表示小球位置误差的字符串
 *   2. 利用屏幕按键切换当前工作，应对不同题目
 *
 * 核心原理：
 *   1. 分离图像的RGB通道，用R通道减去G，B通道来增强红色检测精准度，且避免白色区域对基于阈值检测所带来的误判
 *   2. 利用一维卡尔曼滤波对输出结果进行预判和平滑处理，弥补视觉模块处理图像所带来的时间误差
 *
 * 输入/输出：
 *   - 输入：电机倾角
 *   - 输出：小球相对于指定标准位置的距离(左负右正)
 *
 * 硬件依赖：
 *   - A18，A19引脚作为串口
 *   - A20，A21，A22，A23，A26，A27驱动红色LED
 *
 * 注意事项：
 *   - 需使用红色补光灯和绿色小球滑轨
 */
// 检测参数
const int MIN_BALL_WIDTH = 1;
const int MAX_BALL_WIDTH = 20;
const int GRAY_THRESHOLD = 70;

const float OLD_LAST_OFFSET = 5.0f;

// FPS 相关变量
static uint64_t fps_start_time = 0;
static int frame_count = 0;
static float fps = 0.0f;
static const float FPS_UPDATE_INTERVAL = 1.0f;

// 触摸屏相关变量
int tx, ty;

// 屏幕按键相关变量
float mid_offset = 0;
int8_t cm = 0;
int8_t mm = 0;
int8_t* set_bit = &cm;


static const uint16_t add_x = 500;
static const uint8_t add_y = 10;
static const uint8_t add_w = 40;
static const uint8_t add_h = 20;

static const uint16_t reduce_x = 500;
static const uint8_t reduce_y = 40;
static const uint8_t reduce_w = 40;
static const uint8_t reduce_h = 20;

static const uint16_t set_bit_x = 560;
static const uint8_t set_bit_y = 40;
static const uint8_t set_bit_w = 40;
static const uint8_t set_bit_h = 20;

static const uint16_t mode_x = 300;
static const uint8_t mode_y = 60;
static const uint8_t mode_w = 40;
static const uint8_t mode_h = 20;

const int TOUCH_HEIGHT = 470;
const int IMAGE_HEIGHT = 100;
const float CENTER_RATIO = 1.0f;

// mode切换
uint8_t mode = 0;

// 串口
// ======== 发送偏移量 ==========
void send_scaled_value(peripheral::uart::UART& uart, float value) {
    char buf[32];
    int scaled = (int)(value * 100.0f) + 4000;//提前说好的，把需发送数值进行这样的处理，将有正负的小数转化成恒等长字符串
    snprintf(buf, sizeof(buf), "[%04d]", scaled);
    uart.write_str(buf);
}

// ======== 串口调试使用 ==========
void send_scaled_a(peripheral::uart::UART& uart, float value) {
    char buf[32];
    int scaled = (int)(value * 100.0f) + 4000;
    snprintf(buf, sizeof(buf), "<%04d>", scaled);
    uart.write_str(buf);
}

// ======== 解析[float]格式 ========
bool parse_float_frame(const std::string& frame, float& value) {
    if (frame.length() < 3) return false;
    if (frame[0] != '[' || frame[frame.length() - 1] != ']') return false;
    std::string content = frame.substr(1, frame.length() - 2);
    if (sscanf(content.c_str(), "%f", &value) == 1) {
        return true;
    }
    return false;
}

// ========== 从串口读取最后一个完整帧 =========
bool get_last_frame(peripheral::uart::UART& uart, std::string& last_frame) {
    std::string buffer;
    char temp[128];
    int len;
    while ((len = uart.read((uint8_t*)temp, sizeof(temp) - 1)) > 0) {
        temp[len] = '\0';
        buffer += std::string(temp, len);
    }
    if (buffer.empty()) return false;
    size_t end_pos = buffer.rfind(']');
    if (end_pos == std::string::npos) return false;
    size_t start_pos = buffer.rfind('[', end_pos);
    if (start_pos == std::string::npos) return false;
    last_frame = buffer.substr(start_pos, end_pos - start_pos + 1);
    return true;
}

// ==================== 一维卡尔曼滤波器 ====================
class KalmanFilter1D {
private:
    float x_;
    float p_;
    float q_;
    float r_;
    bool initialized_;

public:
    KalmanFilter1D() : x_(0.0f), p_(1.0f), q_(0.1f), r_(1.0f), initialized_(false) {}

    void init(float initial_value, float q = 0.1f, float r = 1.0f) {
        x_ = initial_value;
        p_ = 1.0f;
        q_ = q;
        r_ = r;
        initialized_ = true;
    }

    float predict() {
        if (!initialized_) return 0.0f;
        x_ = x_;
        p_ = p_ + q_;
        return x_;
    }

    float update(float measurement) {
        if (!initialized_) {
            init(measurement);
            return measurement;
        }
        float x_pred = x_;
        float p_pred = p_ + q_;
        float k = p_pred / (p_pred + r_);
        x_ = x_pred + k * (measurement - x_pred);
        p_ = (1 - k) * p_pred;
        return x_;
    }

    float get() const { return x_; }

    void reset() {
        initialized_ = false;
        x_ = 0.0f;
        p_ = 1.0f;
    }

    bool is_initialized() const { return initialized_; }
};

// ==================== 边界检测 ==========================
struct PreciseBall {
    int center_x;
    int center_y;
    int width;
    float confidence;
};

// ==================== 映射触摸屏y坐标 ====================
int map_touch_to_image_y(int touch_y) {
    float image_y = (float)touch_y * IMAGE_HEIGHT / TOUCH_HEIGHT;
    int offset = (int)(IMAGE_HEIGHT * (1 - CENTER_RATIO) / 2);
    int result = (int)image_y + offset;
    if (result < 0) result = 0;
    if (result >= IMAGE_HEIGHT) result = IMAGE_HEIGHT - 1;
    return result;
}

PreciseBall find_precise_ball(const std::vector<float>& red_line, int raw_x, int roi_center_y, int roi_y_start, int roi_height) {
    PreciseBall result;
    result.center_x = raw_x;
    result.center_y = roi_center_y;
    result.width = 24;
    result.confidence = 0.0f;

    int search_radius = 35;
    int start_x = std::max(0, raw_x - search_radius);
    int end_x = std::min((int)red_line.size() - 1, raw_x + search_radius);

    float max_val = -1.0f;
    int peak_idx = raw_x;
    for (int i = start_x; i <= end_x; i++) {
        if (red_line[i] > max_val) {
            max_val = red_line[i];
            peak_idx = i;
        }
    }

    if (max_val < 10.0f) {
        return result;
    }

    float threshold = max_val * 0.4f;
    int left_bound = peak_idx;
    for (int i = peak_idx; i >= start_x; i--) {
        if (red_line[i] < threshold) {
            left_bound = i;
            break;
        }
        left_bound = i;
    }

    int right_bound = peak_idx;
    for (int i = peak_idx; i <= end_x; i++) {
        if (red_line[i] < threshold) {
            right_bound = i;
            break;
        }
        right_bound = i;
    }

    int new_width = right_bound - left_bound + 1;
    if (new_width < 10 || new_width > 40) {
        result.center_x = raw_x;
        result.width = 24;
        result.confidence = 0.3f;
        return result;
    }

    result.center_x = (left_bound + right_bound) / 2;
    result.width = new_width;
    result.confidence = max_val / 255.0f;
    return result;
}

// ==================== 像素与物理值转换 ====================
const float CAMERA_HEIGHT_MM = 130.0f;
const float GROUND_HALF_MM = 130.0f;
const int IMAGE_CENTER = 315;
const float MAX_ANGLE = atan2f(GROUND_HALF_MM, CAMERA_HEIGHT_MM);

float pixel_to_physical_tilted(int offset_pixel, float theta_deg) {
    float theta_rad = theta_deg * M_PI / 180.0f;
    float cos_theta = cosf(theta_rad);
    float sin_theta = sinf(theta_rad);
    float u = (float)offset_pixel / IMAGE_CENTER;
    float angle_rad = u * MAX_ANGLE;
    float sin_a = sinf(angle_rad);
    float tan_a = tanf(angle_rad);
    float d = 0.0f;
    if (fabs(sin_a) > 1e-6) {
        float numerator = CAMERA_HEIGHT_MM * tan_a + GROUND_HALF_MM;
        float denominator = cos_theta + sin_theta * tan_a;
        if (fabs(denominator) > 1e-6) {
            d = numerator / denominator;
        } else {
            return 0.0f;
        }
    } else {
        d = GROUND_HALF_MM / cos_theta;
    }
    float d_center = GROUND_HALF_MM / cos_theta;
    return (d - d_center) * cos_theta;
}

// =========== 物理值转换像素值 ===========
int physical_to_pixel_tilted(float physical_mm, float theta_deg) {
    float theta_rad = theta_deg * M_PI / 180.0f;
    float cos_theta = cosf(theta_rad);
    float sin_theta = sinf(theta_rad);
    float d_center = GROUND_HALF_MM / cos_theta;
    float d = d_center + physical_mm / cos_theta;
    float target_x = -GROUND_HALF_MM + d * cos_theta;
    float target_y = d * sin_theta;
    float angle_rad = atan2f(target_x, CAMERA_HEIGHT_MM - target_y);
    float u = angle_rad / MAX_ANGLE;
    int offset = (int)(u * IMAGE_CENTER);
    if (offset < -IMAGE_CENTER) offset = -IMAGE_CENTER;
    if (offset > IMAGE_CENTER) offset = IMAGE_CENTER;
    return offset;
}

int _main(int argc, char* argv[])
{
    // 1. 初始化
    // ----- 摄像头 -------
    camera::Camera cam(540, 100, image::Format::FMT_BGR888, NULL, 60);
    cam.exp_mode(maix::camera::AeMode::Manual);
    cam.exposure(33333);
    cam.gain(100);

    // ----- 显示屏 -------
    display::Display disp;
    
    // ----- 串口 --------
    peripheral::pinmap::set_pin_function("A19", "UART1_TX");
    peripheral::pinmap::set_pin_function("A18", "UART1_RX");
    peripheral::uart::UART uart("/dev/ttyS0", 115200);

    // ----- 触摸屏 -------
    touchscreen::TouchScreen ts;
    bool last_pressed = false;

    // ----- 串口读取 -----
    std::string last_frame;
    std::string label;
    float value = 0.0f;

    // 获取图像尺寸
    image::Image *img_test = cam.read();
    if (!img_test) {
        log::error("Failed to read test image");
        return -1;
    }

    int img_h = img_test->height();
    int img_w = img_test->width();
    delete img_test;

    // ROI设置：屏幕中央横向区域（占画面高度的 20%）
    float devite = 0.125f;
    int roi_y_start = (int)(img_h * (0.5f - devite));
    int roi_y_end = (int)(img_h * (0.5f + devite));
    int roi_height = roi_y_end - roi_y_start;

    // ----- FPS ---------
    fps_start_time = time::ticks_ms();

    // ----- GPIO --------
    err::Err A26 = peripheral::pinmap::set_pin_function("A26", "GPIOA26");
    err::check_raise(A26, "set pin function failed");
    err::Err A27 = peripheral::pinmap::set_pin_function("A27", "GPIOA27");
    err::check_raise(A27, "set pin function failed");
    err::Err P20 = peripheral::pinmap::set_pin_function("P20", "GPIOP20");
    err::check_raise(P20, "set pin function failed");
    err::Err P21 = peripheral::pinmap::set_pin_function("P21", "GPIOP21");
    err::check_raise(P21, "set pin function failed");
      err::Err P22 = peripheral::pinmap::set_pin_function("P20", "GPIOP20");
    err::check_raise(P22, "set pin function failed");
    err::Err P23 = peripheral::pinmap::set_pin_function("P21", "GPIOP21");
    err::check_raise(P23, "set pin function failed");

    peripheral::gpio::GPIO LEI_L1_L("GPIOA26", peripheral::gpio::Mode::OUT, peripheral::gpio::Pull::PULL_DOWN);
    peripheral::gpio::GPIO LEI_L1_H("GPIOA27", peripheral::gpio::Mode::OUT, peripheral::gpio::Pull::PULL_UP);
    
    peripheral::gpio::GPIO LEI_R1_L("GPIOP20", peripheral::gpio::Mode::OUT, peripheral::gpio::Pull::PULL_DOWN);
    peripheral::gpio::GPIO LEI_R1_H("GPIOP21", peripheral::gpio::Mode::OUT, peripheral::gpio::Pull::PULL_UP);

    peripheral::gpio::GPIO LEI_R2_L("GPIOP22", peripheral::gpio::Mode::OUT, peripheral::gpio::Pull::PULL_DOWN);
    peripheral::gpio::GPIO LEI_R2_H("GPIOP23", peripheral::gpio::Mode::OUT, peripheral::gpio::Pull::PULL_UP);

    // ----- 卡尔曼滤波器 -------
    KalmanFilter1D kalman_x;
    float lost_frame_count = 0;
    float last_value = NULL;
    bool has_valid_measurement = false;
    const int MAX_LOST_FRAMES = 30;
    const int EXP_MAX_MOVE = 118.5f;

    float last_set_point = 0;

    log::info("Program started, image size: %dx%d", img_w, img_h);
    
    //while主函数
    while (!app::need_exit()) {
        
        //串口接收信息
        if (get_last_frame(uart, last_frame)) {
            if (parse_float_frame(last_frame, value)) {
                if(label == ""){
                    log::info("Recevied");
                    char re_uart[10];
                    snprintf(re_uart, sizeof(re_uart), "Re:%.02f", value);
                    // uart.write_str(re_uart);
                }
            }
        }

        uint64_t frame_start = time::ticks_ms();

        // 1. 读取图像 (BGR888)
        image::Image *img_maix = cam.read();
        if (!img_maix) {
            time::sleep(10);
            continue;
        }

        // 2. 转换为 OpenCV Mat (BGR 格式)
        cv::Mat img_cv;
        image::image2cv(*img_maix, img_cv);

        // 3. 提取红色通道 (BGR 中 R 是第 3 通道，索引 2)
        std::vector<cv::Mat> channels;
        cv::split(img_cv, channels);
        cv::Mat red_channel = channels[2];  // 红色通道

        // 4. 提取 ROI 区域 (在红色通道上)
        cv::Mat roi = red_channel(cv::Range(roi_y_start, roi_y_end), cv::Range::all());

        // 5. 将 ROI 压缩成一行（垂直方向求平均）
        cv::Mat red_line_mat;
        cv::reduce(roi, red_line_mat, 0, cv::REDUCE_AVG);
        std::vector<float> red_line;
        red_line_mat.convertTo(red_line, CV_32F);

        // 6. 对红色曲线进行平滑处理
        int kernel_size = 3;
        std::vector<float> kernel(kernel_size, 1.0f / kernel_size);
        std::vector<float> red_line_smooth(red_line.size(), 0);

        for (size_t i = 0; i < red_line.size(); i++) {
            float sum = 0.0f;
            int count = 0;
            for (int k = -kernel_size/2; k <= kernel_size/2; k++) {
                int idx = (int)i + k;
                if (idx >= 0 && idx < (int)red_line.size()) {
                    sum += red_line[idx] * kernel[k + kernel_size/2];
                    count++;
                }
            }
            red_line_smooth[i] = sum;
        }

        // 7. 平方增强（突出红色峰值）
        float max_val = 0.0f;
        for (auto &val : red_line_smooth) {
            if (val > max_val) max_val = val;
        }
        if (max_val > 1e-6) {
            for (auto &val : red_line_smooth) {
                float normalized = val / max_val;
                val = normalized * normalized * max_val;
            }
        }

        // 8. 检测红色区域（红色球体检测）
        // 球体的特点是红色通道强度高，所以用 > 阈值检测
        float bg_mean = 0.0f;
        for (float v : red_line_smooth) {
            bg_mean += v;
        }
        bg_mean /= red_line_smooth.size();

        // 红色阈值：背景均值 + 偏移量（红色比背景亮）
        float red_threshold = bg_mean + GRAY_THRESHOLD;
        std::vector<bool> ball_mask(red_line_smooth.size(), false);
        for (size_t i = 0; i < red_line_smooth.size(); i++) {
            ball_mask[i] = (red_line_smooth[i] > red_threshold);
        }

        // 9. 找连通区域（带强度计算）
        struct BallRegion {
            int start, end, width;
            float intensity;
        };

        std::vector<BallRegion> ball_regions;
        size_t i = 0;
        while (i < ball_mask.size()) {
            if (ball_mask[i]) {
                size_t start = i;
                while (i < ball_mask.size() && ball_mask[i]) {
                    i++;
                }
                size_t end = i - 1;
                int width = (int)(end - start + 1);
                if (width >= MIN_BALL_WIDTH && width <= MAX_BALL_WIDTH) {
                    float max_intensity = 0.0f;
                    for (int j = (int)start; j <= (int)end; j++) {
                        if (red_line_smooth[j] > max_intensity) {
                            max_intensity = red_line_smooth[j];
                        }
                    }
                    ball_regions.push_back({(int)start, (int)end, width, max_intensity});
                }
            } else {
                i++;
            }
        }

        // 找出红色强度最大的球体
        BallRegion best_ball;
        bool has_ball = false;
        float best_intensity = -1.0f;
        for (auto &region : ball_regions) {
            if (region.intensity > best_intensity) {
                best_intensity = region.intensity;
                best_ball = region;
                has_ball = true;
            }
        }

        // 10. 绘制结果（只绘制最大强度的球体）
        // 绘制 ROI 区域
        img_maix->draw_rect(0, roi_y_start, img_w, roi_height, image::COLOR_GREEN, 2);
        int center_y = roi_y_start + roi_height / 2;
        int radius = best_ball.width / 2;
        switch (mode)
            {
            case 0 :
                mid_offset =(float)(cm + 0.1*mm);

                if (has_ball) {

                    int raw_x = (best_ball.start + best_ball.end) / 2;
                    

                    // ===== 精准边界检测（利用已知宽度 24 像素）=====
                    int search_radius = 35;
                    int start_x = std::max(0, raw_x - search_radius);
                    int end_x = std::min((int)red_line_smooth.size() - 1, raw_x + search_radius);

                    // 1. 在搜索范围内找到红色通道的峰值位置
                    float max_val = -1.0f;
                    int peak_idx = raw_x;
                    for (int i = start_x; i <= end_x; i++) {
                        if (red_line_smooth[i] > max_val) {
                            max_val = red_line_smooth[i];
                            peak_idx = i;
                        }
                    }

                    // 2. 从峰值向左右两边寻找边界（阈值：峰值强度的 40%）
                    float threshold = max_val * 0.4f;
                    int left_bound = peak_idx;
                    for (int i = peak_idx; i >= start_x; i--) {
                        if (red_line_smooth[i] < threshold) {
                            left_bound = i;
                            break;
                        }
                        left_bound = i;
                    }

                    int right_bound = peak_idx;
                    for (int i = peak_idx; i <= end_x; i++) {
                        if (red_line_smooth[i] < threshold) {
                            right_bound = i;
                            break;
                        }
                        right_bound = i;
                    }

                    // 3. 计算精准中心位置
                    int precise_width = right_bound - left_bound + 1;
                    int precise_x = (left_bound + right_bound) / 2;

                    // 如果检测到的宽度与已知宽度差异太大，使用原始值
                    if (precise_width < 10 || precise_width > 40) {
                        precise_x = raw_x;
                        precise_width = best_ball.width;
                    }
                    // ===== 卡尔曼滤波（使用精准的 x 坐标）=====
                    raw_x = (best_ball.start + best_ball.end) / 2;
                    if (last_value == NULL ){
                        last_value = raw_x;
                    }
                    if (!has_valid_measurement && abs(last_value - raw_x) > EXP_MAX_MOVE) {
                        kalman_x.init((float)precise_x, 1.5f, 0.7f);
                        has_valid_measurement = true;
                    }
                    float smooth_x = kalman_x.update((float)precise_x);
                    int draw_x = (int)smooth_x;

                    // ===== 计算偏移量 =====
                    int screen_center_x = img_w / 2;
                    int offset_x = draw_x - screen_center_x;
                    float physical_x_mm = pixel_to_physical_tilted(offset_x, value);
                    // 屏幕显示
                    char disp_buf[64];
                    snprintf(disp_buf, sizeof(disp_buf), "Offset: %.02f", physical_x_mm - mid_offset*10);
                    img_maix->draw_string(100, 10, disp_buf, image::COLOR_YELLOW);

                    // 显示精准宽度信息
                    char width_buf[32];
                    snprintf(width_buf, sizeof(width_buf), "W: %d", precise_width);
                    img_maix->draw_string(100, 40, width_buf, image::COLOR_WHITE);

                    // 绘制中心参考线
                    img_maix->draw_line(screen_center_x, 0, screen_center_x, img_h,
                                        image::COLOR_YELLOW, 1);
                    int pixel_mid_offset  = physical_to_pixel_tilted(mid_offset*10, value);
                    img_maix->draw_line(screen_center_x + pixel_mid_offset, 0, screen_center_x + pixel_mid_offset, img_h,
                                        image::COLOR_GREEN, 1);

                    if (radius > 0) {
                        // ----- 蓝色：原始检测位置（滤波前）-----
                        img_maix->draw_circle(raw_x, center_y, radius,
                                            image::Color::from_rgb(0, 0, 255), 2);
                        img_maix->draw_circle(raw_x, center_y, 3,
                                            image::Color::from_rgb(0, 0, 255), -1);

                        // ----- 黄色：固定 24 像素宽度的圆（最终结果）-----
                        // 使用已知宽度 24，半径为 12
                        int fixed_radius = 12;
                        img_maix->draw_circle(draw_x, center_y, fixed_radius,
                                            image::Color::from_rgb(255, 255, 0), 3);
                        img_maix->draw_circle(draw_x, center_y, 3,
                                            image::Color::from_rgb(255, 0, 0), -1);
                    }
                    // ------ 串口发送 -------

                    float offset_mm = physical_x_mm - mid_offset * 10.0f;
    
                // 计算加速度 (mm/s²)
                // 方法1：基于速度变化（需要存储上一帧速度）
                // 方法2：基于位置二阶差分
                static float prev_velocity = 0.0f;
                static float prev_offset = 0.0f;
                static uint64_t prev_time = 0;
                
                uint64_t current_time = time::ticks_ms();
                float dt = (float)(current_time - prev_time) / 1000.0f;  // 秒
                
                if (dt > 0.001f && prev_time > 0) {
                    float velocity = (offset_mm - prev_offset) / dt;  // mm/s
                    float acceleration = (velocity - prev_velocity) / dt;  // mm/s²
                    prev_velocity = velocity;
                    send_scaled_a(uart, acceleration); 
                    // 发送数据
                    
                }
                
                               
                prev_offset = offset_mm;
                prev_time = current_time;

                    send_scaled_value(uart, physical_x_mm/10.0f - mid_offset);
                }
                else{
                    lost_frame_count++;
                    
                    if (has_valid_measurement && lost_frame_count <= MAX_LOST_FRAMES) {
                        // 使用预测值
                        float pred_x = kalman_x.predict();
                        int draw_x = (int)pred_x;
                        last_value = draw_x;
                        // 用灰色或虚线显示预测位置
                        img_maix->draw_circle(draw_x, center_y, 12,
                                            image::Color::from_rgb(128, 128, 128), 2);
                        img_maix->draw_string(draw_x - 30, center_y - 30,
                                            "PREDICTED", image::COLOR_WHITE);
                        float offset_physical = pixel_to_physical_tilted(draw_x - img_w / 2, value);
                        send_scaled_value(uart, offset_physical/10.0f);
                        log::info("Lost ball, predicted x: %.1f", offset_physical);
                    } else {
                        // 连续丢失超过阈值，重置滤波器
                        if (has_valid_measurement) {
                            kalman_x.reset();
                            has_valid_measurement = false;
                            log::info("Kalman reset: ball lost for too long");
                        }
                    }
                }
                break;///////////////////////////
            
            case 1:
                static bool finish_on_set = false;
                if(finish_on_set == false){
                    mid_offset = 5;
                }
                else{
                    mid_offset = -5;
                }
                
                const uint16_t on_set_threshold = 250;
                static int16_t frame_on_set = 0;
                bool on_set = false ;
                if (has_ball) {
                    int raw_x = (best_ball.start + best_ball.end) / 2;

                    // ===== 精准边界检测（利用已知宽度 24 像素）=====
                    int search_radius = 35;
                    int start_x = std::max(0, raw_x - search_radius);
                    int end_x = std::min((int)red_line_smooth.size() - 1, raw_x + search_radius);

                    // 1. 在搜索范围内找到红色通道的峰值位置
                    float max_val = -1.0f;
                    int peak_idx = raw_x;
                    for (int i = start_x; i <= end_x; i++) {
                        if (red_line_smooth[i] > max_val) {
                            max_val = red_line_smooth[i];
                            peak_idx = i;
                        }
                    }

                    // 2. 从峰值向左右两边寻找边界（阈值：峰值强度的 40%）
                    float threshold = max_val * 0.4f;
                    int left_bound = peak_idx;
                    for (int i = peak_idx; i >= start_x; i--) {
                        if (red_line_smooth[i] < threshold) {
                            left_bound = i;
                            break;
                        }
                        left_bound = i;
                    }

                    int right_bound = peak_idx;
                    for (int i = peak_idx; i <= end_x; i++) {
                        if (red_line_smooth[i] < threshold) {
                            right_bound = i;
                            break;
                        }
                        right_bound = i;
                    }

                    // 3. 计算精准中心位置
                    int precise_width = right_bound - left_bound + 1;
                    int precise_x = (left_bound + right_bound) / 2;

                    // 如果检测到的宽度与已知宽度差异太大，使用原始值
                    if (precise_width < 10 || precise_width > 40) {
                        precise_x = raw_x;
                        precise_width = best_ball.width;
                    }
                    // ===== 卡尔曼滤波（使用精准的 x 坐标）=====
                    raw_x = (best_ball.start + best_ball.end) / 2;
                    if (last_value == NULL ){
                        last_value = raw_x;
                    }
                    if (!has_valid_measurement && abs(last_value - raw_x) > EXP_MAX_MOVE) {
                        kalman_x.init((float)precise_x, 1.0f, 0.7f);
                        has_valid_measurement = true;
                    }
                    float smooth_x = kalman_x.update((float)precise_x);
                    int draw_x = (int)smooth_x;

                    // ===== 计算偏移量 =====
                    int screen_center_x = img_w / 2;
                    int offset_x = draw_x - screen_center_x;
                    float physical_x_mm = pixel_to_physical_tilted(offset_x, value);

                    // ===== 判断当前偏移量是否可认为已到位 =====
                    if(fabs(mid_offset*10 - physical_x_mm) <= 7){
                        uint64_t now = time::ticks_ms();
                        frame_on_set ++;
                    }
                    else{
                        frame_on_set = 0;
                    }
                    if(frame_on_set > on_set_threshold){
                        finish_on_set = true;
                        mid_offset = -5;
                    }

                    // 屏幕显示
                    char disp_buf[64];
                    snprintf(disp_buf, sizeof(disp_buf), "Offset: %.02f", physical_x_mm - mid_offset*10);
                    img_maix->draw_string(100, 10, disp_buf, image::COLOR_YELLOW);

                    // 显示精准宽度信息
                    char on_set_buf[32];
                    snprintf(on_set_buf, sizeof(on_set_buf), "W: %d", (int)frame_on_set);
                    img_maix->draw_string(100, 40, on_set_buf, image::COLOR_WHITE);

                    // 绘制中心参考线
                    img_maix->draw_line(screen_center_x, 0, screen_center_x, img_h,
                                        image::COLOR_YELLOW, 1);
                    int pixel_mid_offset  = physical_to_pixel_tilted(mid_offset*10, value);
                    img_maix->draw_line(screen_center_x + pixel_mid_offset, 0, screen_center_x + pixel_mid_offset, img_h,
                                        image::COLOR_GREEN, 1);

                    if (radius > 0) {
                        // ----- 蓝色：原始检测位置（滤波前）-----
                        img_maix->draw_circle(raw_x, center_y, radius,
                                            image::Color::from_rgb(0, 0, 255), 2);
                        img_maix->draw_circle(raw_x, center_y, 3,
                                            image::Color::from_rgb(0, 0, 255), -1);

                        // ----- 黄色：固定 24 像素宽度的圆（最终结果）-----
                        // 使用已知宽度 24，半径为 12
                        int fixed_radius = 12;
                        img_maix->draw_circle(draw_x, center_y, fixed_radius,
                                            image::Color::from_rgb(255, 255, 0), 3);
                        img_maix->draw_circle(draw_x, center_y, 3,
                                            image::Color::from_rgb(255, 0, 0), -1);
                    }

                    // 计算加速度 (mm/s²)
                    // 方法1：基于速度变化（需要存储上一帧速度）
                    // 方法2：基于位置二阶差分
                    static float prev_velocity = 0.0f;
                    static float prev_offset = 0.0f;
                    static uint64_t prev_time = 0;
                    
                    uint64_t current_time = time::ticks_ms();
                    float dt = (current_time - prev_time) / 1000.0f;  // 秒
                    
                    if (dt > 0.001f && prev_time > 0) {
                        float velocity = (physical_x_mm - prev_offset) / dt;  // mm/s
                        float acceleration = (velocity - prev_velocity) / dt;  // mm/s²
                        prev_velocity = velocity;
                        
                        // 发送数据
                        send_scaled_a(uart, acceleration);

                    // ------ 串口发送 -------
                    send_scaled_value(uart, physical_x_mm/10.0f - mid_offset);
                    }
                    else{
                        lost_frame_count++;
                        
                        if (has_valid_measurement && lost_frame_count <= MAX_LOST_FRAMES) {
                            // 使用预测值
                            float pred_x = kalman_x.predict();
                            int draw_x = (int)pred_x;
                            last_value = draw_x;
                            // 用灰色或虚线显示预测位置
                            img_maix->draw_circle(draw_x, center_y, 12,
                                                image::Color::from_rgb(128, 128, 128), 2);
                            img_maix->draw_string(draw_x - 30, center_y - 30,
                                                "PREDICTED", image::COLOR_WHITE);
                            float offset_physical = pixel_to_physical_tilted(draw_x - img_w / 2, value);
                            send_scaled_value(uart, offset_physical/10.0f);
                            log::info("Lost ball, predicted x: %.1f", offset_physical);
                        } else {
                            // 连续丢失超过阈值，重置滤波器
                            if (has_valid_measurement) {
                                kalman_x.reset();
                                has_valid_measurement = false;
                                log::info("Kalman reset: ball lost for too long");
                            }
                        }
                    }
                break;///////////////////////////
                }
            }
        

        // 11. 触摸屏读取
        bool pressed;
            if (ts.read(tx, ty, pressed) == err::ERR_NONE) {
                tx = tx *630 /550;
                ty = map_touch_to_image_y(ty);
            if(pressed && !last_pressed){
                if(tx <= add_x + add_w && tx >= add_x && ty >= add_y && ty < add_y + add_h){
                    *set_bit = *set_bit + 1;
                    if(set_bit == &mm){
                        if(mm > 9){
                            mm = 9;
                        }
                    }
                }
                if(tx <= reduce_x + reduce_w && tx >= reduce_x && ty >= reduce_y && ty < reduce_y + reduce_h){
                    *set_bit = *set_bit - 1;
                    if(set_bit == &mm){
                        if(mm < 0){
                            mm = 0;
                        }
                    }
                }
                if(tx <= mode_x + mode_w && tx >= mode_x && ty >= mode_y && ty < mode_y + mode_h){
                    mode = (mode + 1) % 3;
                }
                if(tx <= set_bit_x + set_bit_w && tx >= set_bit_x && ty >= set_bit_y && ty < set_bit_y + set_bit_h){
                    if(set_bit == &cm){
                        set_bit = &mm;
                    }
                    else{
                        set_bit = &cm;
                    }
                }
            }
            last_pressed = pressed;               
        }

        // 12. FPS 计算和显示

        frame_count++;
        uint64_t now = time::ticks_ms();
        if ((now - fps_start_time) >= (uint64_t)(FPS_UPDATE_INTERVAL * 1000)) {
            fps = frame_count * 1000.0f / (now - fps_start_time);
            frame_count = 0;
            fps_start_time = now;
        }

        char fps_text[32];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
        img_maix->draw_string(10, 10, fps_text, image::COLOR_WHITE);

        char ball_count_text[32];
        snprintf(ball_count_text, sizeof(ball_count_text), "Balls: %zu", ball_regions.size());
        img_maix->draw_string(10, 40, ball_count_text, image::COLOR_WHITE);

        float frame_time = (now - frame_start) / 1.0f;
        char time_text[32];
        snprintf(time_text, sizeof(time_text), "Frame: %.1fms", frame_time);
        img_maix->draw_string(10, 70, time_text, image::COLOR_WHITE);

        char tx_str[4];
        snprintf(tx_str, sizeof(tx_str), "%d", (int)tx);
        char ty_str[4];
        snprintf(ty_str, sizeof(ty_str), "%d", (int)ty);


        img_maix->draw_string(440, 10, tx_str, image::COLOR_WHITE, 1, 1);
        img_maix->draw_rect(add_x, add_y, add_w, add_h, image::COLOR_WHITE, 1.2);
        img_maix->draw_string(add_x, add_y, "+", image::COLOR_WHITE, 1 ,1);
        
        img_maix->draw_string(440, 50, ty_str, image::COLOR_WHITE, 1, 1);
        img_maix->draw_rect(reduce_x, reduce_y, reduce_w, reduce_h, image::COLOR_WHITE, 1.2);
        img_maix->draw_string(reduce_x, reduce_y, "-", image::COLOR_WHITE, 1 ,1);    

        char mode_str[7];
        snprintf(mode_str, sizeof(mode_str), "Mode%d", (int)mode);
        img_maix->draw_rect(mode_x, mode_y, mode_w, mode_h, image::COLOR_WHITE, 1.2);
        img_maix->draw_string(mode_x, mode_y, mode_str, image::COLOR_WHITE, 1 ,1);
        
        char set_bit_str[7];
        if(set_bit == &cm){
            snprintf(set_bit_str, sizeof(set_bit_str), "Set:cm");
        }
        if(set_bit == &mm){
            snprintf(set_bit_str, sizeof(set_bit_str), "Set:mm");
        }
        img_maix->draw_rect(set_bit_x, set_bit_y, set_bit_w, set_bit_h, image::COLOR_WHITE, 1.2);
        img_maix->draw_string(set_bit_x, set_bit_y, set_bit_str, image::COLOR_WHITE, 1 ,1);  

        char mid_offset_str[4];
        snprintf(mid_offset_str, sizeof(mid_offset_str), "%f", mid_offset);
        img_maix->draw_string(add_x + add_w + 20, add_y, mid_offset_str, image::COLOR_WHITE, 1, 1 );
        
        // 12. 显示结果
        char value_buff[10];
        snprintf(value_buff, sizeof(value_buff), "%.02f", value);
        img_maix->draw_string(100,30,value_buff,image::COLOR_WHITE, 1, 1);
        disp.show(*img_maix);
        delete img_maix;

    }

    log::info("Program exit");
    return 0;
}

int main(int argc, char* argv[])
{
    sys::register_default_signal_handle();
    CATCH_EXCEPTION_RUN_RETURN(_main, -1, argc, argv);
}

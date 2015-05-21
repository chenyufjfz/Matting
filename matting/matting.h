#ifndef _MATTING_H_
#define _MATTING_H_
#include "Gpuopt.h"
#include <opencv2/opencv.hpp>
#include <mutex>

struct Color32 {
	unsigned char r, g, b, a;
};

struct TuningPara {
	int show_content;
	int motion_TH;
	int alpha_TH;
	int static_MAX;
	int init_static_num;
	int luma_offset;
	int u_gain;
	int v_gain;
	int static_NUM;
	int long_static_NUM;
	int alpha_filter_r;
	int block_th;
	float rasie_hands_radius;
	float static_SPEED;
	float long_SPEED;
	float GAMMA;	
};

class Matting 
{
public:
	virtual void reset(int out_width_, int out_height_) = 0;
	virtual bool tune_parameter(TuningPara & para) = 0;
	virtual void process(cv::Mat & frame_rgb) = 0;
	virtual int get_out(Color32 * tex) = 0;
	virtual int get_lost() = 0;
	virtual int get_no_update() = 0;	
};

class MattingCPUOrg : public Matting 
{
protected:
	cv::Mat frame_rgb_pre, motion_diff_rgb, bg_diff_yuv, motion_diff, bg_diff_sum, static_num, is_bg, fg_sure;
	cv::Mat y, alpha_weight, bg_yuv, is_body;
	cv::Mat mask, out_mask, out_rgb, out_rgb_buf;
	volatile int out_update, const_updated;
	int out_width, out_height, lost, no_update;
	std::mutex out_lock, const_update_lock;
	long long frame_num;
	TuningPara Const, const_update;
public:
	MattingCPUOrg();
	void reset(int out_width_, int out_height_);
	bool tune_parameter(TuningPara & para);
	void process(cv::Mat & frame_rgb);
	int get_out(Color32 * tex);
	int get_lost();
	int get_no_update();	
};

struct TuningParaFloat {
	int show_content;
	float motion_TH_f;
	float alpha_TH_f;
	int static_MAX;
	int init_static_num;
	float luma_offset_f;
	float u_gain_f;
	float v_gain_f;
	int static_NUM;
	int long_static_NUM;
	int alpha_filter_r;
	int block_th;
	float rasie_hands_radius;
	float static_SPEED_f;
	float long_SPEED_f;
	float GAMMA;
};

class MattingCPU : public Matting
{
protected:
	cv::Mat frame_rgb_pre, frame_rgb, bg_yuv;
	cv::Mat	static_num, is_bg, fg_sure, alpha_erode, is_body;
	cv::Mat mask, out_mask, out_rgb, out_rgb_buf;
	volatile int out_update, const_updated;
	int out_width, out_height, lost, no_update;
	std::mutex out_lock, const_update_lock;
	long long frame_num;
	TuningParaFloat Const;
	TuningPara const_update;
public:
	MattingCPU();
	void reset(int out_width_, int out_height_);
	bool tune_parameter(TuningPara & para);
	void process(cv::Mat & frame);
	int get_out(Color32 * tex);
	int get_lost();
	int get_no_update();
};

#if HAVE_GPU==1
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#define CHANNEL 3
#define PROFILE 1
class MattingGPU : public Matting
{
protected:
	cv::gpu::GpuMat frame_rgb_raw_gpu, frame_rgb_gpu, frame_rgb_pre_gpu, bg_yuv_gpu, static_num_gpu, is_bg_gpu, is_body_gpu, alpha_erode_gpu, box_buf, dilate_buf, erode_buf;
	cv::gpu::CudaMem frame_rgb_cpu, alpha_erode_cpu;
	cv::gpu::CudaMem  bg_yuv_cpu;
	cv::Mat out_mask, out_rgb, out_rgb_buf;
	cv::gpu::Stream stream;
	volatile int out_update, const_updated;
	int out_width, out_height, lost, no_update;
	std::mutex out_lock, const_update_lock;
	long long frame_num;
	TuningPara const_update;
	TuningParaFloat Const;
	cv::Ptr<cv::gpu::FilterEngine_GPU> box_filter;
	cv::Ptr<cv::gpu::FilterEngine_GPU> dilate_filter;
	cv::Ptr<cv::gpu::FilterEngine_GPU> erode_filter;
public:
	MattingGPU();
	void reset(int out_width_, int out_height_);
	bool tune_parameter(TuningPara & para);
	void process(cv::Mat & frame);
	int get_out(Color32 * tex);
	int get_lost();
	int get_no_update();
#if PROFILE==1
	double cpu_wait, gpu_launch, cpu_process;
#endif
};
#endif


class MattingFifo
{
protected:
	cv::Mat mat_buf;
	std::mutex buf_lock;
	int buf_update;
public:
	MattingFifo();
	void put(cv::Mat & frame);
	bool get(cv::Mat & frame);
};

void init_produce_thread(int mode, int produce_rate_);
void wait_produce_thread_end();
void write_texture(Color32 *tex, int wide, int height);
bool raise_hand_left(int &x, int &y);
bool raise_hand_right(int &x, int &y);
#endif
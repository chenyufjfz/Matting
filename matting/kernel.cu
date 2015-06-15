#include "opencv2/gpu/device/common.hpp"
#include <opencv2/core/core.hpp>
using namespace cv::gpu;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Matting/matting.h"
#if HAVE_GPU ==1
#define BLOCK_WIDE 64
#define BLOCK_HIGH 8

#define alpha_top 40
#define alpha_bottom 40
#define alpha_left 120
#define alpha_right 120

texture <float, cudaTextureType2D, cudaReadModeElementType> source_tex(false, cudaFilterModePoint, cudaAddressModeClamp);

namespace cv {
	namespace gpu {
		namespace device {
			/*__device__ const float motion_TH_f = motion_TH / 255.0;
			__device__ const float static_SPEED_f = static_SPEED / 255.0;
			__device__ const float long_SPEED_f = long_SPEED / 255.0;
			__device__ const float luma_offset_f = luma_offset / 255.0f;
			__device__ const float u_gain_f = u_gain;
			__device__ const float v_gain_f = v_gain;*/
			__constant__ TuningParaFloat Const;
			__constant__ HostPara host_para;
			__global__  void trace_bg_kernel(PtrStepSz<float> motion_diff_rgb_filted0, PtrStepSz<float> motion_diff_rgb_filted1, PtrStepSz<float> motion_diff_rgb_filted2, 
					PtrStepSz<float3> frame_yuv, PtrStepSz<float3> bg_yuv, PtrStepSz<float3> bg_diff_yuv, PtrStepSzb static_num, PtrStepSzb is_bg, PtrStepSzb is_body)
			{
				extern __shared__ float smem[];
				typename float * gray = smem;
				unsigned int gray_idx = threadIdx.y * blockDim.x + threadIdx.x;
				unsigned int y = blockIdx.y * (blockDim.y-2) + threadIdx.y;
				unsigned int x = blockIdx.x * (blockDim.x-2) + threadIdx.x;
				
				if (y < static_num.rows && x < static_num.cols) {
					gray[gray_idx] = frame_yuv.ptr(y)[x].x;
					__syncthreads();
					if (threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.x != blockDim.x - 1 && threadIdx.y != blockDim.y - 1 
						&& y + 1<static_num.rows && x + 1<static_num.cols) {
						float edge_offset = MAX(fabs(gray[gray_idx - blockDim.x - 1] - gray[gray_idx + blockDim.x + 1]),
							fabs(gray[gray_idx - blockDim.x + 1] - gray[gray_idx + blockDim.x - 1])) / 2;
						float motion_diff = fabs(motion_diff_rgb_filted0.ptr(y)[x]) + fabs(motion_diff_rgb_filted1.ptr(y)[x]) + fabs(motion_diff_rgb_filted2.ptr(y)[x]);
						unsigned char static_num_reg = static_num.ptr(y)[x];
						if (motion_diff < edge_offset + Const.motion_TH_f)
							static_num_reg = MIN(static_num_reg + 1, Const.static_MAX);
						else
							static_num_reg = 0;
						static_num.ptr(y)[x] = static_num_reg;
						float3 bg_yuv_reg = bg_yuv.ptr(y)[x];
						if (fabs(bg_yuv_reg.x) <= 0.001f && fabs(bg_yuv_reg.y - 1.0f) <= 0.001f && fabs(bg_yuv_reg.z) <=0.001f) {
							if (static_num_reg>= Const.init_static_num) 
								bg_yuv.ptr(y)[x] = frame_yuv.ptr(y)[x];								
						}
						else {
							float update_speed;
							if (is_bg.ptr(y)[x] && static_num_reg >= Const.static_NUM)
								update_speed = Const.static_SPEED_f;
							else if (is_body.ptr(y)[x] == 0 && static_num_reg >= Const.long_static_NUM)
								update_speed = Const.long_SPEED_f;
							else
								update_speed = 0;
							float3 bg_diff_yuv_reg = bg_diff_yuv.ptr(y)[x];
							bg_yuv_reg.x = (bg_diff_yuv_reg.x > 0) ? (bg_yuv_reg.x + update_speed) : (bg_yuv_reg.x - update_speed);
							bg_yuv_reg.y = (bg_diff_yuv_reg.y > 0) ? (bg_yuv_reg.y + update_speed) : (bg_yuv_reg.y - update_speed);
							bg_yuv_reg.z = (bg_diff_yuv_reg.z > 0) ? (bg_yuv_reg.z + update_speed) : (bg_yuv_reg.z - update_speed);
							bg_yuv.ptr(y)[x] = bg_yuv_reg;
						}						
					} 
				}
			}
			
			__global__ void update_mask_bg_kernel(PtrStepSz<float> bg_diff_filted0, PtrStepSz<float> bg_diff_filted1, PtrStepSz<float> bg_diff_filted2, 
					PtrStepSzb fg_sure, PtrStepSzb fg_maybe, PtrStepSzb is_body)
			{			
				unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + alpha_top;
				unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + alpha_left;
				
				float bg_diff_abs_y = fabs(bg_diff_filted0.ptr(y)[x]);
				float bg_diff_abs_u = fabs(bg_diff_filted1.ptr(y)[x]);
				float bg_diff_abs_v = fabs(bg_diff_filted2.ptr(y)[x]);
				
				bg_diff_abs_y = MAX(0.0f, bg_diff_abs_y - Const.luma_offset_f);
				bg_diff_abs_u = bg_diff_abs_u * Const.u_gain_f;
				bg_diff_abs_v = bg_diff_abs_v * Const.v_gain_f;
				float bg_diff_all = (bg_diff_abs_y + bg_diff_abs_u + bg_diff_abs_v)*(fg_sure.ptr(y)[x] + 1);
				float motion_th = Const.alpha_TH_f;
				if ((y >= host_para.body_top - 1) && (y <= host_para.body_bottom - 1) && (x >= host_para.body_left - 1) && (x <= host_para.body_right - 1)) {
					is_body.ptr(y)[x] = 1;
					motion_th = Const.alpha_TH_f / 2;
				} else
					is_body.ptr(y)[x] = 0;

				if (bg_diff_all > motion_th * 2) {					
					fg_sure.ptr(y)[x] = 1;
					fg_maybe.ptr(y)[x] = 1;									
				}
				else {
					fg_sure.ptr(y)[x] = 0;					
					if (bg_diff_all > motion_th)
						fg_maybe.ptr(y)[x] = 1;
					else 
						fg_maybe.ptr(y)[x] = 0;				
				}
			}

			__global__  void box_filter_kernel(PtrStepSz<float> filter_out, int ksize, float scale, int block_high, int block_wide, int PPT)
			{
				extern __shared__ float smem[];
				int y_ldram = (int)blockIdx.y * block_high - ksize;
				int y_ldram_end = MIN((blockIdx.y + 1) * block_high + ksize, filter_out.rows + ksize);
				int x_ldram = (int)blockIdx.x * block_wide + threadIdx.x - ksize;
				int x_ldram_end = MIN((blockIdx.x + 1) * block_wide + ksize, filter_out.cols + ksize);
				int load_line = blockDim.x / (block_wide / PPT);
				float * sum_line = smem;
				float * raw_line = &smem[(2 * ksize + 1) * block_wide];
				int raw_line_len = block_wide + ksize * 2 + 1;
				int x_raw_line = threadIdx.x / load_line;
				int y_raw_line = threadIdx.x % load_line;
				float * raw0 = &raw_line[y_raw_line * raw_line_len + x_raw_line*PPT];
				int y_wrram = (int)blockIdx.y * block_high - ksize * 2;
				int y_wrsum = 0;
				float out = 0;

				if (threadIdx.x < block_wide)
					for (int i = 0; i < 2 * ksize + 1; i++)
						sum_line[i*block_wide + threadIdx.x] = 0;
				__syncthreads();

				while (y_ldram < y_ldram_end) {
					for (unsigned ll = 0; ll < load_line && y_ldram < y_ldram_end; ll++, y_ldram++)
						if (x_ldram < x_ldram_end)
							raw_line[ll*raw_line_len + threadIdx.x] = tex2D(source_tex, x_ldram, y_ldram);

					__syncthreads();
					float s0 = raw0[0];
					for (int i = 1; i < ksize * 2 + 1; i++)
						s0 += raw0[i];
					if (PPT == 8) {
						float s1, s2, s3, s4, s5, s6, s7;
						s1 = s0 + raw0[ksize * 2 + 1] - raw0[0];
						s2 = s1 + raw0[ksize * 2 + 2] - raw0[1];
						s3 = s2 + raw0[ksize * 2 + 3] - raw0[2];
						s4 = s3 + raw0[ksize * 2 + 4] - raw0[3];
						s5 = s4 + raw0[ksize * 2 + 5] - raw0[4];
						s6 = s5 + raw0[ksize * 2 + 6] - raw0[5];
						s7 = s6 + raw0[ksize * 2 + 7] - raw0[6];
						__syncthreads();
						raw0[0] = s0;
						raw0[1] = s1;
						raw0[2] = s2;
						raw0[3] = s3;
						raw0[4] = s4;
						raw0[5] = s5;
						raw0[6] = s6;
						raw0[7] = s7;
					}
					else if (PPT == 4) {
						float s1, s2, s3;
						s1 = s0 + raw0[ksize * 2 + 1] - raw0[0];
						s2 = s1 + raw0[ksize * 2 + 2] - raw0[1];
						s3 = s2 + raw0[ksize * 2 + 3] - raw0[2];
						__syncthreads();
						raw0[0] = s0;
						raw0[1] = s1;
						raw0[2] = s2;
						raw0[3] = s3;
					}
					__syncthreads();
					if (x_ldram < x_ldram_end - ksize * 2) {
						int x_wrram = x_ldram + ksize;
						for (int i = 0; i < load_line && y_wrram < y_ldram_end - ksize; i++, y_wrram++) {
							out += raw_line[i*raw_line_len + threadIdx.x] - sum_line[y_wrsum*block_wide + threadIdx.x];
							sum_line[y_wrsum*block_wide + threadIdx.x] = raw_line[i*raw_line_len + threadIdx.x];
							y_wrsum = (y_wrsum >= ksize * 2) ? 0 : y_wrsum + 1;
							if (y_wrram >= (int)blockIdx.y * block_high)
								filter_out.ptr(y_wrram)[x_wrram] = out *scale;
						}
					}
					__syncthreads();

				}
			}

			void box_filter_(PtrStepSzb & filter_out, int ksize, float scale, cudaStream_t stream)
			{
				int block_high = 0, block_wide = 256, PPT = 0;

				cudaDeviceProp deviceProp;
				cudaGetDeviceProperties(&deviceProp, 0);

				if (PPT == 0) {
					int threadx = block_wide + block_wide / 8 * (1 + (ksize * 2 - 1) / (block_wide / 8));
					int smem_size = (2 * ksize + 1)*block_wide*sizeof(float) + threadx / (block_wide / 8) * (block_wide + ksize * 2 + 1) *sizeof(float);
					if (smem_size + 10 >= deviceProp.sharedMemPerBlock)
						PPT = 4;
					else
						PPT = 8;
				}

				if (block_high == 0) {
					int max = 0, col = divUp(filter_out.cols, block_wide);
					for (int i = 1; i <= 5; i++) {
						int e = (col * i) % deviceProp.multiProcessorCount;
						if (e == 0)
							e = deviceProp.multiProcessorCount;
						if (max < e) {
							max = e;
							block_high = (filter_out.rows - 1) / i + 1;
						}
					}
				}
				//printf("row =%d, col=%d, BH=%d, BW=%d, PPT=%d, k=%d\n", filter_out.rows, filter_out.cols, block_high, block_wide, PPT, ksize);
				CV_Assert(PPT == 4 || PPT == 8);

				const dim3 block(block_wide + MAX(block_wide / PPT * (1 + (ksize * 2 - 1) / (block_wide / PPT)), 32));
				const dim3 grid(divUp(filter_out.cols, block_wide), divUp(filter_out.rows, block_high));
				const size_t smemSize = (2 * ksize + 1)*block_wide*sizeof(float) + block.x / (block_wide / PPT) * (block_wide + ksize * 2 + 1) *sizeof(float);
				CV_Assert(block.x >= 2 * ksize + block_wide && smemSize < deviceProp.sharedMemPerBlock);

				//printf("thread =%d, Dimx=%d, Dimy=%d, smemSize=%d\n", block.x, grid.x, grid.y, smemSize);
				box_filter_kernel << <grid, block, smemSize, stream >> > (static_cast<PtrStepSz<float>>(filter_out), ksize, scale, block_high, block_wide, PPT);
			}

			void trace_bg_(PtrStepSzb motion_diff_rgb_filted0, PtrStepSzb motion_diff_rgb_filted1, PtrStepSzb motion_diff_rgb_filted2, 
					PtrStepSzb frame_yuv, PtrStepSzb bg_yuv, PtrStepSzb bg_diff_yuv, PtrStepSzb static_num, PtrStepSzb is_bg, PtrStepSzb is_body, cudaStream_t stream)
			{
				const dim3 block(BLOCK_WIDE, BLOCK_HIGH);
				const dim3 grid(divUp(frame_yuv.cols - 2, BLOCK_WIDE - 2), divUp(frame_yuv.rows - 2, BLOCK_HIGH - 2));
				const size_t smemSize = BLOCK_WIDE * BLOCK_HIGH * sizeof(float);
				
				trace_bg_kernel<< <grid, block, smemSize, stream >> > (static_cast<PtrStepSz<float>>(motion_diff_rgb_filted0), static_cast<PtrStepSz<float>>(motion_diff_rgb_filted1), static_cast<PtrStepSz<float>>(motion_diff_rgb_filted2), 
					static_cast<PtrStepSz<float3>>(frame_yuv), static_cast<PtrStepSz<float3>>(bg_yuv), static_cast<PtrStepSz<float3>>(bg_diff_yuv), static_num, is_bg, is_body);
			}
			
			void update_mask_bg_(PtrStepSzb bg_diff_filted0, PtrStepSzb bg_diff_filted1, PtrStepSzb bg_diff_filted2, 
					PtrStepSzb fg_sure, PtrStepSzb fg_maybe, PtrStepSzb is_body, cudaStream_t stream)
			{
				const dim3 block(BLOCK_WIDE, BLOCK_HIGH);
				const dim3 grid(divUp(fg_sure.cols - alpha_left - alpha_right, BLOCK_WIDE), divUp(fg_sure.rows - alpha_top - alpha_bottom, BLOCK_HIGH));
				const size_t smemSize = 0;
				
				update_mask_bg_kernel << <grid, block, smemSize, stream >> > (static_cast<PtrStepSz<float>>(bg_diff_filted0), static_cast<PtrStepSz<float>>(bg_diff_filted1), static_cast<PtrStepSz<float>>(bg_diff_filted2),
					fg_sure, fg_maybe, is_body);
			}
		}
	}
}

void trace_bg(PtrStepSzb motion_diff_rgb_filted0, PtrStepSzb motion_diff_rgb_filted1, PtrStepSzb motion_diff_rgb_filted2, 
					PtrStepSzb frame_yuv, PtrStepSzb bg_yuv, PtrStepSzb bg_diff_yuv, PtrStepSzb static_num, PtrStepSzb is_bg, PtrStepSzb is_body, cudaStream_t stream)
{
	CV_Assert(motion_diff_rgb_filted0.cols==is_bg.cols && frame_yuv.cols==is_bg.cols && bg_yuv.cols==is_bg.cols && bg_diff_yuv.cols==is_bg.cols
		&& static_num.cols==is_bg.cols && is_body.cols==is_bg.cols);
	CV_Assert(motion_diff_rgb_filted0.rows==is_bg.rows && frame_yuv.rows==is_bg.rows && bg_yuv.rows==is_bg.rows && bg_diff_yuv.rows==is_bg.rows
		&& static_num.rows==is_bg.rows && is_body.rows==is_bg.rows);
		
	device::trace_bg_(motion_diff_rgb_filted0, motion_diff_rgb_filted1, motion_diff_rgb_filted2, frame_yuv, bg_yuv, 
		bg_diff_yuv, static_num, is_bg, is_body, stream);
}

void update_mask_bg(PtrStepSzb bg_diff_filted0, PtrStepSzb bg_diff_filted1, PtrStepSzb bg_diff_filted2, 
					PtrStepSzb fg_sure, PtrStepSzb fg_maybe, PtrStepSzb is_body, cudaStream_t stream)
{
	CV_Assert(bg_diff_filted0.cols==is_body.cols && bg_diff_filted1.cols==is_body.cols && bg_diff_filted2.cols==is_body.cols
		&& fg_sure.cols==is_body.cols && fg_maybe.cols==is_body.cols);
	CV_Assert(bg_diff_filted0.rows==is_body.rows && bg_diff_filted1.rows==is_body.rows && bg_diff_filted2.rows==is_body.rows
		&& fg_sure.rows==is_body.rows && fg_maybe.rows==is_body.rows);

	device::update_mask_bg_(bg_diff_filted0, bg_diff_filted1, bg_diff_filted2, fg_sure, fg_maybe, is_body, stream);
}

void box_filter_gpu(PtrStepSzb raw_in, PtrStepSzb filter_out, int ksize, float scale = -1, cudaStream_t stream = NULL)
{
	CV_Assert(raw_in.cols == filter_out.cols && raw_in.rows == filter_out.rows);
	if (scale == -1)
		scale = 1.0f / ((ksize * 2 + 1) *(ksize * 2 + 1));

	cv::gpu::device::bindTexture(&source_tex, static_cast<PtrStepSzf> (raw_in));
	cv::gpu::device::box_filter_(filter_out, ksize, scale, stream);
}

void tune_gpu_parameter(TuningParaFloat *c)
{
	checkCudaErrors(cudaMemcpyToSymbol(device::Const, c, sizeof(TuningParaFloat)));
}

void update_host_para(HostPara *p)
{
	checkCudaErrors(cudaMemcpyToSymbol(device::host_para, p, sizeof(HostPara)));
}

#endif

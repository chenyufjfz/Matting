// MattingDll.cpp : 定義 DLL 應用程式的匯出函式。
//
#include "targetver.h"
#include "../matting/Gpuopt.h"
#include "../matting/matting.h"
#include "../matting/MattingPostProcess.h"
static Color32 * tex[2];
static int width;
static int height;
static long long index;
extern Matting * matting;
//bool start_camera_capture();
//void stop_camera_capture();
extern "C" {
	__declspec(dllexport) int MattingdllVersion()
	{
		return 1;
	}

	__declspec(dllexport) int MattingReset(int gpu, int width_, int height_)
	{
#if HAVE_GPU==1
		if (gpu)
			matting = new MattingGPU();
		else
#endif
			matting = new MattingCPU();
		matting->reset(width_, height_);
		return 1;
	}

	__declspec(dllexport) int MattingGetOutput(Color32 * tex)
	{
		return matting->get_out(tex);
	}

	__declspec(dllexport) int MattingGetLost()
	{
		return matting->get_lost();
	}

	__declspec(dllexport) int MattingGetNoUpdate()
	{
		return matting->get_no_update();
	}
	__declspec(dllexport) void MattingStartProduce(int mode, int produce_rate)
	{
		init_produce_thread(mode, produce_rate);
	}

	__declspec(dllexport) void MattingWaitProduceEnd(int mode)
	{
		wait_produce_thread_end();
	}

	__declspec(dllexport) void MattingWriteDisk(Color32 * tex, int wide, int height)
	{
		write_texture(tex, wide, height);
	}
	__declspec(dllexport) bool MattingGetLeftHand(int *x, int *y)
	{
		return raise_hand_left(*x, *y);
	}
	__declspec(dllexport) bool MattingGetRightHand(int *x, int *y)
	{
		return raise_hand_right(*x, *y);
	}
}

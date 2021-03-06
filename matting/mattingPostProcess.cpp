#include "matting.h"
#include "MattingPostProcess.h"
using namespace cv;


int body_top = 156;
int body_bottom = 529;
int body_left = 664;
int body_right = 708;


int raise_hand_l = 0;
int raise_hand_r = 0;

const float rasie_hands_th = 0.05f;
const unsigned raise_frames = 0;


vector<Point2f> body_metries;

void GammaCorrection2(Mat& src, float fGamma)//提高函数运行效率
{
	CV_Assert(src.data);
	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));
	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	const int channels = src.channels();
	switch (channels)
	{
	case 1:
	{
		for (int y = 0; y < src.rows; y++)
		{
			unsigned char * p_src = src.ptr<unsigned char>(y);
			for (int x = 0; x < src.cols; x++)
			{
				p_src[x] = lut[p_src[x]];
			}
		}
		/*MatIterator_<uchar> it, end;
		for (it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++)
		*it = lut[(*it)];*/

		break;
	}
	case 3:
	{
		for (int y = 0; y < src.rows; y++)
		{
			unsigned char * p_src = src.ptr<unsigned char>(y);
			for (int x = 0; x < src.cols; x++)
			{
				p_src[3 * x] = lut[p_src[3 * x]];
				p_src[3 * x + 1] = lut[p_src[3 * x + 1]];
				p_src[3 * x + 2] = lut[p_src[3 * x + 2]];
			}
		}
		/*MatIterator_<Vec3b> it, end;
		for (it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; it++)
		{
		(*it)[0] = lut[((*it)[0])];
		(*it)[1] = lut[((*it)[1])];
		(*it)[2] = lut[((*it)[2])];
		}*/

		break;

	}
	}
}

void GammaCorrection(Mat& src, float fGamma)
{
	CV_Assert(src.data);
	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));
	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	const int channels = src.channels();
	switch (channels)
	{
	case 1:
	{
		MatIterator_<uchar> it, end;
		for (it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++)
			*it = lut[(*it)];

		break;
	}
	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; it++)
		{
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}


void FindContour(Mat & img, int block_th)
{
	//img是一个由0，255组成的二值图
	IplImage* src = &(img.operator IplImage());
	IplImage* dst = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* msk = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* msk2 = cvCloneImage(src);
	IplImage* msk3 = cvCloneImage(src);
	cvZero(msk);
	cvZero(dst);		// 清空数组

	// 提取人体外围轮廓
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	int contour_num = cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	CvSeq *_contour = contour;
	double maxarea = 0;
	double minarea = 100;
	for (; contour != 0; contour = contour->h_next)
	{

		double tmparea = fabs(cvContourArea(contour));
		if (tmparea < minarea)
		{
			cvSeqRemove(contour, 0); // 删除面积小于设定值的轮廓
			continue;
		}
		if (tmparea > maxarea)
		{
			maxarea = tmparea;
		}
	}
	contour = _contour;
	int count = 0;
	for (; contour != 0; contour = contour->h_next)
	{
		count++;
		double tmparea = fabs(cvContourArea(contour));
		if ((tmparea == maxarea) && (tmparea > minarea))
		{
			CvScalar color = CV_RGB(255, 255, 255);
			cvDrawContours(msk, contour, color, color, -1, -1, 8);
		}
	}

	Mat msk2_;
	msk2_ = Mat(msk2);
	Mat msk3_;
	msk3_ = Mat(msk3);
	for (int y = 0; y < msk2->height; y++)
	{
		unsigned char* p_msk2 = msk2_.ptr<unsigned char>(y);
		unsigned char* p_msk3 = msk3_.ptr<unsigned char>(y);
		const unsigned char* p_msk = ((Mat)msk).ptr<unsigned char>(y);
		for (int x = 0; x < msk2->width; x++)
		{
			if (p_msk[x] == 0)
			{
				p_msk2[x] = 0;
				p_msk3[x] = 0;
			}
			else
			{
				p_msk3[x] = 255;
				if (p_msk2[x] == 0)
					p_msk2[x] = 255;
				else
					p_msk2[x] = 0;
			}
		}

	}

	//提取人体包含内部轮廓
	CvMemStorage* storage2 = cvCreateMemStorage(0);
	CvSeq* contour2 = 0;
	int contour_num2 = cvFindContours(msk2, storage2, &contour2, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	double minarea2 = block_th;
	int count2 = 0;
	for (; contour2 != 0; contour2 = contour2->h_next)
	{

		double tmparea = fabs(cvContourArea(contour2));
		if (tmparea < minarea2)
		{
			cvSeqRemove(contour2, 0); // 删除面积小于设定值的轮廓
			continue;
		}
		// 创建一个色彩值
		CvScalar color = CV_RGB(255, 255, 255);
		cvDrawContours(dst, contour2, color, color, 0, -1, 8);	//绘制外部和内部的轮廓
		count2++;
	}

	for (int y = 0; y < msk3->height; y++)
	{
		unsigned char* p_msk3 = msk3_.ptr<unsigned char>(y);
		const unsigned char* p_msk = ((Mat)dst).ptr<unsigned char>(y);
		for (int x = 0; x < msk3->width; x++)
		{
			if (p_msk[x] == 255)
				p_msk3[x] = 0;
		}

	}

	//img = msk3;//返回结果

	Mat img_tmp;
	img_tmp = msk3;//返回结果
	img = img_tmp.clone();

	cvReleaseMemStorage(&storage);
	cvReleaseMemStorage(&storage2);
	//cvReleaseImage(&src);
	cvReleaseImage(&dst);
	cvReleaseImage(&msk);
	cvReleaseImage(&msk2);
	cvReleaseImage(&msk3);
}

bool get_body_centroid(int &x, int &y)
{
	if (body_metries.size() < 6)
		return false;
	Point2f  centroid;
	centroid = body_metries[0];
	x = (int)centroid.x;
	y = (int)centroid.y;
	return true;
}

bool raise_hand_left(int &x, int &y)
{
	if (body_metries.size() < 6)
		return false;
	Point2f  left;
	left = body_metries[1];
	x = (int)left.x;
	y = (int)left.y;
	return raise_hand_l > raise_frames;
}
bool raise_hand_right(int &x, int &y)
{
	if (body_metries.size() < 6)
		return false;
	Point2f  right;
	right = body_metries[2];
	x = (int)right.x;
	y = (int)right.y;

	return raise_hand_r > raise_frames;
}

float distance_l2(Point2f p1, Point2f p2)
{
	return sqrtf((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}


void FindGeometry(const Mat & img, float rasie_hands_radius, float ratio_x, float ratio_y)
{
	//质心，宽度，极值点
	Point2f  centroid, left, right, top, bottom;
	Point2f body_width;
	left.x = img.cols;
	top.y = img.rows;

	int area = 0;//像素数目
	for (int y = 0; y < img.rows; y++)
	{
		const unsigned char * p_img = img.ptr<unsigned char>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (p_img[x] == 255)
			{
				centroid.x += x;
				centroid.y += y;
				if (x < left.x)
				{
					left.x = x;
					left.y = y;
				}
				if (x > right.x)
				{
					right.x = x;
					right.y = y;
				}
				if (y < top.y)
				{
					top.x = x;
					top.y = y;
				}
				if (y > bottom.y)
				{
					bottom.x = x;
					bottom.y = y;
				}
				area++;
			}
		}
	}
	if (area>0)
	{
		centroid.x /= area;
		centroid.y /= area;
		body_width.x = area / (bottom.y - top.y);
	}
	left.x = img.cols;
	right.x = 0;
	top.y = img.rows;
	const int top_width = body_width.x / 8;
	for (int y = 0; y < centroid.y; y++)
	{
		const unsigned char * p_img = img.ptr<unsigned char>(y);
		for (int x = 0; x < img.cols; x++)
		{
			if (p_img[x] == 255)
			{
				if (x < left.x)
				{
					left.x = x;
					left.y = y;
				}
				if (x > right.x)
				{
					right.x = x;
					right.y = y;
				}

				if (x>centroid.x - top_width&&x<centroid.x + top_width)//头顶点
				{
					if (y<top.y)
					{
						top.x = x;
						top.y = y;
						/*static queue<Point2f> tops;
						if (tops.size()>=500)
						{
						tops.pop();
						}
						tops.push(top);*/
					}
				}

			}
		}
	}
	Point2f body_tl, body_br;
	//设置body区域
	{
		body_tl.x = centroid.x - body_width.x;
		body_tl.y = top.y + (centroid.y - top.y) / 3;
		body_br.x = centroid.x + body_width.x;
		body_br.y = bottom.y;
		body_top = body_tl.y;
		body_left = body_tl.x;
		body_right = body_br.x;
		body_bottom = body_br.y;
	}
	//判断是否升左右手
	static Point2f pre_left;
	static Point2f pre_right;
	{
		Point2f heart;

		heart.x = (top.x + centroid.x) / 2;
		heart.y = (top.y + centroid.y) / 2;
		if ((distance_l2(heart, left) > body_width.x*rasie_hands_radius)
			//&& (distance_l2(pre_left, left) < body_width.x*rasie_hands_th))
			&& (left.x<centroid.x - body_width.x))
		{
			raise_hand_l++;
		}
		else
		{
			raise_hand_l = 0;
		}
		if ((distance_l2(heart, right) > body_width.x*rasie_hands_radius)
			//&& (distance_l2(pre_right, right) < body_width.x*rasie_hands_th))
			&& (right.x > centroid.x + body_width.x))
		{

			raise_hand_r++;
		}
		else
		{
			raise_hand_r = 0;
		}
	}
	pre_left = left;
	pre_right = right;


	vector<Point2f> ret_val;
	ret_val.push_back(centroid);
	ret_val.push_back(left);
	ret_val.push_back(right);
	ret_val.push_back(top);
	ret_val.push_back(bottom);
	ret_val.push_back(body_width);

	/*cout << centroid << endl;
	cout << left << right << top << bottom << endl;
	cout << body_width << endl;*/
	for (int i = 0; i < body_metries.size(); i++) {
		ret_val[i].x = ret_val[i].x  * ratio_x;
		ret_val[i].y = ret_val[i].y  * ratio_y;
	}
	body_metries = ret_val;
}

// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

/// ------LICENSE PLATE PROJECT-----------
/// --------------------------------------

bool isInside(int i, int j, int h, int w)
{
	if (i < h && i >= 0 && j < w && j >= 0)
		return true;
	return false;
}

Mat dilationHelper(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();

	int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar maxVal = src.at<uchar>(i, j);
			for (int k = 0; k < 8; ++k) {
				int ni = i + di[k], nj = j + dj[k];
				if (isInside(ni, nj, height, width)) {
					maxVal = max(maxVal, src.at<uchar>(ni, nj));
				}
			}
			dst.at<uchar>(i, j) = maxVal;
		}
	}

	return dst;
}


Mat erosionHelper(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();

	int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };


	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar minVal = src.at<uchar>(i, j);
			for (int k = 0; k < 8; ++k) {
				int ni = i + di[k], nj = j + dj[k];
				if (isInside(ni, nj, height, width)) {
					minVal = min(minVal, src.at<uchar>(ni, nj));
				}
			}
			dst.at<uchar>(i, j) = minVal;
		}
	}

	return dst;
}


Mat closingHelper(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
	dst = dilationHelper(src);
	dst = erosionHelper(dst);
	return dst;
}

Mat convertToGrayscale(const Mat& src) {
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	return gray;
}

Mat resizeForStandardization(const Mat& src, int targetWidth = 600) {
	int originalWidth = src.cols;
	int originalHeight = src.rows;

	if (originalWidth <= targetWidth) {
		return src.clone(); 
	}

	float aspectRatio = static_cast<float>(originalHeight) / originalWidth;
	int newHeight = static_cast<int>(targetWidth * aspectRatio);

	Mat resized;
	resize(src, resized, Size(targetWidth, newHeight), 0, 0, INTER_AREA);
	return resized;
}

Mat preprocessImage(const Mat& src)
{

	Mat resized = resizeForStandardization(src);
	Mat gray = convertToGrayscale(resized);
	return gray;
}

Mat blackHat(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat blackHatM = Mat(height, width, CV_8UC1);
	Mat closed = closingHelper(src);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int diff = static_cast<int>(closed.at<uchar>(i, j)) - static_cast<int>(src.at<uchar>(i, j));
			blackHatM.at<uchar>(i, j) = (diff < 0) ? 0 : static_cast<uchar>(diff);
		}
	}
	return blackHatM;
}

int computeOtsuThreshold(const Mat& gray)
{
	// Step 1: Compute histogram
	int hist[256] = { 0 };
	int totalPixels = gray.rows * gray.cols;

	for (int i = 0; i < gray.rows; i++) {
		for (int j = 0; j < gray.cols; j++) {
			uchar pixel = gray.at<uchar>(i, j);
			hist[pixel]++;
		}
	}

	// Step 2: Compute global mean (μT)
	double sum = 0;
	for (int t = 0; t < 256; t++) {
		sum += t * hist[t];
	}

	double sumB = 0;
	int wB = 0;
	int wF = 0;
	double maxVariance = 0;
	int bestThreshold = 0;

	for (int t = 0; t < 256; t++) {
		wB += hist[t];               // Weight of background
		if (wB == 0) continue;

		wF = totalPixels - wB;       // Weight of foreground
		if (wF == 0) break;

		sumB += t * hist[t];

		double mB = sumB / wB;       // Mean of background
		double mF = (sum - sumB) / wF; // Mean of foreground

		// Between-class variance
		double varBetween = static_cast<double>(wB) * wF * (mB - mF) * (mB - mF);

		// Maximize the between-class variance
		if (varBetween > maxVariance) {
			maxVariance = varBetween;
			bestThreshold = t;
		}
	}

	return bestThreshold;
}

Mat applyBinaryThreshold(const Mat& srcGray, int thresholdValue, uchar maxValue = 255)
{
	int height = srcGray.rows;
	int width = srcGray.cols;

	Mat binary = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar pixel = srcGray.at<uchar>(i, j);
			binary.at<uchar>(i, j) = (pixel >= thresholdValue) ? maxValue : 0;
		}
	}

	return binary;
}

Mat myScharrX(const Mat& srcGray)
{
	int height = srcGray.rows;
	int width = srcGray.cols;

	Mat dst = Mat::zeros(height, width, CV_8UC1);

	int kernelX[3][3] = {
		{ -3,  0,  3 },
		{-10,  0, 10 },
		{ -3,  0,  3 }
	};

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			int sum = 0;

			for (int k = -1; k < 2; k++) {
				for (int l = -1; l < 2; l++) {
					sum += kernelX[k + 1][l + 1] * srcGray.at<uchar>(i + k, j + l);
				}
			}
			dst.at<uchar>(i, j) = min(255, abs(sum));
		}
	}

	return dst;
}

Mat myGaussianBlur(Mat src)
{
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		int dimension = 3;
		int h[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

		for (int i = dimension / 2; i < height - dimension / 2; i++) {
			for (int j = dimension / 2; j < width - dimension / 2; j++) {
				int sum = 0;
				for (int l = 0; l < dimension; l++) {
					for (int k = 0; k < dimension; k++) {
						sum += src.at<uchar>(i - dimension / 2 + l, j - dimension / 2 + k) * h[l][k];
					}
				}
				dst.at<uchar>(i, j) = sum / 16;
			}
		}
		return dst;
}

Mat cleanBinaryImage(Mat src, int erodeCount = 1, int dilateCount = 2) {
	Mat cleaned = src.clone();

	// Apply erosion
	for (int i = 0; i < erodeCount; i++) {
		cleaned = erosionHelper(cleaned);
	}

	// Apply dilation
	for (int i = 0; i < dilateCount; i++) {
		cleaned = dilationHelper(cleaned);
	}

	//// Apply closing (dilation + erosion)
	//for (int i = 0; i < closingCount; i++) {
	//	cleaned = closingHelper(cleaned);
	//}

	return cleaned;
}

cv::Rect myBoundingRect(const std::vector<cv::Point>& contour) {
	if (contour.empty()) return cv::Rect();  

	int minX = contour[0].x;
	int minY = contour[0].y;
	int maxX = contour[0].x;
	int maxY = contour[0].y;

	for (const auto& point : contour) {
		if (point.x < minX) minX = point.x;
		if (point.y < minY) minY = point.y;
		if (point.x > maxX) maxX = point.x;
		if (point.y > maxY) maxY = point.y;
	}

	int width = maxX - minX + 1;
	int height = maxY - minY + 1;

	return cv::Rect(minX, minY, width, height);
}


void licensePlateDetection()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);

		Mat preprocessedImage = preprocessImage(src);
		Mat blackHatMatrix = blackHat(preprocessedImage);
		//Mat preBin = applyBinaryThreshold(preprocessedImage, 128); 
		//Mat closedImage = closingHelper(preprocessedImage);

		Mat closedImage = closingHelper(preprocessedImage);
		int otsuTreshold = computeOtsuThreshold(closedImage);
		std::cout << "Otsu treshold = " << otsuTreshold << '\n';
		Mat binaryImage = applyBinaryThreshold(closedImage, otsuTreshold);
		

		Mat scharrResult = myScharrX(blackHatMatrix);
		Mat gaussinBlur = myGaussianBlur(scharrResult);

		Mat closed2;
		Mat rectKernel = getStructuringElement(MORPH_RECT, Size(17, 3)); // Wider than tall
		morphologyEx(gaussinBlur, closed2, MORPH_CLOSE, rectKernel);

		int otsuTreshold2 = computeOtsuThreshold(closed2);
		std::cout << "Otsu treshold2 = " << otsuTreshold2 << '\n';
		Mat binaryImage2 = applyBinaryThreshold(closed2, otsuTreshold2);

		Mat clearBinaryImage = cleanBinaryImage(binaryImage2, 4, 6);

		Mat finalImage; 
		bitwise_and(clearBinaryImage, binaryImage, finalImage);

		std::string savePath = "output/final_" + std::string(fname).substr(std::string(fname).find_last_of("/\\") + 1);
		//imwrite(savePath, finalImage);

		std::vector<std::vector<cv::Point>> contours;
		findContours(finalImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		
		double maxArea = 0;
		cv::Rect bestPlate;

		for (const auto& contour : contours) {
			cv::Rect rect = myBoundingRect(contour);
			double aspectRatio = (double)rect.width / rect.height;
			double area = rect.area();

			if (aspectRatio > 2 && aspectRatio < 6 && rect.width > 50 && rect.height > 15 && rect.y > src.rows * 0.2) {
				if (area > maxArea) {
					maxArea = area;
					bestPlate = rect;
				}
			}
		}

		if (maxArea > 0) {

			cv::Mat roi = src(bestPlate); // crop from original image
			// Convert to grayscale and threshold
			cv::Mat gray, binary;
			gray = convertToGrayscale(roi);
			//cvtColor(roi, gray, COLOR_BGR2GRAY);
			
			int otsuTreshold3 = computeOtsuThreshold(gray);
			binary = applyBinaryThreshold(gray, otsuTreshold3);

			cv::rectangle(src, bestPlate, cv::Scalar(0, 255, 0), 2); // Green border

			imshow("output/plate_binary.png", binary); 
			imwrite("detected_plate.png", binary);
			imshow("output/plate_gray.png", gray);      
		}


		imshow("grayscale", preprocessedImage);
		imshow("blackhat", blackHatMatrix);
		imshow("closed Image", closedImage);
		imshow("otsu treshold", binaryImage);
		imshow("Scharr gradient", scharrResult);
		imshow("gaussian blur", gaussinBlur);
		imshow("binaryImage2", binaryImage2);
		imshow("closed2", closed2);
		imshow("clear binary image", clearBinaryImage);
		imshow("finalImage", finalImage);
		imshow("License Plate", src);
		waitKey(0);
	}
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - License Plate Detection\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				licensePlateDetection();
				break;
		}
	}
	while (op!=0);
	return 0;
}
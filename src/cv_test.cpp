#include <opencv2/highgui.hpp> //display image
#include <opencv2/calib3d.hpp> //find homography
#include <opencv2/imgproc.hpp> //warping
#include <opencv2/photo.hpp> //inpaint
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>


using namespace std;

int main(int argc, char* argv[]) {
	char image[256];
	char target_video[256];
	char point_file[256];

	if (argc < 4)
		exit(0);

	strcpy(image, argv[1]);
	strcpy(target_video, argv[2]);
	strcpy(point_file, argv[3]);

	ifstream in(point_file);

	vector<cv::Point2f> points;
	vector<cv::Point2f> curr_points;

	for (int i = 0; i < 4; i++) {
		double x, y;

		in >> x;
		in >> y;

		points.push_back(cv::Point2f(x, y));
	}
	
	cv::Mat img;
	img = cv::imread(image, CV_LOAD_IMAGE_COLOR);
	img.convertTo(img, CV_8UC3);

	vector<cv::Point2f> img_points;

	img_points.push_back(cv::Point2f(0, 0));
	img_points.push_back(cv::Point2f(img.cols - 1, 0));
	img_points.push_back(cv::Point2f(img.cols - 1, img.rows - 1));
	img_points.push_back(cv::Point2f(0, img.rows - 1));

	cv::VideoCapture cap(target_video);
	cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(255));

	int frame_count = 0;

	cv::Mat warped_image;
	cv::Mat warped_mask;

	cv::Mat prevFrame;
	cv::Mat currFrame;

	cv::Mat H;

	cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	cap.set(cv::CAP_PROP_FPS, 30.0);

	cv::VideoWriter outputVideo;
	outputVideo.open("output.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, size, true);

	while (true) {
		cap >> currFrame;

		if (currFrame.rows == 0 || currFrame.cols == 0)
			break;

		currFrame.convertTo(currFrame, CV_8UC3);

		if (frame_count == 0) {
			H = cv::findHomography(img_points, points);

			cv::warpPerspective(img, warped_image, H, currFrame.size());
			cv::warpPerspective(mask, warped_mask, H, currFrame.size(), cv::INTER_NEAREST);
		}
		else {
			cv::Mat prevGray, currGray;
			cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

			vector<unsigned char> status;
			vector<float> err;

			status.resize(points.size());
			err.resize(points.size());
			curr_points = points;

			cv::calcOpticalFlowPyrLK(prevFrame, currFrame, points, curr_points, status, err);

			H = cv::findHomography(img_points, curr_points);

			cv::warpPerspective(img, warped_image, H, currFrame.size());
			cv::warpPerspective(mask, warped_mask, H, currFrame.size(), cv::INTER_NEAREST);

			for (int i = 0; i < 4; i++)
				points[i] = curr_points[i];
		}

		cv::Mat output = currFrame.clone();

		for (int r = 0; r < output.rows; r++) {
			for (int c = 0; c < output.cols; c++) {
				if (warped_mask.at<unsigned char>(r, c) & 1) {
					output.at<cv::Vec3b>(r, c)[0] = warped_image.at<cv::Vec3b>(r, c)[0];
					output.at<cv::Vec3b>(r, c)[1] = warped_image.at<cv::Vec3b>(r, c)[1];
					output.at<cv::Vec3b>(r, c)[2] = warped_image.at<cv::Vec3b>(r, c)[2];
				}
			}
		}

		prevFrame = currFrame.clone();

		//cv::imshow("output", output);
		//cv::waitKey(0);

		outputVideo << output;

		frame_count++;
	}
	cout << "Done!"<< endl;

	std::cout << "<press key to continue>" << std::endl;
	getchar();

	return 0;
}
#include <iostream>
#include <opencv2/opencv.hpp>
#include "CameraCalibration.h"

using namespace std;

int main(int argc, char **argv) {

    const string keys =
            "{file_path1||directory for store left eye images}"
            "{file_path2||directory for store right eye images}"
            "{board_width|6|width of board}"
            "{board_height|9|height of board}"
            "{show_corners_flag flag|0|whether show corners found}";
    cv::CommandLineParser parser = cv::CommandLineParser(argc, argv, keys);
    if (argc < 5) {
        cout << "----- Calibrate a camera -----\n"
             << "Call:\ncamera_calibration.exe {file_path1} {file_path2}{board_width} {board_height} {show_corners_flag}\n"
             << "Example:\ncamera_calibration.exe -file_path1=./left -file_path2=./right 6 9 -flag=1 " << endl;
        return -1;
    }
    bool show_corners_flag = parser.get<bool>("show_corners_flag");
    int board_width = parser.get<int>("board_width");
    int board_height = parser.get<int>("board_height");
    string file_path1 = parser.get<string>("file_path1");
    string file_path2 = parser.get<string>("file_path2");

    CameraCalibration camera1(board_width, board_height, file_path1, show_corners_flag);
    CameraCalibration camera2(board_width, board_height, file_path2, show_corners_flag);
    //camera1.Calibrate();
    //camera1.Calibrate();
    cv::Mat R, T;
    cv::Mat E, F;
    double error = cv::stereoCalibrate(camera1.object_points, camera1.image_points, camera2.image_points,
                                       camera1.intrinsic_matrix, camera1.dist_coeffs, camera2.intrinsic_matrix,
                                       camera2.dist_coeffs,
                                       camera1.image_size, R, T, E, F,
                                       cv::CALIB_SAME_FOCAL_LENGTH | cv::CALIB_ZERO_TANGENT_DIST);
    cout << "image_size:" << camera1.image_size << endl;
    cout << "repeojection error: " << error << endl;
    cout << "rotation matrix:" << R << endl << "translation matrix" << T << endl;

    // Calculate epipoplar line projection error
    double avgErr = 0;
    vector<cv::Point3f> lines[2];
    int size = camera1.object_points.size();
    for (int i = 0; i < size; i++) {
        vector<cv::Point2f> &pt0 = camera1.image_points[i];
        vector<cv::Point2f> &pt1 = camera2.image_points[i];
        cv::undistortPoints(pt0, pt0, camera1.intrinsic_matrix, camera1.dist_coeffs, cv::Mat(),
                            camera1.intrinsic_matrix);
        cv::undistortPoints(pt1, pt1, camera2.intrinsic_matrix, camera2.dist_coeffs, cv::Mat(),
                            camera2.intrinsic_matrix);
        cv::computeCorrespondEpilines(pt0, 1, F, lines[0]);
        cv::computeCorrespondEpilines(pt1, 2, F, lines[1]);

        for (int j = 0; j < camera1.board_num; j++) {
            double err = fabs(pt0[j].x * lines[1][j].x + pt0[j].y * lines[1][j].y +
                              lines[1][j].z) +
                         fabs(pt1[j].x * lines[0][j].x + pt1[j].y * lines[0][j].y +
                              lines[0][j].z);
            avgErr += err;
        }
    }
    cout << "epipolar line projection average err = " << avgErr / (size * camera1.board_num) << endl;

    //do rectify and disparity map using SGBM
    vector<cv::Point2f> allpoints[2];
    for (int i = 0; i < camera1.object_points.size(); i++) {
        copy(camera1.image_points[i].begin(), camera1.image_points[i].end(),
             back_inserter(allpoints[0]));
        copy(camera2.image_points[i].begin(), camera2.image_points[i].end(),
             back_inserter(allpoints[1]));
    }

    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(allpoints[0], allpoints[1], F, camera1.image_size,
                                  H1, H2, 3);

    cv::Mat R1, R2, P1, P2, map11, map12, map21, map22;
    R1 = camera1.intrinsic_matrix.inv() * H1 * camera1.intrinsic_matrix;
    R2 = camera1.intrinsic_matrix.inv() * H2 * camera1.intrinsic_matrix;

    cv::initUndistortRectifyMap(camera1.intrinsic_matrix, camera1.dist_coeffs, R1, P1, camera1.image_size, CV_16SC2,
                                map11,
                                map12);
    cv::initUndistortRectifyMap(camera2.intrinsic_matrix, camera2.dist_coeffs, R2, P2, camera2.image_size, CV_16SC2,
                                map21,
                                map22);

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
            -64, 128, 11, 100, 1000, 32, 0, 15, 1000, 16, cv::StereoSGBM::MODE_HH);

    vector<string> file_names1;
    vector<string> file_names2;
    cv::glob(file_path1, file_names1);
    cv::glob(file_path2, file_names2);

    for (int i = 0; i < file_names1.size(); i++) {
        cv::Mat img1 = cv::imread(file_names1[i]);
        cv::Mat img2 = cv::imread(file_names2[i]);
        cv::Mat img1r, img2r, disp, vdisp;
        if (img1.empty() || img2.empty()) {
            continue;
        }

        cv::remap(img1, img1r, map11, map12, cv::INTER_LINEAR);
        cv::remap(img2, img2r, map21, map22, cv::INTER_LINEAR);
        stereo->compute(img1r, img2r, disp);
        cv::normalize(disp, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);
        cv::imshow("disparity", vdisp);

        cv::Mat pair_origin, pair_rectify;
        cv::hconcat(img1, img2, pair_origin);
        cv::hconcat(img1r, img2r, pair_rectify);
        for (int j = 0; j < img1.size().height; j += img1.size().height / 10) {
            cv::line(pair_rectify, cv::Point(0, j), cv::Point(img1.size().width * 2, j),
                     cv::Scalar(0, 255, 0));
            cv::line(pair_origin, cv::Point(0, j), cv::Point(img1.size().width * 2, j),
                     cv::Scalar(0, 255, 0));
        }
        cv::imshow("rectified", pair_rectify);
        cv::imshow("origin", pair_origin);
        if ((cv::waitKey() & 255) == 27)
            break;
    }

    return 0;
}
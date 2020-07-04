

#include "CameraCalibration.h"

CameraCalibration::CameraCalibration(int board_w, int board_h, string &file_path, bool show_corners_flag) {
    board_width = board_w;
    board_height = board_h;
    board_num = board_height * board_width;
    board_size = cv::Size(board_width, board_height);


    cv::glob(file_path, file_names
    );

    LoadImages(show_corners_flag);

}

void CameraCalibration::LoadImages(bool show_corners) {
    for (size_t i = 0; i < file_names.size(); i++) {
        cv::Mat image = cv::imread(file_names[i]);
        if (image.empty()) {
            cerr << file_names[i] << "is empty!" << endl;
        }
        image_size = image.size();
        vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(image, board_size, corners);
        if (show_corners) {
            cv::drawChessboardCorners(image, board_size, corners, found);
            cv::imshow("find_corners", image);
            cv::waitKey(100);
        }

        if (found) {
            image_points.push_back(corners);
            object_points.emplace_back();
            vector<cv::Point3f> &opts = object_points.back();

            opts.resize(board_num);
            for (int j = 0; j < board_num; j++) {
                opts[j] = cv::Point3f(static_cast<float>(j / board_width),
                                      static_cast<float>(j % board_width), 0.0f);
            }
        }
        cout << "Found" << image_points.size() << "total boards."
             << "This one from chessboard iamge #" << i + 1 << "," << file_names[i] << endl;
    }
}

void CameraCalibration::Calibrate() {
    double error = cv::calibrateCamera(object_points, image_points, image_size, intrinsic_matrix, dist_coeffs,
                                       cv::noArray(), cv::noArray(), cv::CALIB_ZERO_TANGENT_DIST);
    cout << "----- DONE! ----- Reprojection error is " << error << endl;
    cout << "\nimage width:" << static_cast<int>(image_size.width);
    cout << "\nimage height: " << static_cast<int>(image_size.height);
    cout << "\nintrinsic matrix:" << intrinsic_matrix;
    cout << "\ndistortion coefficients: " << dist_coeffs << "\n" << endl;
}

void CameraCalibration::Dedestrotion() {
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(intrinsic_matrix, dist_coeffs, cv::Mat(),
                                intrinsic_matrix, image_size, CV_16SC2, map1, map2);
    for (const auto &file_name : file_names) {
        cv::Mat image;
        cv::Mat image0 = cv::imread(file_name);
        if (image0.empty()) {
            cerr << file_name << " is empty!" << endl;
            continue;
        }
        cv::remap(image0, image, map1, map2, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT, cv::Scalar());
        cv::undistort(image0, image, intrinsic_matrix, dist_coeffs);
        cv::imshow("Original", image0);
        cv::imshow("Undistorted", image);
        cv::waitKey();
    }

}




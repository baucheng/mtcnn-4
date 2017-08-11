#include <vector>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <caffe/caffe.hpp>
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;
using std::ifstream;
using std::vector;
using caffe::Blob;
using namespace cv;

#define mydataFmt float
#define NumPoint   5

struct Bbox {
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
  float area;
  bool exist;
  mydataFmt ppoint[2 * NumPoint];
  mydataFmt regreCoord[4];

  operator Rect() {
    return Rect(x1, y1, x2-x1+1, y2-y1+1);
  }
};

struct orderScore {
  mydataFmt score;
  int oriOrder;
};

bool cmpScore(struct orderScore lsh, struct orderScore rsh) {
  return lsh.score < rsh.score;
}
void nms(vector<struct Bbox> &boundingBox_,
         vector<struct orderScore> &bboxScore_,
         const float overlap_threshold, string modelname) {
  if (boundingBox_.empty())
    return;

  std::vector<int> heros;
  //sort the score
  sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

  int order = 0;
  float IOU = 0;
  float maxX = 0;
  float maxY = 0;
  float minX = 0;
  float minY = 0;
  while (bboxScore_.size() > 0) {
    order = bboxScore_.back().oriOrder;
    bboxScore_.pop_back();
    if (order < 0) continue;
    heros.push_back(order);
    boundingBox_.at(order).exist = false; // delete it

    for (int num = 0; num < boundingBox_.size(); ++num) {
      if (boundingBox_.at(num).exist) {
        //the iou
        maxX = (boundingBox_.at(num).x1 > boundingBox_.at(order).x1) ?
               boundingBox_.at(num).x1 : boundingBox_.at(order).x1;
        maxY = (boundingBox_.at(num).y1 > boundingBox_.at(order).y1) ?
               boundingBox_.at(num).y1 : boundingBox_.at(order).y1;
        minX = (boundingBox_.at(num).x2 < boundingBox_.at(order).x2) ?
               boundingBox_.at(num).x2 : boundingBox_.at(order).x2;
        minY = (boundingBox_.at(num).y2 < boundingBox_.at(order).y2) ?
               boundingBox_.at(num).y2 : boundingBox_.at(order).y2;
        //maxX1 and maxY1 reuse
        maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
        maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
        //IOU reuse for the area of two bbox
        IOU = maxX * maxY;
        if (!modelname.compare("Union"))
          IOU = IOU / (boundingBox_.at(num).area +
                       boundingBox_.at(order).area - IOU);
        else if (!modelname.compare("Min")) {
          IOU = IOU / ((boundingBox_.at(num).area < boundingBox_.at(order).area) ?
                       boundingBox_.at(num).area : boundingBox_.at(order).area);
        }
        if (IOU > overlap_threshold) {
          boundingBox_.at(num).exist = false;
          for (vector<orderScore>::iterator it = bboxScore_.begin();
               it != bboxScore_.end(); it++) {
            if ((*it).oriOrder == num) {
              (*it).oriOrder = -1;
              break;
            }
          }
        }
      }
    }
  }
  for (int i = 0; i<heros.size(); i++)
    boundingBox_.at(heros.at(i)).exist = true;
}

vector<float> getScales(int w, int h, int minsize = 12) {
  float minl = h < w ? h : w;
  int MIN_DET_SIZE = 12;
  float m = (float)MIN_DET_SIZE / minsize;
  float factor = 0.709;
  vector<float> scales;

  while (minl * m > MIN_DET_SIZE) {
    scales.push_back(m);
    m *= factor;
  }
  return scales;
}

void cvMat2Blob(cv::Mat src, Blob<float>* input_layer,
                const float* mean, float scale) {
  int H = src.rows;
  int W = src.cols;
  assert(H * W == input_layer->height() * input_layer->width());
  float* pData = input_layer->mutable_cpu_data();
  int offset = H * W;
  //int H = H;
  //int W = W;
  for(int h = 0; h < H; ++h) {
    for(int w = 0; w < W; ++w) {
      pData[w * H + h + 0 * offset] =
          (float(src.at<cv::Vec3b>(h, w)[2]) - mean[0]) * scale;
      pData[w * H + h + 1 * offset] =
          (float(src.at<cv::Vec3b>(h, w)[1]) - mean[1]) * scale;
      pData[w * H + h + 2 * offset] =
          (float(src.at<cv::Vec3b>(h, w)[0]) - mean[2]) * scale;
    }
  }
}

/*
  #ifdef DEMO_0
  //这是一个简单的演示程序，用来看网络的结果是否正确
  void main() {
  float means[] = { 127.5f, 127.5f, 127.5f };
  Classifier c("deploy-12pnet.prototxt", "12pnet-_iter_145542.caffemodel",
  0.0078125, 0, 3, means, -1);
  Mat img = imread("300.jpg");
  //c.reshape(img.cols, img.rows);
  c.forward(img);
  WPtr<BlobData> cls = c.getBlobData("cls_loss");
  WPtr<BlobData> box = c.getBlobData("conv4-2");

  Mat cls_map(cls->height, cls->width, CV_32F, cls->list+cls->width * cls->height);
  }
  #endif

  #ifdef DEMO_1
  //这是ONet的简单演示程序
  void main() {
  float means[] = {127.5f,127.5f,127.5f};
  Classifier c("deploy-48onet.prototxt", "48onet-_iter_5726.caffemodel",
  0.0078125, 0, 3, means, 0);
  Mat img = imread("300.jpg");
  Rect roi(623, 170, 109, 109);
  Mat img2;
  resize(img(roi), img2, Size(48, 48));

  c.forward(img2);
  WPtr<BlobData> cls = c.getBlobData("cls_loss");
  float conf = cls->list[1];
  WPtr<BlobData> box = c.getBlobData("conv6-2");
  float lx = box->list[0];
  float ly = box->list[1];
  float rx = box->list[2];
  float ry = box->list[3];

  float newlx = roi.x + lx*roi.width;
  float newly = roi.y + ly*roi.height;
  float newrx = roi.x + rx*roi.width;
  float newry = roi.y + ry*roi.height;
  rectangle(img, Point(newlx, newly), Point(newrx, newry), Scalar(0, 255), 2);
  }
  #endif
*/
#define modelpath "../model_DuinoDu/"
#define modelpath_train "../model_48/"

int main(int argc, char** argv) {
  float means[] = {127.5f, 127.5f, 127.5f};
  float scale_param = 0.0078125;
  if (argc != 3) {
    cout << "Usage for example: *.bin 500000 path_list" << endl;
    return -1;
  }

  boost::shared_ptr<caffe::Net<float> > pnet(new caffe::Net<float>(
      modelpath"det1.prototxt", caffe::TEST));
  pnet->CopyTrainedLayersFrom(modelpath"det1.caffemodel");
  // (0.0078125, 0, 3, means, -1);
  boost::shared_ptr<caffe::Net<float> > rnet(new caffe::Net<float>(
      modelpath"det2.prototxt", caffe::TEST));
  rnet->CopyTrainedLayersFrom(modelpath"det2.caffemodel");
  // (0.0078125, 0, 3, means, -1);
  boost::shared_ptr<caffe::Net<float> > onet(new caffe::Net<float>(
      modelpath"det3.prototxt", caffe::TEST));
  std::string model_onet = std::string("../model_48/_iter_") +
                           argv[1] + std::string(".caffemodel");
  onet->CopyTrainedLayersFrom(model_onet.c_str());
  // (0.0078125, 0, 3, means, -1);

  // VideoCapture cap(0);
  ifstream in(argv[2]);
  string line;
  if (in) {
    while (getline(in, line)) {
      //double tick = 0;
      Mat img;
      Mat BGRImg = imread(line);
      //cv::cvtColor(BGRImg, img, cv::COLOR_BGR2RGB);
      //cout << "rows H = " << BGRImg.rows << endl;
      //cout << "cols W = " << BGRImg.cols << endl;
      Mat raw_img = BGRImg.clone();
      // cap >> raw_img;
      /*for (int a1 = 1; a1 < 5; ++a1) {
        for (int a2 = 1; a2 < 5; ++a2) {
        for (int a3 = 1; a3 < 5; ++a3) {
        for (int b1 = 1; b1 < 3; ++b1) {
        for (int b2 = 1; b2 < 3; ++b2) {
        for (int b3 = 1; b3 < 3; ++b3) {*/
      float confs[] = {0.5, 0.7, 0.7};
      float nmss[] = {0.5, 0.7, 0.7};
      /*  confs[0] = 0.2*a1;
          confs[1] = 0.2*a2;
          confs[2] = 0.2*a3;
          nmss[0] = 0.2*b1;
          nmss[1] = 0.2*b2;
          nmss[2] = 0.2*b3;*/
      // while for video
      while (!raw_img.empty()) {
        clock_t start = clock();

        // PNet
        vector<Bbox> pnetBoxAll;
        vector<orderScore> pnetOrderAll;
        int wnd = 12;
        vector<float> scales = getScales(raw_img.cols, raw_img.rows, wnd);
        for (int k = 0; k < scales.size(); ++k) {
          vector<Bbox> pnetBox;
          vector<orderScore> pnetOrder;
          float scale = scales[k];
          cv::resize(raw_img, img,
                     cv::Size(ceil(raw_img.cols * scale), ceil(raw_img.rows * scale)),
                     0, 0, cv::INTER_LINEAR);
          pnet->blob_by_name("data")->Reshape(1, 3, img.cols, img.rows);
          pnet->Reshape();
          cvMat2Blob(img, pnet->input_blobs()[0], means, scale_param);
          pnet->Forward();

          Blob<float>* cls_loss = pnet->blob_by_name("prob1").get();
          Blob<float>* box = pnet->blob_by_name("conv4-2").get();

          int stride = 2;
          //int num = cls_loss->num();
          //int channels = cls_loss->channels();
          int height = cls_loss->height();
          int width = cls_loss->width();
          for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
              // float conf = cls_loss->data_at(0, 1, i, j); //?
              const float conf = cls_loss->cpu_data()[1 * height * width + i*width + j];
              //printf("PNet: %f\n", conf);
              if (conf >= confs[0]) {
                printf("PNet: %f > %f\n", conf, confs[0]);
                // box should be fixed {
                int cellx = i;
                int celly = j;
                float raw_x = (cellx * stride) / scale;
                float raw_y = (celly * stride) / scale;
                float raw_r = (cellx * stride + wnd) / scale;
                float raw_b = (celly * stride + wnd) / scale;

                int width_box = box->width();
                int height_box = box->height();
                assert(width_box == width);
                assert(height_box == height);
                int plane_size = width_box * height_box;
                float x1 = box->cpu_data()[j + i * width_box + plane_size * 0];
                float y1 = box->cpu_data()[j + i * width_box + plane_size * 1];
                float x2 = box->cpu_data()[j + i * width_box + plane_size * 2];
                float y2 = box->cpu_data()[j + i * width_box + plane_size * 3];
                // printf("bbox: x1 = %f, y1 = %f, x2 = %f, y2 = %f\n", x1, y1, x2, y2);

                float raw_w = raw_r - raw_x + 1;
                float raw_h = raw_b - raw_y + 1;
                x1 = raw_x + x1 * raw_w;
                y1 = raw_y + y1 * raw_h;
                x2 = raw_r + x2 * raw_w;
                y2 = raw_b + y2 * raw_h;
                // printf("origin: x1 = %f, y1 = %f, x2 = %f, y2 = %f\n", x1, y1, x2, y2);

                raw_w = x2 - x1 + 1;
                raw_h = y2 - y1 + 1;

                Bbox b;
                b.area = raw_w * raw_h;
                b.exist = true;
                b.score = conf;
                b.x1 = x1;
                b.y1 = y1;
                b.x2 = x2;
                b.y2 = y2;
                pnetBox.push_back(b);

                orderScore order;
                order.oriOrder = pnetOrder.size();
                order.score = conf;
                pnetOrder.push_back(order);
                // }
              }
            }
          }

          nms(pnetBox, pnetOrder, 0.7, "Union");
          for (int i = 0; i < pnetBox.size(); ++i) {
            if (pnetBox[i].exist) {
              pnetBoxAll.push_back(pnetBox[i]);

              orderScore order;
              order.oriOrder = pnetOrderAll.size();
              order.score = pnetBox[i].score;
              pnetOrderAll.push_back(order);
            }
          }
        }
        nms(pnetBoxAll, pnetOrderAll, nmss[0], "Union");

        // RNet
        vector<Bbox> rnetBoxAll;
        vector<orderScore> rnetOrderAll;
        for (int i = 0; i < pnetBoxAll.size(); ++i) {
          if (pnetBoxAll[i].exist) {
            Rect cropBox = pnetBoxAll[i];
            // printf("pnetBoxAll[%d]: x1 = %f, y1 = %f, x2 = %f, y2 = %f\n", i, pnetBoxAll[i].x1, pnetBoxAll[i].y1, pnetBoxAll[i].x2, pnetBoxAll[i].y2);
            // printf("cropBox[%d]: x = %d, y = %d, width = %d, height = %d\n", i, cropBox.x, cropBox.y, cropBox.width, cropBox.height);
            int size = (cropBox.width + cropBox.height) * 0.5;
            int cx = cropBox.x + cropBox.width * 0.5;
            int cy = cropBox.y + cropBox.height * 0.5;
            cropBox.x = cx - size * 0.5;
            cropBox.y = cy - size * 0.5;
            cropBox.width = size;
            cropBox.height = size;

            cropBox = cropBox & Rect(0, 0, raw_img.cols, raw_img.rows);
            // printf("cropBox[%d]: x = %f, y = %f, width = %f, height = %f\n", i, cropBox.x, cropBox.y, cropBox.width, cropBox.height);
            if (cropBox.width <= 0 && cropBox.height <= 0)
              continue;
            // printf("cropBox, width = %f, height = %f\n", cropBox.width, cropBox.height);
            // printf("cropBox, width = %d, height = %d\n", int(cropBox.width), int(cropBox.height));
            Mat img2;
            cv::resize(raw_img(cropBox), img2, cv::Size(24, 24));
            rnet->blob_by_name("data")->Reshape(1, 3, img2.cols, img2.rows);
            rnet->Reshape();
            cvMat2Blob(img2, rnet->input_blobs()[0], means, scale_param);
            rnet->Forward();
            Blob<float>* cls_loss = rnet->blob_by_name("prob1").get();
            //int num = cls_loss->num();
            //int channels = cls_loss->channels();
            //float conf = cls_loss->data_at(0, 1, 0, 0);
            int height = cls_loss->height();
            int width = cls_loss->width();
            const float conf = cls_loss->cpu_data()[1 * height * width];
            //printf("RNet: %f\n", conf);
            if (conf > confs[1]) {
              printf("RNet: %f > %f\n", conf, confs[1]);
              // box should be fixed {
              Blob<float>* box = rnet->blob_by_name("conv5-2").get();
              float lx = box->cpu_data()[0];
              float ly = box->cpu_data()[1];
              float rx = box->cpu_data()[2];
              float ry = box->cpu_data()[3];

              float newlx = cropBox.x + lx*cropBox.width;
              float newly = cropBox.y + ly*cropBox.height;
              float newrx = cropBox.x + cropBox.width + rx*cropBox.width;
              float newry = cropBox.y + cropBox.height + ry*cropBox.height;
              //rectangle(raw_img, Point(newlx, newly), Point(newrx, newry),
              //Scalar(0, 255), 2);

              orderScore order;
              order.oriOrder = rnetOrderAll.size();
              order.score = conf;
              rnetOrderAll.push_back(order);

              Bbox b;
              b = pnetBoxAll[i];
              b.x1 = newlx;
              b.y1 = newly;
              b.x2 = newrx;
              b.y2 = newry;
              b.score = conf;
              b.exist = true;
              b.area = (b.x2 - b.x1)*(b.y2 - b.y1);
              rnetBoxAll.push_back(b);
              // }
            }
          }
        }
        nms(rnetBoxAll, rnetOrderAll, nmss[1], "Union");

        // ONet
        vector<Bbox> onetBoxAll;
        vector<orderScore> onetOrderAll;
        for (int i = 0; i < rnetBoxAll.size(); ++i) {
          if (rnetBoxAll[i].exist) {
            Rect cropBox = rnetBoxAll[i];
            int size = (cropBox.width + cropBox.height) * 0.5;
            int cx = cropBox.x + cropBox.width * 0.5;
            int cy = cropBox.y + cropBox.height * 0.5;
            cropBox.x = cx - size * 0.5;
            cropBox.y = cy - size * 0.5;
            cropBox.width = size;
            cropBox.height = size;

            cropBox = cropBox & Rect(0, 0, raw_img.cols, raw_img.rows);
            if (cropBox.width <= 0 && cropBox.height <= 0)
              continue;

            Mat img3;
            cv::resize(raw_img(cropBox), img3, cv::Size(48, 48));
            onet->blob_by_name("data")->Reshape(1, 3, img3.cols, img3.rows);
            onet->Reshape();
            cvMat2Blob(img3, onet->input_blobs()[0], means, scale_param);
            onet->Forward();
            Blob<float>* cls_loss = onet->blob_by_name("prob1").get();
            // int num = cls_loss->num();
            //int channels = cls_loss->channels();
            //float conf = cls_loss->data_at(0, 1, 0, 0);
            int height = cls_loss->height();
            int width = cls_loss->width();
            const float conf = cls_loss->cpu_data()[1 * height * width];
            //printf("ONet: %f\n", conf);
            if (conf > confs[2]) {
              printf("ONet: %f > %f\n", conf, confs[2]);
              // box should be fixed {
              Blob<float>* box = onet->blob_by_name("conv6-2").get();
              float lx = box->cpu_data()[0];
              float ly = box->cpu_data()[1];
              float rx = box->cpu_data()[2];
              float ry = box->cpu_data()[3];
              printf("lx = %f ", lx);
              printf("ly = %f ", ly);
              printf("rx = %f ", rx);
              printf("ry = %f ", ry);
              printf("\n");
              float newlx = cropBox.x + lx*cropBox.width;
              float newly = cropBox.y + ly*cropBox.height;
              float newrx = cropBox.x + cropBox.width + rx*cropBox.width;
              float newry = cropBox.y + cropBox.height + ry*cropBox.height;
              //rectangle(raw_img, Point(newlx, newly), Point(newrx, newry), Scalar(0, 255), 2);

              orderScore order;
              order.oriOrder = onetOrderAll.size();
              order.score = conf;
              onetOrderAll.push_back(order);

              //如果有坐标就使用他
              //WPtr<BlobData> pts = onet.getBlobData("conv6-3");
              Blob<float>* pts = onet->blob_by_name("conv6-3").get();
              Bbox b;
              b = rnetBoxAll[i];
              b.x1 = newlx;
              b.y1 = newly;
              b.x2 = newrx;
              b.y2 = newry;
              b.score = conf;
              b.exist = true;
              b.area = (b.x2 - b.x1)*(b.y2 - b.y1);

              for (int k = 0; k < NumPoint; ++k) {
                int xp = k * 2;
                int yp = k * 2 + 1;
                float x = pts->cpu_data()[k] * cropBox.width + cropBox.x;
                float y = pts->cpu_data()[k+5] * cropBox.height + cropBox.y;
                b.ppoint[xp] = x;
                b.ppoint[yp] = y;
                printf("x[%d] = %f  ", k, pts->cpu_data()[k]);
                printf("y[%d] = %f\n", k, pts->cpu_data()[k+5]);
              }

              onetBoxAll.push_back(b);
              // }
            }
          }
        }
        nms(onetBoxAll, onetOrderAll, nmss[2], "Min");

        for (int i = 0; i < onetBoxAll.size(); ++i) {
          if (onetBoxAll[i].exist) {
            cv::rectangle(raw_img, onetBoxAll[i], Scalar(0, 255), 2);
            for (int j = 0; j < NumPoint; ++j) {
              cv::circle(raw_img, cv::Point2d(onetBoxAll[i].ppoint[2 * j], onetBoxAll[i].ppoint[2 * j + 1]),2,cv::Scalar(0,0,255),-1);
            }
          }
        }

        clock_t end = clock();
        printf("times: %.2f ms\n", (double)(end - start) / CLOCKS_PER_SEC / 1000);
        //printf("%d, %d, %d, %d, %d, %d \n", a1, a2, a3, b1, b2, b3);
        imshow("video-demo", raw_img);
        waitKey(0);

        break;
        //cap >> raw_img;
      }
    }
  }
  return 0;
}

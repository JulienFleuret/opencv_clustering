//Copyright 2018 University Laval, CVSL-MIVIM

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: Julien FLEURET, julien.fleuret.1@ulaval.ca


#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>

#include "clustering.h"

#include <chrono>

class chrono_t
{
private:

    std::chrono::steady_clock::time_point _start;
    std::chrono::steady_clock::time_point _stop;

public:

    chrono_t() = default;

    chrono_t(const chrono_t&) = delete;
    chrono_t(chrono_t&&) = delete;

    ~chrono_t() = default;

    chrono_t& operator =(const chrono_t&) = delete;
    chrono_t& operator =(chrono_t&&) = delete;

    inline void start()
    {
        this->_start = std::chrono::steady_clock::now();
    }

    inline void stop()
    {
        this->_stop = std::chrono::steady_clock::now();
    }

    inline std::intmax_t getms()const{ return std::chrono::duration_cast<std::chrono::milliseconds>(this->_stop - this->_start).count();}
    inline std::intmax_t getmus()const{ return std::chrono::duration_cast<std::chrono::microseconds>(this->_stop - this->_start).count();}
};

chrono_t chrono;



void test_kmeans(); // for evaluate mini batch kmeans modify this function.

void test_dbscan();



int main( int /*argc*/, char** /*argv*/ )
{



    test_dbscan();
    test_kmeans();

    return EXIT_SUCCESS;
}

float normL2(const float* a, const float* b, int dims)
{
    return std::sqrt(cv::normL2Sqr(a,b,dims));
}

void test_kmeans()
{

    const int MAX_CLUSTERS = 5;
    cv::Scalar colorTab[] =
    {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0,255,0),
        cv::Scalar(255,100,100),
        cv::Scalar(255,0,255),
        cv::Scalar(0,255,255)
    };

    cv::Mat img(500, 500, CV_8UC3);
    cv::RNG rng(12345);

    for(;;)
    {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
        int i, sampleCount = rng.uniform(1, 1000001);
        cv::Mat points(sampleCount, 1, CV_32FC2), labels;

        clusterCount = std::min(clusterCount, sampleCount);
       cv::Mat centers;

        /* generate random sample from multigaussian distribution */
        for( k = 0; k < clusterCount; k++ )
        {
            cv::Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            cv::Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                             k == clusterCount - 1 ? sampleCount :
                                             (k+1)*sampleCount/clusterCount);
            rng.fill(pointChunk, cv::RNG::NORMAL, cv::Scalar(center.x, center.y), cv::Scalar(img.cols*0.05, img.rows*0.05));
        }

        randShuffle(points, 1, &rng);

        chrono.start();

//        double compactness = cv::kmeans(points, clusterCount, labels,
//            cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
//               3, cv::KMEANS_PP_CENTERS, centers);

//        double compactness = clustering::kmeans(points, clusterCount, labels,
//            cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
//               3, cv::KMEANS_PP_CENTERS, centers);

        double compactness = clustering::mini_batch_kmeans(points, clusterCount, 1024, labels,
            cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

        chrono.stop();

        img = cv::Scalar::all(0);

        centers = centers.reshape(2);

//        std::cout<<labels<<std::endl;

        for( i = 0; i < sampleCount; i++ )
        {
            int clusterIdx = labels.at<int>(i);

            if(clusterIdx<0)
                continue;

//            std::cout<<"CHECK "<<clusterIdx<<std::endl;
            cv::Point ipt = points.at<cv::Point2f>(i);
            cv::circle( img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA );
        }
        for (i = 0; i < (int)centers.rows; ++i)
        {
            cv::Point2f c = centers.at<cv::Point2f>(i);
            cv::circle( img, c, 40, colorTab[i], 1, cv::LINE_AA );
        }
        std::cout << "Compactness: " << compactness << std::endl;
        std::cout << "Processing time: " << chrono.getmus() << std::endl;
        std::cout << "Number of points: " << sampleCount << std::endl;
        std::cout << "Number of cluster to find: " << clusterCount << std::endl;
        std::cout << std::endl;

        cv::imshow("clusters", img);

        char key = (char)cv::waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }


}

void test_dbscan()
{

    const int MAX_CLUSTERS = 5;

    std::vector<cv::Scalar> _colorTab(1024);
    cv::Scalar* colorTab = _colorTab.data();

    for(int i=0;i<1024;i++)
    {
        cv::Scalar color(cv::theRNG().uniform(0,255), cv::theRNG().uniform(0,255), cv::theRNG().uniform(0,255));

        colorTab[i] = color;
    }

       cv::Mat img(500, 500, CV_8UC3);
       cv::RNG rng(12345);

       for(;;)
       {
           int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
           int i, sampleCount = rng.uniform(100, 1001);
           cv::Mat points(sampleCount, 1, CV_32FC2), labels;

           clusterCount = MIN(clusterCount, sampleCount);
           cv::Mat centers;

           /* generate random sample from multigaussian distribution */
           for( k = 0; k < clusterCount; k++ )
           {
               cv::Point center;
               center.x = rng.uniform(0, img.cols);
               center.y = rng.uniform(0, img.rows);
               cv::Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                                k == clusterCount - 1 ? sampleCount :
                                                (k+1)*sampleCount/clusterCount);
               rng.fill(pointChunk, cv::RNG::NORMAL, cv::Scalar(center.x, center.y), cv::Scalar(img.cols*0.05, img.rows*0.05));
           }

           cv::randShuffle(points, 1, &rng);


           double compactness = clustering::dbscan(points, rng.uniform(25,45), rng.uniform(1,10), labels, centers);


           img = cv::Scalar::all(0);

           for( i = 0; i < sampleCount; i++ )
           {
               int clusterIdx = labels.at<int>(i);
               cv::Point ipt = points.at<cv::Point2f>(i);
               cv::circle( img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA );
           }

           centers = centers.reshape(2);

           for (i = 0; i < centers.rows; ++i)
           {
               cv::Point2f c = centers.at<cv::Point2f>(i);
               cv::circle( img, c, 40, colorTab[i], 1, cv::LINE_AA );
           }

           std::cout << "Compactness: " << compactness << std::endl;

           cv::imshow("clusters", img);

           char key = (char)cv::waitKey();
           if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
               break;
       }

}

//Copyright 2018 University Laval, CVSL-MIVIM

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: Julien FLEURET, julien.fleuret.1@ulaval.ca


#include "clustering.h"

#include <opencv2/core/cv_cpu_helper.h>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/hal.hpp>

#include <list>

namespace clustering
{

namespace
{

class DBSCANProcessNeighbourhoodIndex : public cv::ParallelLoopBody
{
private:

    const cv::Mat1f data;
    const float eps;
    std::vector<cv::Mat1i>& neighoudhood;
    cv::Mat1i& labels;
    cv::Mat1b& visited;
    const int minPts;

public:

    inline DBSCANProcessNeighbourhoodIndex(const cv::Mat1f& _data, const float& _eps, std::vector<cv::Mat1i>& _neighoudhood, cv::Mat1i& _labels, cv::Mat1b& _visited, const int& _minPts):
        data(_data),
        eps(_eps),
        neighoudhood(_neighoudhood),
        labels(_labels),
        visited(_visited),
        minPts(_minPts)
    {}

    virtual ~DBSCANProcessNeighbourhoodIndex() = default;

    virtual void operator()(const cv::Range& range)const
    {

        std::list<int> lst;

        float leps = this->eps;

        leps*=leps;

        const int dims = this->data.cols;

        for(int r=range.start; r<range.end; r++)
        {

            const float* point = this->data[r];

            for(int c=0;c<this->data.rows;c++)
            {
                if(c==r)
                    continue;

                const float* current = this->data[c];

                if(cv::normL2Sqr(point,current,dims) < leps )
                    lst.push_back(c);
            }


            if(lst.size() < this->minPts)
            {
                this->visited(r) = 0xFF;
                this->labels(r) = -1; // NOISE.
            }
            else
            {
                cv::Mat1i tmp(1, lst.size());

                std::copy(lst.begin(), lst.end(), tmp.begin());

                this->neighoudhood.at(r) = tmp;
            }

            lst.clear();
        }

    }

};

class DBSCANCalculateCentres : public cv::ParallelLoopBody
{
private:

    cv::Mat1f data;
    cv::Mat1i labels;
    cv::Mat1f& centres;

    mutable cv::Mutex mtx;

public:

    inline DBSCANCalculateCentres(const cv::Mat1f& _data,const cv::Mat1i& _labels, cv::Mat1f& _centres):
        data(_data),
        labels(_labels),
        centres(_centres)
    {}

    virtual ~DBSCANCalculateCentres() = default;

    virtual void operator()(const cv::Range& range)const
    {

        cv::Mat1f lcentres = cv::Mat1f::zeros(this->centres.size());
        cv::Mat1f lweight = cv::Mat1f::zeros(1, this->centres.rows);

        const int dims = lcentres.cols;

        for(int r=range.start;r<range.end;r++)
        {
            if(this->labels(r)!=-1 && this->labels(r)!=0)
            {
                int cidx = this->labels(r)-1;

                const float* left = this->data[r];
                const float* right = lcentres[cidx];
                float* dst = lcentres[cidx];

                cv::hal::add32f(left, 0, right, 0, dst, 0, dims, 1, nullptr);
                lweight(cidx)++;
            }
        }

        for(int r=0;r<lcentres.rows ;r++)
        {
            float w = lweight(r);

            int c=0;

            float* cptr = lcentres[r];

#if CV_SIMD128
            cv::v_float32x4 v_w = cv::v_setall_f32(w);

            for(;c<=dims-4; c+=4, cptr+=4)
            {
                cv::v_float32x4 v = cv::v_load(cptr);

                v/=v_w;

                cv::v_store(cptr, v);
            }
#else

#if CV_ENABLE_UNROLLED

            for(;c<=dims-4;c+=4, cptr+=4)
            {
                float v0 = cptr[0] / w;
                float v1 = cptr[1] / w;

                cptr[0] = v0;
                cptr[1] = v1;

                v0 = cptr[2] / w;
                v1 = cptr[3] / w;

                cptr[2] = v0;
                cptr[3] = v1;
            }

#endif
#endif

            for(;c<dims;c++, cptr++)
                cptr[0] /= w;
        }






        cv::AutoLock lck(this->mtx);

        cv::hal::add32f(lcentres[0], lcentres.step, this->centres[0], this->centres.step, this->centres[0], this->centres.step, lcentres.cols, lcentres.rows, nullptr);

        for(int r=0;r<this->centres.rows ;r++)
        {
            float w = 0.5f;

            int c=0;

            float* cptr = this->centres[r];

#if CV_SIMD128
            cv::v_float32x4 v_w = cv::v_setall_f32(w);

            for(;c<=dims-4;c+=4, cptr+=4)
            {
                cv::v_float32x4 v = cv::v_load(cptr);

                v*=v_w;

                cv::v_store(cptr, v);
            }
#else
#if CV_ENABLE_UNROLLED

            for(;c<=dims-4;c+=4, cptr+=4)
            {
                float v0 = cptr[0] * w;
                float v1 = cptr[1] * w;

                cptr[0] = v0;
                cptr[1] = v1;

                v0 = cptr[2] * w;
                v1 = cptr[3] * w;

                cptr[2] = v0;
                cptr[3] = v1;
            }

#endif
#endif
            for(;c<dims; c++, cptr++)
                cptr[0] *= w;
        }
    }


};

#if USE_COMPACTNESS
class DBSCANCalculateCompactness : public cv::ParallelLoopBody
{
private:

    static const int NOISE = -1;
    static const int UNCLASSIFIED = 0;

    cv::Mat1f data;
    cv::Mat1i labels;
    cv::Mat1f centres;

    cv::Mat1d& compactness;

public:

    inline DBSCANCalculateCompactness(const cv::Mat1f& _data, const cv::Mat1i& _labels, const cv::Mat1f& _centres, cv::Mat1d& _compactness):
        data(_data),
        labels(_labels),
        centres(_centres),
        compactness(_compactness)
    {}

    virtual ~DBSCANCalculateCompactness() = default;

    virtual void operator()(const cv::Range& range)const
    {

        const int dims = this->centres.cols;


        for(int r=range.start; r<range.end; r++)
        {
            int cidx = this->labels(r);

            if(cidx != NOISE && cidx != UNCLASSIFIED)
            {
                cidx--;

                const float* point = this->data[r];
                const float* centre = this->centres[cidx];

                this->compactness(r) = static_cast<double>(cv::hal::normL2Sqr_(point, centre, dims));
            }
        }
    }

};
#endif

}

double dbscan(cv::InputArray _data, const float &neighbourhoud_distance, const int &minPts, cv::OutputArray _labels, cv::OutputArray _centres)
{

    static const int NOISE = -1;
    static const int UNCLASSIFIED = 0;

    cv::Mat data0 = _data.getMat();
    bool isrow = data0.rows == 1;
    int N = isrow ? data0.cols : data0.rows;
    int dims = (isrow ? 1 : data0.cols)*data0.channels();
    int type = data0.depth();


    CV_Assert( data0.dims <= 2 && type == CV_32F  );

    cv::Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

    cv::Mat1i labels(1, N, UNCLASSIFIED);
    cv::Mat1b visited = cv::Mat1b::zeros(1, N);

    std::vector<cv::Mat1i> regionQuery(N);

    // Process the neigbourhood for every points and also label the noisy points.
    cv::parallel_for_(cv::Range(0, N), DBSCANProcessNeighbourhoodIndex(data, neighbourhoud_distance, regionQuery, labels, visited, minPts), 0x400);


    for(int i=0, clusterId=1;i<data.rows; i++)
    {
        if(visited(i))
            continue;

        int lcID = clusterId++;

        labels(i) = lcID;

        cv::Mat1i neighbours = regionQuery.at(i);

        for(int idx : neighbours)
        {
            visited(idx) = 0xFF;

            if(labels(idx) != NOISE)
            {
                cv::Mat1i neighbours_of_neighbours = regionQuery.at(idx);

                for(int idx2 : neighbours_of_neighbours)
                    if(labels(idx2) == UNCLASSIFIED)
                        labels(idx2) = lcID;
            }
        }
    }


    if(_labels.needed())
        labels.copyTo(_labels);




    double max(0.);

    cv::minMaxIdx(labels, nullptr, &max);

    CV_Assert(max>0);

    cv::Mat1f centres(max, dims, 0.f);

    cv::parallel_for_(cv::Range(0, N),DBSCANCalculateCentres(data, labels, centres), 0x80);

    if(_centres.needed())
        centres.copyTo(_centres);

    // The compactness doesn't work like it should ... :(. The macro prevent it's processing in order to save computational ressourses.
#if USE_COMPACTNESS

    cv::Mat1d dist(data.rows,1, 0.);

    cv::parallel_for_(cv::Range(0, N), DBSCANCalculateCompactness(data, labels, centres, dist), 0x80);

    double compactness = cv::sum(dist)(0);

    return std::isnan(compactness) ? 0. : compactness;
#else
    return 0.;
#endif

}

}

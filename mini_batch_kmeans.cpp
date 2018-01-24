//Copyright 2018 University Laval, CVSL-MIVIM

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This file is deeply insipired by OpenCV's KMeans implementation.

// Author: Julien FLEURET, julien.fleuret.1@ulaval.ca


#include "clustering.h"
#include "clustering.priv.hpp"

#include <numeric>
#include <atomic>

#include <opencv2/core/cv_cpu_helper.h>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/hal.hpp>

#include <iostream>


namespace clustering
{

namespace
{

#if CV_AVX

inline int update_center_avx(const float*& center, const float& beta, const float*& sample, const float& alpha, float*& dst, const int& dims)
{
    static const bool useAVX = cv::checkHardwareSupport(CV_CPU_AVX);

    if(!useAVX)
        return 0.f;

    int i=0;

    const __m256 v_alpha256 = _mm256_set1_ps(alpha);
    const __m256 v_beta256 = _mm256_set1_ps(beta);

    for(;i<=dims-8;i+=8, center+=8, sample+=8, dst+=8)
    {
        __m256 v_c = _mm256_loadu_ps(center);
        __m256 v_s = _mm256_loadu_ps(sample);

        v_s = _mm256_mul_ps(v_alpha256,v_s);
#if CV_FMA3
        __m256 v_d = _mm256_fmadd_ps(v_c, v_beta256, v_s);
#else
        v_c = _mm256_mul_ps(v_c, v_beta256);
        __m256 v_d = _mm256_add_ps(v_s,v_c);
#endif

        _mm256_storeu_ps(dst, v_d);
    }

    return i;
}

#else

inline int update_center_avx(const float*& center, const float& beta, const float*& sample, const float& alpha, float*& dst, const int& dims)
{
    return 0;
}
#endif


inline int update_center_intrin(const float *& center, const float &beta, const float *& sample, const float &alpha, float *& dst, const int &dims)
{
    int i=0;
#if CV_SIMD128

    // The new vector type since opencv 3.2 such as v_float32x4 are able to adapt themself to the architechture intrinsic.
    // If the architecure is unknow the instruction will allow the compiler auto vectorize to do its job efficiently.

    const cv::v_float32x4 v_alpha = cv::v_setall_f32(alpha);
    const cv::v_float32x4 v_beta = cv::v_setall_f32(beta);

    for(;i<=dims-4; i+=4, center+=4, sample+=4, dst+=4)
    {
        cv::v_float32x4 v_c = cv::v_load(center);
        cv::v_float32x4 v_s = cv::v_load(sample);

        v_s*=v_alpha;

#if CV_SSE && CV_FMA3
        v_c.val = _mm_fmadd_ps(v_c.val, v_beta.val, v_s.val);
#else
        v_c = v_c * v_beta + v_s;
#endif

        cv::v_store(dst, v_c);
    }

#else // CV_SIMD128

    for(;i<=dims-4; i+=4)
    {
#ifdef FP_FAST_FMAF
        dst[i] = std::fmaf(center[i], beta, sample[i] * alpha);
        dst[i+1] = std::fmaf(center[i+1], beta, sample[i+1] * alpha);
        dst[i+2] = std::fmaf(center[i+2], beta, sample[i+2] * alpha);
        dst[i+3] = std::fmaf(center[i+3], beta, sample[i+3] * alpha);
#else
        dst[i] = center[i] * beta + sample[i] * alpha;
        dst[i+1] = center[i+1] * beta + sample[i+1] * alpha;
        dst[i+2] = center[i+2] * beta + sample[i+2] * alpha;
        dst[i+3] = center[i+3] * beta + sample[i+3] * alpha;
#endif
    }

    center+=i;
    sample+=i;
    dst+=i;

#endif

    return i;
}

void update_center(const float* center, const float& beta, const float* sample, const float& alpha, float* dst, int dims)
{
    int i= update_center_avx(center, beta, sample, alpha, dst, dims) + update_center_intrin(center, beta, sample, alpha, dst, dims);


    for(;i<dims;i++, center++, sample++, dst++)
#ifdef FP_FAST_FMAF
        *dst = std::fmaf(*center, beta, *sample * alpha);
#else
        *dst = *center * beta + *sample * alpha;
#endif

}


//class MiniBatchKMeansUpdateCentres : public cv::ParallelLoopBody
//{

//public:

//    inline MiniBatchKMeansUpdateCentres(const cv::Mat& _labels, const cv::Mat& _data, const cv::Mat1i& _index, cv::Mat& _centres, std::vector<int>& _counters):
//        counters(reinterpret_cast<std::atomic_int*>(_counters.data())),
//        labels(_labels),
//        data(_data),
//        index(_index),
//        centers(_centres)
//    {}

//    virtual ~MiniBatchKMeansUpdateCentres() = default;

//    virtual void operator ()(const cv::Range& range)const
//    {

//        static const int dims = this->data.cols;

//        cv::Mat lcenters = cv::Mat::zeros(this->centers.size(), this->centers.type());

//        double scalars[3] = {0.,0.,1.};

//        for(int i=range.start; i<range.end; i++)
//        {

//            const int idx = index(i);

//            const float* sample = data.ptr<float>(idx);
//            int k = labels.at<int>(idx);
//            float* center = lcenters.ptr<float>(k);



//            float alpha = counters[k] ? 1.f/static_cast<float>(counters[k]) : 1.f;
//            float beta = 1.f - alpha;

//            update_center(center, beta, sample, alpha, center, dims);

////            scalars[0] = beta;
////            scalars[1] = alpha;

////            cv::hal::addWeighted32f(center, 0, sample, 0, center, 0, dims, 1, scalars);

//            counters[k]++;
//        }


//        cv::AutoLock lck(this->mtx);

//        scalars[0] = scalars[1] = 0.5;

//        cv::hal::addWeighted32f(lcenters.ptr<float>(), lcenters.step,
//                                this->centers.ptr<float>(), this->centers.step,
//                                this->centers.ptr<float>(), this->centers.step,
//                                lcenters.cols, lcenters.rows, scalars);
//    }

//private:

//    std::atomic_int* counters;

//    const cv::Mat& labels;
//    const cv::Mat& data;
//    const cv::Mat1i& index;
//    cv::Mat& centers;

//    mutable cv::Mutex mtx;


//};


class MiniBatchKMeansFindLabel : public cv::ParallelLoopBody
{
public:

    typedef float(*function_type)(const float*, const float*, int);

    inline MiniBatchKMeansFindLabel(const cv::Mat& _data,
                           const cv::Mat& _labels,
                           const cv::Mat1i& _index,
                           const float* old_center,
                           const int& _max_k,
                           double& _max_dist,
                           int& _farest_i,
                           const function_type _fun):
        data(_data),
        labels(_labels),
        index(_index),
        _old_center(old_center),
        max_k(_max_k),
        max_dist(_max_dist),
        farest_i(_farest_i),
        fun(_fun)
    {}

    virtual ~MiniBatchKMeansFindLabel() = default;

    virtual void operator()(const cv::Range& range)const
    {
        static const int dims = this->data.cols;

        double lmax_dist = 0.;
        int lfarthest_i = 0;

        if(this->fun)
        {
            for(int i=range.start; i<range.end; i++)
            {
                const int idx = this->index(i);

                if( labels.at<int>(idx) != max_k )
                    continue;
                const float* sample = data.ptr<float>(idx);

                double dist = fun(sample, _old_center, dims);
                dist*=dist;

                if( lmax_dist <= dist )
                {
                    lmax_dist = dist;
                    lfarthest_i = idx;
                }
            }
        }
        else
        {
            for(int i=range.start; i<range.end; i++)
            {
                const int idx = this->index(i);

                if( labels.at<int>(idx) != max_k )
                    continue;

                const float* sample = data.ptr<float>(idx);

                double dist = cv::normL2Sqr(sample, _old_center, dims);

                if( lmax_dist <= dist )
                {
                    lmax_dist = dist;
                    lfarthest_i = idx;
                }
            }
        }

        cv::AutoLock lck(this->mtx);

        if(this->max_dist <= lmax_dist)
        {
            this->max_dist = lmax_dist;
            this->farest_i = lfarthest_i;
        }
    }

private:

    const cv::Mat& data;
    const cv::Mat& labels;
    const cv::Mat1i& index;
    const float* _old_center;
    const int& max_k;
    double& max_dist;
    int& farest_i;
    const function_type fun;

    mutable cv::Mutex mtx;


};



class MiniBatchKMeansDistanceComputer : public cv::ParallelLoopBody
{
public:

    typedef float(*function_type)(const float*, const float*, int);

    inline MiniBatchKMeansDistanceComputer( double *_distances,
                            int *_labels,
                            const cv::Mat& _data,
                            const cv::Mat& _centers,
                            const cv::Mat1i& _index,
                            const function_type _fun,
                            bool _onlyDistance = false )
        : distances(_distances),
          labels(_labels),
          data(_data),
          centers(_centers),
          onlyDistance(_onlyDistance),
          fun(_fun),
          index(_index)
    {
    }

    virtual ~MiniBatchKMeansDistanceComputer() = default;

    virtual void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;
        const int K = centers.rows;
        const int dims = centers.cols;

        if(this->fun)
        {
            for( int i = begin; i<end; ++i)
            {
                const int idx = this->index(i);

                const float *sample = data.ptr<float>(idx);
                if (onlyDistance)
                {
                    //                std::cout<<"LABELS "<<labels[idx]<<std::endl;

                    int lbl = std::max(labels[idx],0);

                    const float* center = centers.ptr<float>(lbl < centers.rows ? lbl : centers.rows-1);
                    //                distances[i] = cv::normL2Sqr(sample, center, dims);
                    const double tmp = this->fun(sample, center, dims);
                    distances[i] = tmp*tmp;
                    continue;
                }
                int k_best = 0;
                double min_dist = std::numeric_limits<double>::max();

                for( int k = 0; k < K; k++ )
                {
                    const float* center = centers.ptr<float>(k);
                    //                const double dist = cv::normL2Sqr(sample, center, dims);
                    double dist = this->fun(sample, center, dims);
                    dist*=dist;

                    if( min_dist > dist )
                    {
                        min_dist = dist;
                        k_best = k;
                    }
                }

                distances[i] = min_dist;
                labels[idx] = k_best;
            }
        }
        else
        {
            for( int i = begin; i<end; ++i)
            {
                const int idx = this->index(i);

                const float *sample = data.ptr<float>(idx);
                if (onlyDistance)
                {
                    //                std::cout<<"LABELS "<<labels[idx]<<std::endl;

                    int lbl = std::max(labels[idx],0);

                    const float* center = centers.ptr<float>(lbl < centers.rows ? lbl : centers.rows-1);
                    //                distances[i] = cv::normL2Sqr(sample, center, dims);
                    double dist = cv::normL2Sqr(sample, center, dims);
                    distances[i] = dist;
                    continue;
                }
                int k_best = 0;
                double min_dist = std::numeric_limits<double>::max();

                for( int k = 0; k < K; k++ )
                {
                    const float* center = centers.ptr<float>(k);
                    //                const double dist = cv::normL2Sqr(sample, center, dims);
                    double dist = cv::normL2Sqr(sample, center, dims);

                    if( min_dist > dist )
                    {
                        min_dist = dist;
                        k_best = k;
                    }
                }

                distances[i] = min_dist;
                labels[idx] = k_best;
            }
        }
    }

private:
    MiniBatchKMeansDistanceComputer& operator=(const MiniBatchKMeansDistanceComputer&); // to quiet MSVC

    double *distances;
    int *labels;
    const cv::Mat& data;
    const cv::Mat& centers;
    bool onlyDistance;
    const function_type fun;
    const cv::Mat1i& index;
};

class MiniBatchKMeans_Assignment : public cv::ParallelLoopBody
{
public:

    typedef float(*function_type)(const float*, const float*, int);

private:

    const cv::Mat& data;
    const cv::Mat& centres;
    cv::Mat& labels;
    const function_type fun;

public:

    inline MiniBatchKMeans_Assignment(const cv::Mat& _data, const cv::Mat& _centres, cv::Mat& _labels, const function_type _fun):
        data(_data),
        centres(_centres),
        labels(_labels),
        fun(_fun)
    {}

    virtual ~MiniBatchKMeans_Assignment() = default;

    virtual void operator ()(const cv::Range& range)const
    {
//        std::cout<<"CHECK FUNCTION"<<std::endl;

        if(this->fun)
        {
            for(int r=range.start; r<range.end; r++)
            {
                if(this->labels.at<int>(r) < 0 || this->labels.at<int>(r)>=this->centres.rows)
                {
                    const float* sample = this->data.ptr<float>(r);

                    double dist = this->fun(sample, this->centres.ptr<float>(), this->centres.cols);
                    dist*=dist;
                    int idx=0;

                    for(int k=1; k<this->centres.rows;k++)
                    {
                        double dist2 = this->fun(sample, this->centres.ptr<float>(k), this->centres.cols);
                        dist2*=dist2;

                        if(dist2<dist)
                        {
                            dist = dist2;
                            idx = k;
                        }
                    }

                    this->labels.at<int>(r) = idx;
                }
            }
        }
        else
        {
            for(int r=range.start; r<range.end; r++)
            {
                if(this->labels.at<int>(r) < 0 || this->labels.at<int>(r)>=this->centres.rows)
                {
                    const float* sample = this->data.ptr<float>(r);

                    double dist = cv::normL2Sqr(sample, this->centres.ptr<float>(), this->centres.cols);

                    int idx=0;

                    for(int k=1; k<this->centres.rows;k++)
                    {
                        double dist2 = cv::normL2Sqr(sample, this->centres.ptr<float>(k), this->centres.cols);

                        if(dist2<dist)
                        {
                            dist = dist2;
                            idx = k;
                        }
                    }

                    this->labels.at<int>(r) = idx;
                }
            }
        }
    }

};


}

double mini_batch_kmeans(cv::InputArray _data,
               int K,
               int B,
               cv::InputOutputArray _bestLabels,
               cv::TermCriteria criteria,
               int attempts,
               int flags,
               cv::OutputArray _centers,
               const DistanceBody &norm
               )
{
    typedef float(*function_type)(const float*, const float*, int);

    function_type fun = norm ? *norm.target<function_type>() : nullptr;

    static const int SPP_TRIALS = 3;
    cv::Mat data0 = _data.getMat();
    bool isrow = data0.rows == 1;
    int N = isrow ? data0.cols : data0.rows;
    int dims = (isrow ? 1 : data0.cols)*data0.channels();
    int type = data0.depth();

    attempts = std::max(attempts, 1);
    CV_Assert( data0.dims <= 2 && type == CV_32F && K > 0 );
    CV_Assert( N >= K );

    cv::Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

    _bestLabels.create(N, 1, CV_32S, -1, true);

    cv::Mat _labels, best_labels = _bestLabels.getMat();
    if( flags & cv::KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                  best_labels.cols*best_labels.rows == N &&
                  best_labels.type() == CV_32S &&
                  best_labels.isContinuous());
        best_labels.copyTo(_labels);
    }
    else
    {
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
             best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S);
        _labels.create(best_labels.size(), best_labels.type());
    }
    int* labels = _labels.ptr<int>();

    std::memset(_labels.data, -1, _labels.rows*_labels.step);

    cv::Mat centers(K, dims, type), old_centers(K,dims, type), temp(1, dims, type);



//    cv::Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
    std::vector<int> counters(K);
    std::vector<cv::Vec2f> _box(dims);
    cv::Mat dists(1, B, CV_64F, 0.);
    cv::Vec2f* box = &_box[0];
    double best_compactness = std::numeric_limits<double>::max(), compactness = 0;
    cv::RNG& rng = cv::theRNG();
    int a, iter, i, j, k;

    if( criteria.type & cv::TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = std::numeric_limits<float>::epsilon();
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & cv::TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 1), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

    const float* sample = nullptr;

    if(flags & cv::KMEANS_RANDOM_CENTERS)
    {
        sample = data.ptr<float>(0);
        for( j = 0; j < dims; j++ )
            box[j] = cv::Vec2f(sample[j], sample[j]);

        for( i = 1; i < N; i++ )
        {
            sample = data.ptr<float>(i);
            for( j = 0; j < dims; j++ )
            {
                float v = sample[j];
                box[j][0] = std::min(box[j][0], v);
                box[j][1] = std::max(box[j][1], v);
            }
        }
    }


    cv::Mat1i index(1,B);

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = std::numeric_limits<double>::max();

        for( iter = 0;; )
        {
            swap(centers, old_centers);

            if( iter == 0 && (a > 0 || !(flags & cv::KMEANS_USE_INITIAL_LABELS)) )
            {
                if( flags & cv::KMEANS_PP_CENTERS )
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS, fun);
                else
                {
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( (iter == 0) && (a == 0) && (flags & cv::KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }


                // compute centers
//                centers = cv::Scalar(0);
//                for( k = 0; k < K; k++ )
//                    counters[k] = 0;

                std::memset(centers.data, 0, centers.rows*centers.step);
                std::memset(counters.data(), 0, counters.size()*sizeof(int));

                for( i = 0; i < B; i++ )
                {
                    const int idx = index(i);

                    sample = data.ptr<float>(idx);
                    k = labels[idx];
                    float* center = centers.ptr<float>(k);
                    j=0;

                    float alpha = counters[k] ? 1.f/static_cast<float>(counters[k]) : 1.f;
                    float beta = 1.f - alpha;

                    update_center(center, beta, sample, alpha, center, dims);

                    counters[k]++;
                }

//                cv::parallel_for_(cv::Range(0, B),
//                                  MiniBatchKMeansUpdateCentres(_labels, data, index, centers, counters),
//                                  cvCeil(B / static_cast<double>(cv::getNumberOfCPUs()))
//                                  );

                if( iter > 0 )
                    max_center_shift = 0;

                for( k = 0; k < K; k++ )
                {
                    if( counters[k] != 0 )
                        continue;

                    // if some cluster appeared to be empty then:
                    //   1. find the biggest cluster
                    //   2. find the farthest from the center point in the biggest cluster
                    //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                    int max_k = 0;
                    for( int k1 = 1; k1 < K; k1++ )
                    {
                        if( counters[max_k] < counters[k1] )
                            max_k = k1;
                    }

                    double max_dist = 0;
                    int farthest_i = -1;
                    float* new_center = centers.ptr<float>(k);
                    float* old_center = centers.ptr<float>(max_k);
                    float* _old_center = temp.ptr<float>(); // normalized
                    float scale = 1.f/counters[max_k];

                    apply_scale(old_center, scale, _old_center, dims);

                    cv::parallel_for_(cv::Range(0, B), MiniBatchKMeansFindLabel(data, _labels, index, _old_center, max_k, max_dist, farthest_i, fun), cvCeil(B / static_cast<double>(cv::getNumberOfCPUs())));

                    counters[max_k]--;
                    counters[k]++;
                    labels[farthest_i] = k;
                    sample = data.ptr<float>(farthest_i);

                    update_new_old_centres(sample, new_center, old_center, dims);
                }

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    float* it_centre = center;

                    CV_Assert( counters[k] != 0 );

                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        const float* it_old_centre = old_center;

#if CV_AVX
                        float CV_DECL_ALIGNED(0x20) buffer[8];
#else
                        float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

                        it_centre = center;

                        j=0;

#if CV_SIMD128
                        cv::v_float32x4 v_dist = cv::v_setzero_f32();
#endif

#if CV_AVX
                        __m256 v_dist256 = _mm256_setzero_ps();

                        for(;j<=dims-4;j+=4, it_old_centre+=4, it_centre+=4)
                        {
                            __m256 v_c = _mm256_loadu_ps(it_centre);
                            __m256 v_oc = _mm256_loadu_ps(it_old_centre);

                            __m256 v_t = _mm256_sub_ps(v_c, v_oc);

#if CV_FMA3
                            v_dist256 = _mm256_fmadd_ps(v_t, v_t, v_dist256);
#else
                            v_t = _mm256_mul_ps(v_t, v_t);
                            v_dist256 = _mm256_add_ps(v_t, v_dist256);
#endif
                        }

                        _mm256_stream_ps(buffer, v_dist256);

                        dist = std::accumulate(buffer, buffer+8, 0.);
#endif

#if CV_SIMD128
                        for(;j<=dims-4;j+=4, it_old_centre+=4, it_centre+=4)
                        {
                            cv::v_float32x4 v_t = cv::v_load(it_centre) - cv::v_load(it_old_centre);

                            v_dist += v_t*v_t;
                        }
                        cv::v_store_aligned(buffer, v_dist);
#else
                        for(;j<=dims-4;j+=4, it_old_centre+=4, it_centre+=4)
                        {
                            float t0 = it_centre[0] - it_old_centre[0];
                            float t1 = it_centre[1] - it_old_centre[1];

                            t0 *= t0;
                            t1 *= t1;

                            dist += t0;
                            dist += t1;

                            t0 = it_centre[2] - it_old_centre[2];
                            t1 = it_centre[3] - it_old_centre[3];

                            dist += t0;
                            dist += t1;
                        }
#endif

                        dist += std::accumulate(buffer, buffer+4, 0.);

                        for( ; j < dims; j++, it_centre++, it_old_centre++ )
                        {
                            double t = *it_centre - *it_old_centre;
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }

            bool isLastIter = (++iter == std::max(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);

            for(int i=0;i<B;i++)
                index(i) = cv::theRNG().uniform(0,N);

            // assign labels
            dists = 0;
            double* dist = dists.ptr<double>(0);
            cv::parallel_for_(cv::Range(0, B), MiniBatchKMeansDistanceComputer(dist, labels, data, centers, index, fun, isLastIter),
                              cvCeil(B / static_cast<double>(cv::getNumberOfCPUs()))
                              );

//            (KMeansDistanceComputer(dist, labels, data, centers, index, fun, isLastIter))(cv::Range(0, B));

            compactness = sum(dists)[0];

            if (isLastIter)
            {
                cv::parallel_for_(cv::Range(0, N), MiniBatchKMeans_Assignment(data, centers, _labels, fun), cvCeil(N / static_cast<double>(cv::getNumberOfCPUs())));

                break;
            }
        }

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            if( _centers.needed() )
            {
                cv::Mat reshaped = centers;
                if(_centers.fixedType() && _centers.channels() == dims)
                    reshaped = centers.reshape(dims);
                reshaped.copyTo(_centers);
            }
            _labels.copyTo(best_labels);
        }
    }



    return best_compactness;
}


} //clustering

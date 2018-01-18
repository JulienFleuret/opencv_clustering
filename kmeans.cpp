//Copyright 2018 University Laval, CVSL-MIVIM

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "clustering.h"

#include <numeric>

#include <opencv2/core/cv_cpu_helper.h>
#include <opencv2/core/hal/intrin.hpp>

#include <immintrin.h>

namespace clustering
{

namespace
{

void generateRandomCenter(const std::vector<cv::Vec2f>& box, float* center, cv::RNG& rng)
{
    size_t j;
    const size_t dims = box.size();
    const float margin = 1.f/dims;
    const float cst = 1.f+margin*2.f;

    for( j = 0; j < dims; j++ )
    {
        const cv::Vec2f& vb = box[j];

        center[j] = ((float)rng*cst-margin)*(vb(1) - vb(0)) + vb(0);
    }
}

class KMeansPPDistanceComputer : public cv::ParallelLoopBody
{
public:

    typedef float(*function_type)(const float*, const float*, int);

    inline KMeansPPDistanceComputer( float *_tdist2,
                              const float *_data,
                              const float *_dist,
                              int _dims,
                              size_t _step,
                              size_t _stepci,
                              const function_type _fun)
        : tdist2(_tdist2),
          data(_data),
          dist(_dist),
          dims(_dims),
          step(_step),
          stepci(_stepci),
          fun(_fun)
    {}

    virtual ~KMeansPPDistanceComputer() = default;

    virtual void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;

        const float* tmp_left = data + step*begin;
        const float* tmp_right = data + stepci;

        for ( int i = begin; i<end; i++, tmp_left+=step )
        {
//            float tmp = this->fun(data + step*i, data + stepci, dims);
            float tmp = this->fun(tmp_left, tmp_right, dims);

//            tdist2[i] = std::min(cv::normL2Sqr(data + step*i, data + stepci, dims), dist[i]);

            tdist2[i] = std::min(tmp * tmp, dist[i]);
        }
    }

private:
    KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&); // to quiet MSVC

    float *tdist2;
    const float *data;
    const float *dist;
    const int dims;
    const size_t step;
    const size_t stepci;
    const function_type fun;
};

/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
void generateCentersPP(const cv::Mat& _data, cv::Mat& _out_centers,
                              int K, cv::RNG& rng, int trials, float(*fun)(const float*, const float*, int))
{

    int i, j, k, dims = _data.cols, N = _data.rows, N_algn = cv::alignSize(N,4);
    const float* data = _data.ptr<float>(0);
    size_t step = _data.step/sizeof(data[0]);
    std::vector<int> _centers(K);
    int* centers = &_centers[0];
    std::vector<float> _dist(N_algn*3);
    float* dist = &_dist[0], *tdist = dist + N_algn, *tdist2 = tdist + N_algn;
    double sum0 = 0;

    centers[0] = (unsigned)rng % N;

    for( i = 0; i < N; i++ )
    {
//        dist[i] = cv::normL2Sqr(data + step*i, data + step*centers[0], dims);
        float tmp = fun(data + step*i, data + step*centers[0], dims);
        dist[i] = tmp*tmp;
        sum0 += dist[i];
    }

    cv::Vec4f buffer;
    for( k = 1; k < K; k++ )
    {
        double bestSum = std::numeric_limits<double>::max();
        int bestCenter = -1;

        for( j = 0; j < trials; j++ )
        {
            double p = (double)rng*sum0, s = 0.;

            for( i = 0; i < N-1; i++ )
                if( (p -= dist[i]) <= 0 )
                    break;
            int ci = i;

            cv::parallel_for_(cv::Range(0, N),
                         KMeansPPDistanceComputer(tdist2, data, dist, dims, step, step*ci, fun),
                              cvCeil(N / static_cast<double>(cv::getNumberOfCPUs())));
            i=0;
            float* it_tdist2 = tdist2;

            cv::v_float32x4 v_s = cv::v_setzero_f32();
            for(;i<=N-4;i+=4, it_tdist2+=4)
            {
                cv::v_float32x4 v_td = cv::v_load(it_tdist2);
                v_s+= v_td;
            }
            cv::v_store(buffer.val, v_s);

            s = std::accumulate(buffer.val, buffer.val+4, 0.);

            for( ; i < N; i++, it_tdist2++ )
//                s += tdist2[i];
                s+=*it_tdist2;


            if( s < bestSum )
            {
                bestSum = s;
                bestCenter = ci;
                std::swap(tdist, tdist2);
            }
        }

        centers[k] = bestCenter;
        sum0 = bestSum;

        std::swap(dist, tdist);
    }

    for( k = 0; k < K; k++ )
    {
        const float* src = data + step*centers[k];
        float* dst = _out_centers.ptr<float>(k);
//        for( j = 0; j < dims; j++ )
//            dst[j] = src[j];
        std::memcpy(dst, src, dims*sizeof(float));
    }
}

class KMeansDistanceComputer : public cv::ParallelLoopBody
{
public:

    typedef float(*function_type)(const float*, const float*, int);

    inline KMeansDistanceComputer( double *_distances,
                            int *_labels,
                            const cv::Mat& _data,
                            const cv::Mat& _centers,
                            const function_type _fun,
                            bool _onlyDistance = false )
        : distances(_distances),
          labels(_labels),
          data(_data),
          centers(_centers),
          onlyDistance(_onlyDistance),
          fun(_fun)
    {
    }

    virtual ~KMeansDistanceComputer() = default;

    virtual void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;
        const int K = centers.rows;
        const int dims = centers.cols;

        for( int i = begin; i<end; ++i)
        {
            const float *sample = data.ptr<float>(i);
            if (onlyDistance)
            {
                const float* center = centers.ptr<float>(labels[i]);
//                distances[i] = cv::normL2Sqr(sample, center, dims);
                const float tmp = this->fun(sample, center, dims);
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
            labels[i] = k_best;
        }
    }

private:
    KMeansDistanceComputer& operator=(const KMeansDistanceComputer&); // to quiet MSVC

    double *distances;
    int *labels;
    const cv::Mat& data;
    const cv::Mat& centers;
    bool onlyDistance;
    const function_type fun;
};

}

double kmeans( cv::InputArray _data,
               int K,
               cv::InputOutputArray _bestLabels,
               cv::TermCriteria criteria,
               int attempts,
               int flags,
               cv::InputOutputArray _centers,
               const std::function<float(const float*, const float*, int)>& _fun
               )
{
    if(!_fun)
        return cv::kmeans(_data, K, _bestLabels, criteria, attempts, flags, _centers);

    typedef float(*function_type)(const float*, const float*, int);

    function_type fun = _fun ? *_fun.target<function_type>() : (function_type)cv::normL2Sqr;

    const int SPP_TRIALS = 3;
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

    cv::Mat centers, old_centers(K,dims, type), temp(1, dims, type);

    bool preprocess_labels = false;



    if(_centers.empty())
        centers.create(K,dims,type);
    else
    {
        CV_Assert(_centers.rows() == K && _centers.cols() == dims);

        centers = _centers.getMat();

        if(cv::countNonZero(centers))
        {
           if(flags&cv::KMEANS_PP_CENTERS)
               flags &= ~cv::KMEANS_PP_CENTERS;

           if(flags&cv::KMEANS_RANDOM_CENTERS)
               flags &= ~ cv::KMEANS_RANDOM_CENTERS;

           flags |= cv::KMEANS_USE_INITIAL_LABELS;

           preprocess_labels = true;
        }
    }


//    cv::Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
    std::vector<int> counters(K);
    std::vector<cv::Vec2f> _box(dims);
    cv::Mat dists(1, N, CV_64F);
    cv::Vec2f* box = &_box[0];
    double best_compactness = DBL_MAX, compactness = 0;
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


    if(preprocess_labels)
        cv::parallel_for_(cv::Range(0, N), KMeansDistanceComputer(dists.ptr<double>(), labels, data, centers, fun), cvCeil(N / static_cast<double>(cv::getNumberOfCPUs())) );


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

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<float>(i);
                    k = labels[i];
                    float* center = centers.ptr<float>(k);
                    j=0;

                    float* it_center = center;
                    const float* it_sample = sample;

#if CV_AVX
                    for(;j<=dims-8;j+=8, it_center+=8, it_sample+=8)
                    {
                        __m256 v_centre = _mm256_loadu_ps(it_center);
                        __m256 v_sample = _mm256_loadu_ps(it_sample);

                        v_centre = _mm256_add_ps(v_centre, v_sample);

                        _mm256_storeu_ps(it_center, v_centre);
                    }
#endif

                    for(;j<=dims-4;j+=4, it_center+=4, it_sample+=4)
                    {
                        cv::v_float32x4 v_centre = cv::v_load(it_center);
                        cv::v_float32x4 v_sample = cv::v_load(it_sample);

                        v_centre+=v_sample;

                        cv::v_store(it_center, v_centre);
                    }

                    for( ; j < dims; j++, it_center++, it_sample++ )
//                        center[j] += sample[j];
                        *it_center += *it_sample;
                    counters[k]++;
                }

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

                    float* it_oc = old_center;
                    float* it_tmp = _old_center;

                    j=0;

#if CV_AVX
                    __m256 v_scale256 = _mm256_set1_ps(scale);

                    for(;j<=dims-8; j+=8, it_oc+=8, it_tmp+=8)
                    {
                        __m256 v_oc = _mm256_loadu_ps(it_oc);

                        v_oc = _mm256_mul_ps(v_oc, v_scale256);

                        _mm256_storeu_ps(it_tmp, v_oc);
                    }
#endif

                    cv::v_float32x4 v_scale = cv::v_setall_f32(scale);

                    for(;j<=dims-4;j+=4, it_oc+=4, it_tmp+=4)
                        cv::v_store(it_tmp, cv::v_load(it_oc)*v_scale);


                    for( ; j < dims; j++, it_oc++, it_tmp++ )
                        *it_tmp = *it_oc*scale;

                    for( i = 0; i < N; i++ )
                    {
                        if( labels[i] != max_k )
                            continue;
                        sample = data.ptr<float>(i);
//                        double dist = cv::normL2Sqr(sample, _old_center, dims);
                        double dist = fun(sample, _old_center, dims);
                        dist*=dist;

                        if( max_dist <= dist )
                        {
                            max_dist = dist;
                            farthest_i = i;
                        }
                    }

                    counters[max_k]--;
                    counters[k]++;
                    labels[farthest_i] = k;
                    sample = data.ptr<float>(farthest_i);

                    j=0;

                    it_oc = old_center;
                    float* it_nc = new_center;
                    const float* it_sample = sample;

#if CV_AVX
                    for(;j<=dims-8;j+=8,
                        it_oc+=8,
                        it_nc+=8,
                        it_sample+=8)
                    {
                        __m256 v_oc = _mm256_loadu_ps(it_oc);
                        __m256 v_nc = _mm256_loadu_ps(it_nc);
                        __m256 v_s = _mm256_loadu_ps(it_sample);

                        v_oc = _mm256_sub_ps(v_oc, v_s);
                        v_nc = _mm256_add_ps(v_nc, v_s);

                        _mm256_storeu_ps(it_oc, v_oc);
                        _mm256_storeu_ps(it_nc, v_nc);
                    }
#endif

                    for(;j<=dims-4;j+=4,
                        it_oc+=4,
                        it_nc+=4,
                        it_sample+=4)
                    {
                        cv::v_float32x4 v_oc = cv::v_load(it_oc);
                        cv::v_float32x4 v_nc = cv::v_load(it_nc);
                        cv::v_float32x4 v_s = cv::v_load(it_sample);

                        v_oc -= v_s;
                        v_nc += v_s;

                        cv::v_store(it_oc, v_oc);
                        cv::v_store(it_nc, v_nc);
                    }

                    for( ; j < dims; j++, it_oc++, it_nc++, it_sample++ )
                    {
                        float v_s = *it_sample;

                        *it_oc -= v_s;
                        *it_nc += v_s;
                    }
                }

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    float* it_centre = center;

                    CV_Assert( counters[k] != 0 );

                    float scale = 1.f/counters[k];

                    j=0;

                    cv::v_float32x4 v_scale = cv::v_setall_f32(scale);

#if CV_AVX
                    __m256 v_scale256 = _mm256_set1_ps(scale);

                    for(;j<=dims-8; j+=8, it_centre+=8)
                    {
                        __m256 v_centre = _mm256_loadu_ps(it_centre);

                        v_centre = _mm256_mul_ps(v_centre, v_scale256);

                        _mm256_storeu_ps(it_centre, v_centre);
                    }
#endif

                    for(;j<=dims-4;j+=4, it_centre+=4)
                    {
                        cv::v_float32x4 v_centre = cv::v_load(it_centre);
                        v_centre *= v_scale;
                        cv::v_store(it_centre, v_centre);
                    }

                    for( ; j < dims; j++, it_centre++ )
                        *it_centre *= scale;
//                        center[j] *= scale;

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

                        cv::v_float32x4 v_dist = cv::v_setzero_f32();

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

                        for(;j<=dims-4;j+=4, it_old_centre+=4, it_centre+=4)
                        {
                            cv::v_float32x4 v_t = cv::v_load(it_centre) - cv::v_load(it_old_centre);

                            v_dist += v_t*v_t;
                        }
                        cv::v_store_aligned(buffer, v_dist);

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

            bool isLastIter = (++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);

            // assign labels
            dists = 0;
            double* dist = dists.ptr<double>(0);
            cv::parallel_for_(cv::Range(0, N), KMeansDistanceComputer(dist, labels, data, centers, fun, isLastIter),
                              cvCeil(N / static_cast<double>(cv::getNumberOfCPUs()))
                              );
            compactness = sum(dists)[0];

            if (isLastIter)
                break;
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

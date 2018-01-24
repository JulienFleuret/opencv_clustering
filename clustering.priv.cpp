#include "clustering.priv.hpp"

#include <opencv2/core/cv_cpu_helper.h>
#include <opencv2/core/hal/intrin.hpp>

#include <numeric>

namespace clustering
{

void generateRandomCenter(const std::vector<cv::Vec2f>& box, float* center, cv::RNG& rng)
{
    size_t j;
    const size_t dims = box.size();
    const float margin = 1.f/dims;
    const float cst = 1.f+margin*2.f;

    for( j = 0; j < dims; j++ )
    {
        const cv::Vec2f vb = box[j];

        center[j] = ((float)rng*cst-margin)*(vb(1) - vb(0)) + vb(0);
    }
}


void KMeansPPDistanceComputer::operator ()(const cv::Range& range)const
{
    const int begin = range.start;
    const int end = range.end;

    const float* tmp_left = data + step*begin;
    const float* tmp_right = data + stepci;

    if(this->fun)
    {
        for ( int i = begin; i<end; i++, tmp_left+=step )
        {
            //            float tmp = this->fun(data + step*i, data + stepci, dims);
            float tmp = this->fun(tmp_left, tmp_right, dims);

            //            tdist2[i] = std::min(cv::normL2Sqr(data + step*i, data + stepci, dims), dist[i]);

            tdist2[i] = std::min(tmp * tmp, dist[i]);
        }
    }
    else
    {
        for ( int i = begin; i<end; i++, tmp_left+=step )
            tdist2[i] = std::min(cv::normL2Sqr(tmp_left, tmp_right, dims), dist[i]);
    }
}


/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/

class KMeansPPInit : public cv::ParallelLoopBody
{

public:

    typedef float(*function_type)(const float*, const float*, int);

private:
    const float* data;
    const size_t step;
    const float* right;
    const int dims;
    const function_type fun;

    float* dist;
    double& sum0;

    mutable cv::Mutex mtx;

public:

    inline KMeansPPInit(const float* _data, const size_t& _step, const float* _right, const int& _dims, const function_type _fun, float* _dist, double& _sum0):
        data(_data),
        step(_step),
        right(_right),
        dims(_dims),
        fun(_fun),
        dist(_dist),
        sum0(_sum0)
    {}

    virtual ~KMeansPPInit() = default;

    virtual void operator ()(const cv::Range& range)const
    {
        const float* it_data = this->data + range.start * this->step;

        float* it_dist = this->dist + range.start;

        double lsum0 = 0.;

        if(fun)
        {
            for(int r=range.start; r<range.end; r++, it_data += this->step, it_dist++)
            {
                float tmp = this->fun(it_data, this->right, this->dims);
                *it_dist = tmp*tmp;

                lsum0 += *it_dist;
            }
        }
        else
        {
            for(int r = range.start; r < range.end; r++, it_data+=step, it_dist++ )
            {
                *it_dist = cv::normL2Sqr(it_data, this->right, dims);

                lsum0 += *it_dist;
            }
        }

        cv::AutoLock lck(this->mtx);

        this->sum0+=lsum0;

    }

};

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

//    const float* it_data = data;
//    const size_t cst = step*centers[0];

//    if(fun)
//    {
//        for( i = 0; i < N; i++, it_data+=step )
//        {
//            float tmp = fun(it_data, data + cst, dims);
//            dist[i] = tmp*tmp;

//            sum0 += dist[i];
//        }
//    }
//    else
//    {
//        for( i = 0; i < N; i++, it_data+=step )
//        {
//            dist[i] = cv::normL2Sqr(it_data, data + cst, dims);

//            sum0 += dist[i];
//        }
//    }

    cv::parallel_for_(cv::Range(0, N), KMeansPPInit(data, step, data + step*centers[0], dims, fun, dist, sum0), cvCeil(N / static_cast<double>(cv::getNumberOfCPUs())));


#if CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif
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

#if CV_SIMD128
            cv::v_float32x4 v_s = cv::v_setzero_f32();
            for(;i<=N-4;i+=4, it_tdist2+=4)
            {
                cv::v_float32x4 v_td = cv::v_load(it_tdist2);
                v_s+= v_td;
            }
            cv::v_store_aligned(buffer, v_s);

            s = std::accumulate(buffer, buffer+4, 0.);
#else
            for(;i<=N-4;i+=4, it_tdist2+=4)
            {
                s+=it_tdist2[0];
                s+=it_tdist2[1];
                s+=it_tdist2[2];
                s+=it_tdist2[3];
            }
#endif

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


void apply_scale(const float* src, const float& scale, float* dst, const int& dims)
{

    int i=0;

#if CV_AVX

   const __m256 v_scale256 = _mm256_set1_ps(scale);

#if CV_SIMD128
   for(;i<=dims-8;i+=8, src+=8, dst+=8)
   {
        __m256 v_s = _mm256_loadu_ps(src);

        v_s = _mm256_mul_ps(v_s, v_scale256);

        _mm256_storeu_ps(dst, v_s);
   }
#else
   for(;i<=dims-8;i+=8)
   {
        __m256 v_s = _mm256_loadu_ps(src+i);

        v_s = _mm256_mul_ps(v_s, v_scale256);

        _mm256_storeu_ps(dst+i, v_s);
   }
#endif

#endif

#if CV_SIMD128
   const cv::v_float32x4 v_scale = cv::v_setall_f32(scale);

   for(;i<=dims-4;i+=4, src+=4, dst+=4)
       cv::v_store(dst, cv::v_load(src) * v_scale);

   for(;i<dims;i++, src++, dst++)
       *dst = *src * scale;
#else
   for(;i<=dims-4;i+=4)
   {
       dst[i] = src[i] * scale;
       dst[i+1] = src[i+1] * scale;
       dst[i+2] = src[i+2] * scale;
       dst[i+3] = src[i+3] * scale;
   }

   for(;i<dims;i++)
       dst[0] = src[0] * scale;
#endif

}

void update_new_old_centres(const float* samples, float* nc, float* oc, const int& dims)
{
    int i=0;

#if CV_AVX
#if CV_SIMD128
   for(;i<=dims-8;i+=8, samples+=8, nc+=8, oc+=8)
   {
       __m256 v_s = _mm256_loadu_ps(sample);
       __m256 v_nc = _mm256_loadu_ps(nc);
       __m256 v_oc = _mm256_loadu_ps(oc);

       v_oc = _mm256_sub_ps(v_oc, v_s);
       v_oc = _mm256_add_ps(v_oc, v_s);

       _mm256_storeu_ps(oc,v_oc);
       _mm256_storeu_ps(nc,v_oc);
   }
#else
    for(;i<=dims-8;i+=8)
    {
        __m256 v_s = _mm256_loadu_ps(sample+i);
        __m256 v_nc = _mm256_loadu_ps(nc+i);
        __m256 v_oc = _mm256_loadu_ps(oc+i);

        v_oc = _mm256_sub_ps(v_oc, v_s);
        v_oc = _mm256_add_ps(v_oc, v_s);

        _mm256_storeu_ps(oc+i,v_oc);
        _mm256_storeu_ps(nc+i,v_oc);
    }
#endif

#endif

#if CV_SIMD128
    for(;i<=dims-4;samples+=4, nc+=4,oc+=4)
    {
        cv::v_float32x4 v_s = cv::v_load(samples);
        cv::v_float32x4 v_oc = cv::v_load(oc);
        cv::v_float32x4 v_nc = cv::v_load(nc);

        v_oc-=v_s;
        v_nc+=v_s;

        cv::v_store(oc, v_oc);
        cv::v_store(nc, v_nc);
    }

    for(;i<dims;i++,samples++,oc++,nc++)
    {
        float s = *samples;

        *oc -= s;
        *nc += s;
    }

#else
    for(;i<=dims-4;samples+=4)
    {
        float s0 = samples[i];
        float s1 = samples[i+1];

        oc[i] -= s0;
        oc[i+1] -= s1;

        nc[i] += s0;
        nc[i+1] += s1;

        s0 = samples[i+2];
        s1 = samples[i+3];

        oc[i+2] -= s0;
        oc[i+3] -= s1;

        nc[i+2] += s0;
        nc[i+3] += s1;

    }

    for(;i<dims;i++)
    {
        float s = samples[i];

        oc[i] -= s;
        nc[i] += s;
    }

#endif

}


}

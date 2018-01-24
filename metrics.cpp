#include "metrics.h"

#include <opencv2/core/cv_cpu_helper.h>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>

namespace clustering
{

namespace hal
{

float normL2(const float*  a, const float*  b, int dims)
{
    return std::sqrt(cv::hal::normL2Sqr_(a,b,dims));
}

float normLp(const float*  a,const float*  b,int dims,const float& p)
{

    if(p==1.f)
        return cv::hal::normL1_(a,b,dims);

    if(p==2.f)
        return normL2(a,b,dims);

    float n(0.f);

    int l(0); // length.


#if CV_ENABLE_UNROLLED
    for(;l<dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        a0 = std::pow(std::abs(a0 - b0),p);
        a1 = std::pow(std::abs(a1 - b1),p);

        n += a0 + a1;

        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        a0 = std::pow(std::abs(a0 - b0),p);
        a1 = std::pow(std::abs(a1 - b1),p);

        n += a0 + a1;
    }
#endif

    for(;l<dims;l++, a++, b++)
        n += std::pow(std::abs(a[0] - b[0]),p);

#if CV_SSE2
   _mm_mfence();
#endif

    return std::sqrt(n);
}



float normChebyshev(const float*  a,const float*  b,int dims)
{
    float n(0.f);

    int l(0);


#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[8];
#endif

#if CV_AVX

   __m256 v_n = _mm256_setzero_ps();

   static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));


   for(;l<=dims-8; l+=8, a+=8, b+=8)
   {
       __m256 v_p = _mm256_loadu_ps(a);
       __m256 v_q = _mm256_loadu_ps(b);

       __m256 tmp = _mm256_sub_ps(v_p, v_q);
       tmp = _mm256_and_ps(absmask, tmp);

       v_n = _mm256_max_ps(v_n, tmp);
   }

   _mm256_stream_ps(buffer, tmp);

   n = std::max(n,*std::max_element(buffer, buffer + 8));

#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n = cv::v_max(v_n, cv::v_abs(v_p-v_q));
    }

    cv::v_store_aligned(buffer, v_n);

    n = std::max(n,*std::max_element(buffer, buffer+4));
#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        a0 = std::abs(a0 - b0);
        a1 = std::abs(a1 - b1);

        a0 = std::max(a0,a1);

        n = std::max(n,a0);

        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        a0 = std::abs(a0 - b0);
        a1 = std::abs(a1 - b1);

        a0 = std::max(a0,a1);

        n = std::max(n,a0);
    }

#endif


    for(; l<dims;l++, a++, b++)
        n = std::max(n,std::abs(*a - *b));

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}


float normSorensen(const float*  a,const float*  b,int dims)
{
    float n(0.f);
    float d(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[8];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();
    __m256 _d = _n;

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for(;l<=dims-8;l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_and_ps(tmp_n,absmask);

        __m256 tmp_d = _mm256_add_ps(_p,_q);

        _n = _mm256_add_ps(_n, tmp_n);
        _d = _mm256_add_ps(_d, tmp_d);
    }

    _mm256_stream_ps(buffer,_n);
    _mm256_stream_ps(buffer+8,_d);

    for(int i=0, j=8;i<8;i++,j++)
    {
        n += buffer[i];
        d += buffer[j];
    }

#endif

    //#elif CV_SSE2

    //    __m128 _n = _mm_setzero_ps();
    //    __m128 _d = _n;

    //    static const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));

    //    for(;l<dims-4;l+=4, a+=4, b+=4)
    //    {
    //        __m128 _p = _mm_loadu_ps(a);
    //        __m128 _q = _mm_loadu_ps(b);

    //        __m128 tmp_n = _mm_sub_ps(_p,_q);
    //        tmp_n = _mm_and_ps(tmp_n,absmask);

    //        __m128 tmp_d = _mm_add_ps(_p,_q);

    //        _n = _mm_add_ps(_n, tmp_n);
    //        _d = _mm_add_ps(_d, tmp_d);
    //    }

    //    _mm_stream_ps(buffer,_n);
    //    _mm_stream_ps(buffer+4,_d);

    //    for(int i=0;i<4;++i)
    //    {
    //        n += buffer[i];
    //        d += buffer[i+4];
    //    }
    //#endif

#if CV_SIMD128


//    static const cv::v_float32x4 absmask = cv::v_cvt_f32(cv::v_setall_s32(0x7fffffff));

    cv::v_float32x4 v_n = cv::v_setzero_f32();
    cv::v_float32x4 v_d = v_n;

    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n += cv::v_abs(v_p-v_q);
        v_d += v_p+v_q;
    }

    cv::v_store_aligned(buffer, v_n);
    cv::v_store_aligned(buffer+4, v_d);

    for(int i=0, j=4;i<4;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }

#elif CV_ENABLE_UNROLLED
    for(;l<dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        n += std::abs(a0 - b0);
        n += std::abs(a1 - b1);

        d += a0 + b0;
        d += a1 + b1;

        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        n += std::abs(a0 - b0);
        n += std::abs(a1 - b1);

        d += a0 + b0;
        d += a1 + b1;

    }


#endif


    for(; l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        n += std::abs(p - q);
        d += p + q;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/d);

}



float normCzekanowski(const float*  a,const float*  b,int dims)
{
    return normSorensen(a,b,dims);
}

float normGower(const float*  a,const float*  b,int dims)
{
    return cv::hal::normL1_(a,b,dims)/(float)dims;
}

float normSoergel(const float*  a,const float*  b,int dims)
{
    float n(0.f);
    float d(0.f);

    int l(0);


#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[8];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();
    __m256 _d = _n;

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for(;l<=dims-8;l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_and_ps(tmp_n,absmask);

        __m256 tmp_d = _mm256_max_ps(_p,_q);

        _n = _mm256_add_ps(_n, tmp_n);
        _d = _mm256_add_ps(_d, tmp_d);
    }

    _mm256_stream_ps(buffer,_n);
    _mm256_stream_ps(buffer+8,_d);

    for(int i=0,j=8;i<8;i++, j++)
    {
        n += buffer[i];
        d += buffer[j];
    }
#endif

//#elif CV_SSE2

//    __m128 _n = _mm_setzero_ps();
//    __m128 _d = _n;

//    static const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));

//    for(;l<dims-4;l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_and_ps(tmp_n,absmask);

//        __m128 tmp_d = _mm_max_ps(_p,_q);

//        _n = _mm_add_ps(_n, tmp_n);
//        _d = _mm_add_ps(_d, tmp_d);
//    }

//    _mm_stream_ps(buffer,_n);
//    _mm_stream_ps(buffer+4,_d);

//    for(int i=0;i<4;++i)
//    {
//        n += buffer[i];
//        d += buffer[i+4];
//    }
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();
    cv::v_float32x4 v_d = v_n;

    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n += cv::v_abs(v_p - v_q);
        v_d += cv::v_max(v_p, v_q);
    }

    cv::v_store_aligned(buffer, v_n);
    cv::v_store_aligned(buffer+4, v_d);

    for(int i=0, j=4;i<4;i++, j++)
    {
        n += buffer[i];
        d += buffer[j];
    }

#elif CV_ENABLE_UNROLLED
    for(;l<dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        n += std::abs(a0 - b0);
        n += std::abs(a1 - b1);

        d += std::max(a0, b0);
        d += std::max(a1, b1);

        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        n += std::abs(a0 - b0);
        n += std::abs(a1 - b1);

        d += std::max(a0, b0);
        d += std::max(a1, b1);
    }

#endif

    for(; l<dims;l++, a++, b++)
    {

        float p = *a;
        float q = *b;

        n += std::abs(p - q);
        d += std::max(p, q);
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/d);

}

float normKulczynski(const float*  a,const float*  b,int dims)
{
    float n(0.f);
    float d(0.f);

    int l(0);


#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[8];
#endif


#if CV_AVX


    __m256 _n = _mm256_setzero_ps();
    __m256 _d = _n;

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for(;l<=dims-8;l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_and_ps(tmp_n,absmask);

        __m256 tmp_d = _mm256_min_ps(_p,_q);

        _n = _mm256_add_ps(_n, tmp_n);
        _d = _mm256_add_ps(_d, tmp_d);
    }

    _mm256_stream_ps(buffer,_n);
    _mm256_stream_ps(buffer+8,_d);

    for(int i=0, j=8;i<8; i++, j++)
    {
        n += buffer[i];
        d += buffer[j];
    }

#endif

//#elif CV_SSE2

//    __m128 _n = _mm_setzero_ps();
//    __m128 _d = _n;

//    static const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));

//    for(;l<dims-4;l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_and_ps(tmp_n,absmask);

//        __m128 tmp_d = _mm_min_ps(_p,_q);

//        _n = _mm_add_ps(_n, tmp_n);
//        _d = _mm_add_ps(_d, tmp_d);
//    }

//    _mm_stream_ps(buffer,_n);
//    _mm_stream_ps(buffer+4,_d);

//    for(int i=0;i<4;++i)
//    {
//        n += buffer[i];
//        d += buffer[i+4];
//    }
//#endif

#if CV_SIMD128
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_n = cv::v_setzero_f32();
        cv::v_float32x4 v_d = v_n;

        for(;l<=dims-4;l+=4, a+=4, b+=4)
        {

            cv::v_float32x4 v_p = cv::v_load(a);
            cv::v_float32x4 v_q = cv::v_load(b);

            v_n += cv::v_abs(v_p - v_q);
            v_d += cv::v_min(v_p, v_q);
        }

        cv::v_store_aligned(buffer, v_n);
        cv::v_store_aligned(buffer+4, v_d);

        for(int i=0, j=4;i<4;i++, j++)
        {
            n += buffer[i];
            d += buffer[j];
        }
    }
#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        n += std::abs(a0 - b0);
        n += std::abs(a1 - b1);

        d += std::min(a0, b0);
        d += std::min(a1, b1);

        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        n += std::abs(a0 - b0);
        n += std::abs(a1 - b1);

        d += std::min(a0, b0);
        d += std::min(a1, b1);
    }

#endif

    for(; l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        n += std::abs(p - q);
        d += std::min(p, q);
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/d);

}

float normCanberra(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[4];
#endif


#if CV_AVX

    __m256 _n = _mm256_setzero_ps();

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for(; l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_and_ps(tmp_n,absmask);

        __m256 tmp_d = _mm256_add_ps(_p,_q);

        tmp_n = _mm256_div_ps(tmp_n,tmp_d);

        _n = _mm256_add_ps(_n,tmp_n);
    }

    _mm256_stream_ps(buffer,_n);

    for(int i=0;i<8;++i)
        n += buffer[i];
#endif

//#elif CV_SSE2


//    __m256 _n = _mm_setzero_ps();

//    static const __m128 absmask = _mm_castsi256_ps(_mm_set1_epi32(0x7fffffff));

//    for(; l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_and_ps(tmp_n,absmask);

//        __m128 tmp_d = _mm_add_ps(_p,_q);

//        tmp_n = _mm_div_ps(tmp_n,tmp_d);

//        _mm_add_ps(n,tmp_n);
//    }

//    _mm_stream_ps(buffer,_n);

//    for(int i=0;i<4;++i)
//        n += buffer[i];
//#endif

#if CV_SIMD128


    cv::v_float32x4 v_n = cv::v_setzero_f32();


    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n += (cv::v_abs(v_p-v_q)/(v_p+v_q));
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;i++)
        n+=buffer[i];

#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = std::abs(a0 - b0);
        float tmp_n1 = std::abs(a1 - b1);

        float tmp_d0 = a0 + b0;
        float tmp_d1 = a1 + b1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;


        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = std::abs(a0 - b0);
        tmp_n1 = std::abs(a1 - b1);

        tmp_d0 = a0 + b0;
        tmp_d1 = a1 + b1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;
    }


#endif

    for(; l<dims;l++, a++, b++)
    {
        float a0 = *a;
        float b0 = *b;

        float tmp_n0 = std::abs(a0 - b0);

        float tmp_d0 = a0 + b0;

        n += tmp_n0 / tmp_d0;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;


}

float normLorentzian(const float*  a,const float*  b,int dims)
{
    float n(0.f);

    int l=0;

#if CV_ENABLE_UNROLLED
        for(;l<=dims-4; l+=4, a+=4, b+=4)
        {

            float a0 = a[0];
            float a1 = a[1];

            float b0 = b[0];
            float b1 = b[1];

            a0 = std::abs(a0 - b0);
            a1 = std::abs(a1 - b1);

            a0+=1;
            a1+=1;

            n+= std::log(a0);
            n+= std::log(a1);



            a0 = a[2];
            a1 = a[3];

            b0 = b[2];
            b1 = b[3];

            a0 = std::abs(a0 - b0);
            a1 = std::abs(a1 - b1);

            a0+=1;
            a1+=1;

            n+= std::log(a0);
            n+= std::log(a1);
        }
#endif

        for(;l<dims; l++, a++, b++)
            n += std::log(1 + std::abs(a[0] - b[0]));

#if CV_SSE2
   _mm_mfence();
#endif

        return n;

}

float normIntersection(const float*  a,const float*  b,int dims)
{
    return 0.5f*cv::hal::normL1_(a,b,dims);
}

float simIntersection(const float*  a,const float*  b,int dims)
{
    return 1.f - normIntersection(a,b,dims);
}

float normWaveHedges(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
   float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[4];
#endif


#if CV_AVX

    __m256 _n = _mm256_setzero_ps();

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for(; l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_and_ps(tmp_n,absmask);

        __m256 tmp_d = _mm256_max_ps(_p,_q);

        tmp_n = _mm256_div_ps(tmp_n,tmp_d);

        _n = _mm256_add_ps(_n,tmp_n);
    }

    _mm256_stream_ps(buffer,_n);

    for(int i=0;i<8;++i)
        n += buffer[i];
#endif

//#elif CV_SSE2


//    __m256 _n = _mm_setzero_ps();

//    static const __m128 absmask = _mm_castsi256_ps(_mm_set1_epi32(0x7fffffff));

//    for(; l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_and_ps(tmp_n,absmask);

//        __m128 tmp_d = _mm_max_ps(_p,_q);

//        tmp_n = _mm_div_ps(tmp_n,tmp_d);

//        _mm_add_ps(n,tmp_n);
//    }

//    _mm_stream_ps(buffer,_n);

//    for(int i=0;i<4;++i)
//        n += buffer[i];
//#endif

#if CV_SIMD128


    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n += cv::v_abs(v_p - v_q) / cv::v_max(v_p, v_q);
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;i++)
        n+=buffer[i];

#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = std::abs(a0 - b0);
        float tmp_n1 = std::abs(a1 - b1);

        float tmp_d0 = std::max(a0, b0);
        float tmp_d1 = std::max(a1, b1);

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;


        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = std::abs(a0 - b0);
        tmp_n1 = std::abs(a1 - b1);

        tmp_d0 = std::max(a0, b0);
        tmp_d1 = std::max(a1, b1);

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;
    }

#endif

    for(; l<dims;l++, a++, b++)
    {
        float p = a[0];
        float q = b[0];

        float tmp_n0 = std::abs(p - q);

        float tmp_d0 = std::max(p, q);

        n += tmp_n0 / tmp_d0;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;


}

float simCzekanowski(const float*  a,const float*  b,int dims)
{
    return 1.f - normCzekanowski(a,b,dims);
}

float simMotyka(const float*  a,const float*  b,int dims)
{

    float n(0.f);
    float d(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[8];
#endif


#if CV_AVX

    __m256 _n = _mm256_setzero_ps();
    __m256 _d = _n;

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _a = _mm256_loadu_ps(a);
        __m256 _b = _mm256_loadu_ps(b);

        _n = _mm256_add_ps(_n, _mm256_min_ps(_a,_b));
        _d = _mm256_add_ps(_d, _mm256_add_ps(_a,_b));
    }

    _mm256_stream_ps(buffer,_n);
    _mm256_stream_ps(buffer+8,_d);

    for(int i=0, j=8;i<8;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }

#endif

//#elif CV_SSE

//    float CV_DECL_ALIGNED(0x10) buffer[8];

//    __m128 _n = _mm_setzero_ps();
//    __m128 _d = _n;

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _a = _mm_loadu_ps(a);
//        __m128 _b = _mm_loadu_ps(b);

//        _n = _mm_add_ps(_n, _mm_min_ps(_a,_b));
//        _d = _mm_add_ps(_d, _mm_add_ps(_a,_b));
//    }

//    _mm_stream_ps(buffer,_n);
//    _mm_stream_ps(buffer+4,_d);

//    for(int i=0;i<4;++i)
//    {
//        n+=buffer[i];
//        d+=buffer[i+4];
//    }

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();
    cv::v_float32x4 v_d = v_n;

    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n += cv::v_min(v_p, v_q);
        v_d += v_p + v_q;
    }

    cv::v_store_aligned(buffer, v_n);
    cv::v_store_aligned(buffer+4, v_d);

    for(int i=0, j=4;i<4;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = std::min(a0, b0);
        float tmp_n1 = std::min(a1, b1);

        n+= tmp_n0;
        n+= tmp_n1;


        float tmp_d0 = a0 + b0;
        float tmp_d1 = a1 + b1;

        d+= tmp_d0;
        d+= tmp_d1;



        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = std::min(a0, b0);
        tmp_n1 = std::min(a1, b1);

        n+= tmp_n0;
        n+= tmp_n1;


        tmp_d0 = a0 + b0;
        tmp_d1 = a1 + b1;

        d+= tmp_d0;
        d+= tmp_d1;
    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = a[0];
        float q = b[0];

        n+= std::min(p, q);

        d+= p + q;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/d);

}

float normMotyka(const float*  a,const float*  b,int dims)
{
    return 1.f-simMotyka(a,b,dims);
}

float simKulczynski(const float*  a,const float*  b,int dims)
{

    float n(0.f);
    float d(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[8];
#endif


#if CV_AVX

    __m256 _n = _mm256_setzero_ps();
    __m256 _d = _n;

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _a = _mm256_loadu_ps(a);
        __m256 _b = _mm256_loadu_ps(b);

        __m256 tmp = _mm256_sub_ps(_a, _b);
        tmp = _mm256_and_ps(absmask, tmp);

        _n = _mm256_add_ps(_n, _mm256_min_ps(_a,_b));
        _d = _mm256_add_ps(_d, tmp);
    }

    _mm256_stream_ps(buffer,_n);
    _mm256_stream_ps(buffer+8,_d);

    for(int i=0, j=8;i<8;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }
#endif

//#elif CV_SSE2

//    float CV_DECL_ALIGNED(0x10) buffer[8];

//    __m128 _n = _mm_setzero_ps();
//    __m128 _d = _n;

//    static const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));


//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _a = _mm_loadu_ps(a);
//        __m128 _b = _mm_loadu_ps(b);

//        __m128 tmp = _mm_sub_ps(_a,_b);
//        tmp = _mm_and_ps(absmask, tmp);

//        _n = _mm_add_ps(_n, _mm_min_ps(_a,_b));
//        _d = _mm_add_ps(_d, tmp);
//    }

//    _mm_stream_ps(buffer,_n);
//    _mm_stream_ps(buffer+4,_d);

//    for(int i=0, j=4;i<4;i++, j++)
//    {
//        n+=buffer[i];
//        d+=buffer[j];
//    }

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();
    cv::v_float32x4 v_d = v_n;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n += cv::v_min(v_p, v_q);
        v_d += cv::v_abs(v_p - v_q);
    }

    cv::v_store_aligned(buffer,v_n);
    cv::v_store_aligned(buffer+4,v_d);

    for(int i=0, j=4;i<4;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = std::min(a0, b0);
        float tmp_n1 = std::min(a1, b1);

        n+= tmp_n0;
        n+= tmp_n1;


        float tmp_d0 = std::abs(a0 - b0);
        float tmp_d1 = std::abs(a1 - b1);

        d+= tmp_d0;
        d+= tmp_d1;



        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = std::min(a0, b0);
        tmp_n1 = std::min(a1, b1);

        n+= tmp_n0;
        n+= tmp_n1;


        tmp_d0 = std::abs(a0 - b0);
        tmp_d1 = std::abs(a1 - b1);

        d+= tmp_d0;
        d+= tmp_d1;
    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = a[0];
        float q = b[0];

        n+= std::min(p, q);

        d+= std::abs(p - q);
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/d);

}

float simRuzicka(const float*  a,const float*  b,int dims)
{

    float n(0.f);
    float d(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[8];
#endif



#if CV_AVX

    __m256 _n = _mm256_setzero_ps();
    __m256 _d = _n;

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _a = _mm256_loadu_ps(a);
        __m256 _b = _mm256_loadu_ps(b);

        _n = _mm256_add_ps(_n, _mm256_min_ps(_a, _b));
        _d = _mm256_add_ps(_d, _mm256_max_ps(_a, _b));
    }

    _mm256_stream_ps(buffer, _n);
    _mm256_stream_ps(buffer + 8, _d);

    for(int i=0, j=8;i<8;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }
#endif

//#elif CV_SSE


//    __m128 _n = _mm_setzero_ps();
//    __m128 _d = _n;

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _a = _mm_loadu_ps(a);
//        __m128 _b = _mm_loadu_ps(b);

//        _n = _mm_add_ps(_n, _mm_min_ps(_a, _b));
//        _d = _mm_add_ps(_d, _mm_max_ps(_a, _b));
//    }

//    _mm_stream_ps(buffer, _n);
//    _mm_stream_ps(buffer + 4, _d);

//    for(int i=0, j=0;i<4;i++, j++)
//    {
//        n+=buffer[i];
//        d+=buffer[j];
//    }

//#endif

#if CV_SIMD128


    cv::v_float32x4 v_n = cv::v_setzero_f32();
    cv::v_float32x4 v_d = v_n;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n += cv::v_min(v_p, v_q);
        v_d += cv::v_max(v_p, v_q);
    }

    cv::v_store_aligned(buffer, v_n);
    cv::v_store_aligned(buffer+4, v_d);

    for(int i=0, j=4;i<4;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        n += std::min(a0, b0);
        n += std::min(a1, b1);

        d += std::max(a0, b0);
        d += std::max(a1, b1);


        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        n += std::min(a0, b0);
        n += std::min(a1, b1);

        d += std::max(a0, b0);
        d += std::max(a1, b1);
    }


#endif

    for(;l<dims;++l,++a,++b)
    {
        float p = *a;
        float q = *b;

        n += std::min(p, q);
        d += std::max(p, q);
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/d);

}

float normTanimoto(const float*  a,const float*  b,int dims)
{

    float n(0.f);
    float d(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[8];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();
    __m256 _d = _n;

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _a = _mm256_loadu_ps(a);
        __m256 _b = _mm256_loadu_ps(b);

        __m256 mn = _mm256_min_ps(_a, _b);
        __m256 mx = _mm256_max_ps(_a, _b);

        mn = _mm256_sub_ps(mx,mn);

        _n = _mm256_add_ps(_n, mn);
        _d = _mm256_add_ps(_d, mx);
    }

    _mm256_stream_ps(buffer, _n);
    _mm256_stream_ps(buffer + 8, _d);

    for(int i=0, j=8;i<8;i++, j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }
#endif

//#elif CV_SSE

//    float CV_DECL_ALIGNED(0x10) buffer[8];

//    __m128 _n = _mm_setzero_ps();
//    __m128 _d = _n;

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _a = _mm_loadu_ps(a);
//        __m128 _b = _mm_loadu_ps(b);

//        __m128 mn = _mm_min_ps(_a, _b);
//        __m128 mx = _mm_max_ps(_a, _b);

//        _n = _mm_add_ps(_n, _mm_sub_ps(mx,mn));
//        _d = _mm_add_ps(_d, mx);
//    }

//    _mm_stream_ps(buffer, _n);
//    _mm_stream_ps(buffer + 4, _d);

//    for(int i=0;i<4;++i)
//    {
//        n+=buffer[i];
//        d+=buffer[i+4];
//    }

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();
    cv::v_float32x4 v_d = v_n;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_mn = cv::v_min(v_p, v_q);
        cv::v_float32x4 v_mx = cv::v_max(v_p, v_q);

        v_n += v_mx - v_mn;
        v_d += v_mx;
    }

    cv::v_store_aligned(buffer, v_n);
    cv::v_store_aligned(buffer+4, v_d);

    for(int i=0,j=4;i<4;i++,j++)
    {
        n+=buffer[i];
        d+=buffer[j];
    }

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float mn0 = std::min(a0, b0);
        float mn1 = std::min(a1, b1);

        float mx0 = std::max(a0, b0);
        float mx1 = std::max(a1, b1);

        mn0 = mx0 - mn0;
        mn1 = mx1 - mn1;

        n += mn0;
        n += mn1;

        d += mx0;
        d += mx1;



        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        mn0 = std::min(a0, b0);
        mn1 = std::min(a1, b1);

        mx0 = std::max(a0, b0);
        mx1 = std::max(a1, b1);

        mn0 = mx0 - mn0;
        mn1 = mx1 - mn1;

        n += mn0;
        n += mn1;

        d += mx0;
        d += mx1;
    }


#endif

    for(;l<dims;++l,++a,++b)
    {
        float p = *a;
        float q = *b;

        float mn = std::min(p, q);
        float mx = std::max(p, q);

        n+= mx-mn;
        d+= mx;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/d);

}

float simInnerProduct(const float*  a,const float*  b,int dims)
{
    float s(0.f);

    int l(0);

#if CV_AVX
   float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
   float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _s = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {

        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

#if CV_FMA3
        _s = _mm256_fmadd_ps(_p, _q, _s);
#else
        _s = _mm256_add_ps(_s,_mm256_mul_ps(_p,_q));
#endif

    }

    _mm256_stream_ps(buffer, _s);

    for(int i=0;i<8;++i)
        s+=buffer[i];

#endif
//#elif CV_SSE

//    __m128 _s = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {

//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//#if CV_FMA3
//        _s = _mm_fmadd_ps(_p, _q, _s);
//#elif
//        _s = _mm_add_ps(_s,_mm256_mul_ps(_p,_q));
//#endif

//    }

//    _mm_stream_ps(buffer, _s);

//    for(int i=0;i<4;++i)
//        s+=buffer[i];

//#endif


#if CV_SIMD128

    cv::v_float32x4 v_s = cv::v_setzero_f32();

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_s = cv::v_muladd(v_p, v_q, v_s);
    }

    cv::v_store_aligned(buffer, v_s);

    for(int i=0;i<4;i++)
        s+=buffer[i];

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

#if FP_FAST_FMAF
        s = std::fmaf(p0, q0, s);
        s = std::fmaf(p1, q1, s);
#else
        s+= p0*q0;
        s+= p1*q1;
#endif

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

#if FP_FAST_FMAF
        s = std::fmaf(p0, q0, s);
        s = std::fmaf(p1, q1, s);
#else
        s+= p0*q0;
        s+= p1*q1;
#endif
    }

#endif

    for(;l<dims; l++, a++, b++)
#if FP_FAST_FMAF
        s = std::fmaf(a[0], b[0],s);
#else
        s += a[0] * b[0];
#endif

#if CV_SSE2
   _mm_mfence();
#endif

    return s;

}

float simHarmonicMean(const float*  a,const float*  b,int dims)
{

    float s(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#else
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _s = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 _num = _mm256_mul_ps(_p, _q);
        __m256 _den = _mm256_add_ps(_p, _q);

        _s = _mm256_add_ps(_s, _mm256_div_ps(_num, _den) );
    }

    _mm256_stream_ps(buffer, _s);

    for(int i=0; i<8; ++i)
        s += buffer[i];
#endif

//#elif CV_SSE

//    float CV_DECL_ALIGNED(0x10) buffer[4];

//    __m128 _s = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 _num = _mm_mul_ps(_p, _q);
//        __m128 _den = _mm_add_ps(_p, _q);

//        _s = _mm_add_ps(_s, _mm_div_ps(_num, _den) );
//    }

//    _mm_stream_ps(buffer, _s);

//    for(int i=0; i<4; ++i)
//        s += buffer[i];

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_s = cv::v_setzero_f32();

    for(;l<dims-4;l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_n = v_p * v_q;
        cv::v_float32x4 v_d = v_p + v_q;

        v_s += v_n/v_d;
    }

    cv::v_store_aligned(buffer, v_s);

    for(int i=0;i<4;i++)
        s+=buffer[i];

#elif CV_ENABLE_UNROLLED

    for(;l<dims-4;l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float num0 = p0 * q0;
        float num1 = p1 * q1;

        float den0 = p0 + q0;
        float den1 = p1 + q1;

        s+= num0/den0;
        s+= num1/den1;



        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        num0 = p0 * q0;
        num1 = p1 * q1;

        den0 = p0 + q0;
        den1 = p1 + q1;

        s+= num0/den0;
        s+= num1/den1;
    }

#endif

    for(;l<dims; l++, a++, b++)
    {

        float p = *a;
        float q = *b;

        float num = p * q;
        float den = p + q;

        s+= num/den;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return 2.f*s;

}

float simCosine(const float*  a,const float*  b,int dims)
{

    float n(0.f);
    float d1(0.f);
    float d2(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[24];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[12];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();
    __m256 _d1 = _n;
    __m256 _d2 = _n;

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

#if CV_FMA3
        _n = _mm256_fmadd_ps(_p,_q,_n);
        _d1 = _mm256_fmadd_ps(_p,_p,_d1);
        _d2 = _mm256_fmadd_ps(_q,_q,_d2);
#else
        _n = _mm256_add_ps(_n, _mm256_mul_ps(_p, _q));

        _p = _mm256_mul_ps(_p,_p);
        _q = _mm256_mul_ps(_q,_q);

        _d1 = _mm256_add_ps(_d1, _p);
        _d2 = _mm256_add_ps(_d2, _q);
#endif

    }

    _mm256_stream_ps(buffer,_n);
    _mm256_stream_ps(buffer + 8,_d1);
    _mm256_stream_ps(buffer + 16,_d2);

    for(int i=0, j=8, k=16;i<8;i++, j++, k++)
    {
        n+=buffer[i];
        d1+=buffer[j];
        d2+=buffer[k];
    }
#endif

    //#elif CV_SSE

    //    __m128 _n = _mm_setzero_ps();
    //    __m128 _d1 = _n;
    //    __m128 _d2 = _n;

    //    for(;l<=dims-4; l+=4, a+=4, b+=4)
    //    {
    //        __m128 _p = _mm_loadu_ps(a);
    //        __m128 _q = _mm_loadu_ps(b);

    //#if CV_FMA3
    //        _n = _mm_fmadd_ps(_p,_q,_n);
    //        _d1 = _mm_fmadd_ps(_p,_p,_d1);
    //        _d2 = _mm_fmadd_ps(_q,_q,_d2);
    //#else
    //        _n = _mm_add_ps(_n, _mm_mul_ps(_p, _q));

    //        _p = _mm_mul_ps(_p,_p);
    //        _q = _mm_mul_ps(_q,_q);

    //        _d1 = _mm_add_ps(_d1, _p);
    //        _d2 = _mm_add_ps(_d2, _q);
    //#endif

    //    }
    //.
    //    _mm_stream_ps(buffer,_n);
    //    _mm_stream_ps(buffer + 4,_d1);
    //    _mm_stream_ps(buffer + 8,_d2);

    //    for(int i=0;i<4;++i)
    //    {
    //        n+=buffer[i];
    //        d1+=buffer[i+4];
    //        d2+=buffer[i+8];
    //    }

    //#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();
    cv::v_float32x4 v_d1 = v_n;
    cv::v_float32x4 v_d2 = v_n;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_n = cv::v_muladd(v_p, v_q, v_n);
        v_d1 = cv::v_muladd(v_p, v_p, v_d1);
        v_d2 = cv::v_muladd(v_q, v_q, v_d2);
    }

    cv::v_store_aligned(buffer, v_n);
    cv::v_store_aligned(buffer + 4, v_d1);
    cv::v_store_aligned(buffer + 8, v_d2);

    for(int i=0, j=4, k=8; i<8;i++,j++,k++)
    {
        n+=buffer[i];
        d1+=buffer[j];
        d2+=buffer[k];
    }

#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

#if FP_FAST_FMAF
        n = std::fmaf(p0, q0, n);
        n = std::fmaf(p1, q1, n);

        d1 = std::fmaf(p0, p0, d1);
        d1 = std::fmaf(p1, p1, d1);

        d2 = std::fmaf(q0, q0, d2);
        d2 = std::fmaf(q1, q1, d2);
#else
        n += p0*q0;
        n += p1*q1;

        d1 += p0*p0;
        d1 += p1*p1;

        d2 += q0*q0;
        d2 += q1*q1;
#endif


        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

#if FP_FAST_FMAF
        n = std::fmaf(p0, q0, n);
        n = std::fmaf(p1, q1, n);

        d1 = std::fmaf(p0, p0, d1);
        d1 = std::fmaf(p1, p1, d1);

        d2 = std::fmaf(q0, q0, d2);
        d2 = std::fmaf(q1, q1, d2);
#else
        n += p0*q0;
        n += p1*q1;

        d1 += p0*p0;
        d1 += p1*p1;

        d2 += q0*q0;
        d2 += q1*q1;
#endif

    }
#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = a[0];
        float q = b[0];

#if FP_FAST_FMA
        n = std::fma(p,q,n);
        d1 = std::fma(p,p,d1);
        d2 = std::fma(q,q,d2);
#else
        n+= p*q;
        d1+=p*p;
        d2+=q*q;
#endif

    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (n/(std::sqrt(d1) * std::sqrt(d2) ) );

}

float simKumarHassebrook(const float*  a,const float*  b,int dims)
{
    float t1(0.f);
    float t2(0.f);
    float t3(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[24];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[12];
#endif

#if CV_AVX

    __m256 _t1 = _mm256_setzero_ps();
    __m256 _t2 = _t1;
    __m256 _t3 = _t1;

    for(;l<=dims-8;l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

#if CV_FMA3
        _t1 = _mm256_fmadd_ps(_p,_q,_t1);
        _t2 = _mm256_fmadd_ps(_p,_p,_t2);
        _t3 = _mm256_fmadd_ps(_q,_q,_t3);
#else
        _t1 = _mm256_add_ps(_t1, _mm256_mul_ps(_p,_q));
        _t2 = _mm256_add_ps(_t2, _mm256_mul_ps(_p,_p));
        _t3 = _mm256_add_ps(_t3, _mm256_mul_ps(_q,_q));
#endif
    }

    _mm256_stream_ps(buffer, _t1);
    _mm256_stream_ps(buffer + 8, _t2);
    _mm256_stream_ps(buffer + 16, _t3);

    for(int i=0, j=8, k=16;i<8;i++, j++, k++)
    {
        t1+=buffer[i];
        t2+=buffer[j];
        t3+=buffer[k];
    }

#endif

//#elif CV_SSE


//    __m128 _t1 = _mm_setzero_ps();
//    __m128 _t2 = _t1;
//    __m128 _t3 = _t1;

//    for(;l<dims-4;l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//#if CV_FMA3
//        _t1 = _mm_fmadd_ps(_p,_q,_t1);
//        _t2 = _mm_fmadd_ps(_p,_p,_t2);
//        _t3 = _mm_fmadd_ps(_q,_q,_t3);
//#else
//        _t1 = _mm_add_ps(_t1, _mm_mul_ps(_p,_q));
//        _t2 = _mm_add_ps(_t2, _mm_mul_ps(_p,_p));
//        _t3 = _mm_add_ps(_t3, _mm_mul_ps(_q,_q));
//#endif
//    }

//    _mm_stream_ps(buffer, _t1);
//    _mm_stream_ps(buffer + 4, _t2);
//    _mm_stream_ps(buffer + 8, _t3);

//    for(int i=0;i<4;++i)
//    {
//        t1+=buffer[i];
//        t2+=buffer[i + 4];
//        t3+=buffer[i + 8];
//    }

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_t1 = cv::v_setzero_f32();
    cv::v_float32x4 v_t2 = v_t1;
    cv::v_float32x4 v_t3 = v_t1;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_t1 = cv::v_muladd(v_p, v_q, v_t1);
        v_t2 = cv::v_muladd(v_p, v_p, v_t2);
        v_t3 = cv::v_muladd(v_q, v_q, v_t3);
    }

    cv::v_store_aligned(buffer, v_t1);
    cv::v_store_aligned(buffer + 4, v_t2);
    cv::v_store_aligned(buffer + 8, v_t3);

    for(int i=0, j=4, k=8; i<4; i++, j++, k++)
    {
       t1+=buffer[i];
       t2+=buffer[j];
       t3+=buffer[k];
    }

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

#if FP_FAST_FMAF
        t1 = std::fmaf(p0,q0,t1);
        t1 = std::fmaf(p1,q1,t1);

        t2 = std::fmaf(p0,p0,t2);
        t2 = std::fmaf(p1,p1,t2);

        t3 = std::fmaf(q0,q0,t3);
        t3 = std::fmaf(q1,q1,t3);
#else
        t1 += p0*q0;
        t1 += p1*q1;

        t2 += p0*p0;
        t2 += p1*p1;

        t3 += q0*q0;
        t3 += q1*q1;
#endif

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

#if FP_FAST_FMAF
        t1 = std::fmaf(p0,q0,t1);
        t1 = std::fmaf(p1,q1,t1);

        t2 = std::fmaf(p0,p0,t2);
        t2 = std::fmaf(p1,p1,t2);

        t3 = std::fmaf(q0,q0,t3);
        t3 = std::fmaf(q1,q1,t3);
#else
        t1 += p0*q0;
        t1 += p1*q1;

        t2 += p0*p0;
        t2 += p1*p1;

        t3 += q0*q0;
        t3 += q1*q1;
#endif

    }

#endif

    for(;l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

#ifdef FP_FAST_FMAF
        t1 = std::fmaf(p, q, t1);
        t2 = std::fmaf(p, p, t2);
        t3 = std::fmaf(q, q, t3);
#else
        t1 += p * q;
        t2 += p * p;
        t3 += q * q;
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return (t1 / (t2 + t3 - t1));

}

float simJaccard(const float*  a,const float*  b,int dims)
{
    return simKumarHassebrook(a,b,dims);
}

float normJaccard(const float*  a,const float*  b,int dims)
{
    return 1.f - simJaccard(a,b,dims);
}

float simDice(const float*  a,const float*  b,int dims)
{
    float t1(0.f);
    float t2(0.f);
    float t3(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[24];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[12];
#endif

#if CV_AVX

    __m256 _t1 = _mm256_setzero_ps();
    __m256 _t2 = _t1;
    __m256 _t3 = _t1;

    for(;l<=dims-8;l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

#if CV_FMA3

        _t1 = _mm256_fmadd_ps(_p,_q,_t1);
        _t2 = _mm256_fmadd_ps(_p,_p,_t2);
        _t3 = _mm256_fmadd_ps(_q,_q,_t3);
#else

        _t1 = _mm256_add_ps(_t1, _mm256_mul_ps(_p,_q));
        _t2 = _mm256_add_ps(_t2, _mm256_mul_ps(_p,_p));
        _t3 = _mm256_add_ps(_t3, _mm256_mul_ps(_q,_q));

#endif
    }

    _mm256_stream_ps(buffer, _t1);
    _mm256_stream_ps(buffer + 8, _t2);
    _mm256_stream_ps(buffer + 16, _t3);

    for(int i=0;i<8;++i)
    {
        t1+=buffer[i];
        t2+=buffer[i + 8];
        t3+=buffer[i + 16];
    }
#endif

//#elif CV_SSE

//    __m128 _t1 = _mm_setzero_ps();
//    __m128 _t2 = _t1;
//    __m128 _t3 = _t1;

//    for(;l<dims-4;l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//#if CV_FMA3

//        _t1 = _mm_fmadd_ps(_p,_q,_t1);
//        _t2 = _mm_fmadd_ps(_p,_p,_t2);
//        _t3 = _mm_fmadd_ps(_q,_q,_t3);
//#else

//        _t1 = _mm_add_ps(_t1, _mm_mul_ps(_p,_q));
//        _t2 = _mm_add_ps(_t2, _mm_mul_ps(_p,_q));
//        _t3 = _mm_add_ps(_t3, _mm_mul_ps(_p,_q));

//#endif
//    }

//    _mm_stream_ps(buffer, _t1);
//    _mm_stream_ps(buffer + 4, _t2);
//    _mm_stream_ps(buffer + 8, _t3);

//    for(int i=0;i<4;++i)
//    {
//        t1+=buffer[i];
//        t2+=buffer[i + 4];
//        t3+=buffer[i + 8];
//    }

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_t1 = cv::v_setzero_f32();
    cv::v_float32x4 v_t2 = v_t1;
    cv::v_float32x4 v_t3 = v_t1;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_t1 = cv::v_muladd(v_p, v_q, v_t1);
        v_t2 = cv::v_muladd(v_p, v_p, v_t2);
        v_t3 = cv::v_muladd(v_q, v_q, v_t3);
    }

    cv::v_store_aligned(buffer, v_t1);
    cv::v_store_aligned(buffer + 4, v_t2);
    cv::v_store_aligned(buffer + 8, v_t3);

    for(int i=0, j=4, k=8; i<4; i++, j++, k++)
    {
       t1+=buffer[i];
       t2+=buffer[j];
       t3+=buffer[k];
    }


#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

#if FP_FAST_FMA
        t1 = std::fma(p0,q0,t1);
        t1 = std::fma(p1,q1,t1);

        t2 = std::fma(p0,p0,t2);
        t2 = std::fma(p1,p1,t2);

        t3 = std::fma(q0,q0,t3);
        t3 = std::fma(q1,q1,t3);
#else
        t1 += p0*q0;
        t1 += p1*q1;

        t2 += p0*p0;
        t2 += p1*p1;

        t3 += q0*q0;
        t3 += q1*q1;
#endif

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

#if FP_FAST_FMA
        t1 = std::fma(p0,q0,t1);
        t1 = std::fma(p1,q1,t1);

        t2 = std::fma(p0,p0,t2);
        t2 = std::fma(p1,p1,t2);

        t3 = std::fma(q0,q0,t3);
        t3 = std::fma(q1,q1,t3);
#else
        t1 += p0*q0;
        t1 += p1*q1;

        t2 += p0*p0;
        t2 += p1*p1;

        t3 += q0*q0;
        t3 += q1*q1;
#endif


    }

#endif

    for(;l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

#ifdef FP_FAST_FMAF
        t1 = std::fmaf(p, q, t1);
        t2 = std::fmaf(p, p, t2);
        t3 = std::fmaf(q, q, t3);
#else
        t1 += p * q;
        t2 += p * p;
        t3 += q * q;
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return ( (2.f*t1) / (t2 + t3 ));
}

float normDice(const float*  a,const float*  b,int dims)
{
    return 1.f - simDice(a,b,dims);
}

float simFidelity(const float*  a,const float*  b,int dims)
{
    float s(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _s = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {

        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        _p = _mm256_mul_ps(_p, _q);
        _p = _mm256_sqrt_ps(_p);

        _s = _mm256_add_ps(_s,_p);
    }

    _mm256_stream_ps(buffer, _s);

    for(int i=0;i<8;++i)
        s+=buffer[i];

#endif

//#elif CV_SSE
//    float CV_DECL_ALIGNED(0x10) buffer[4];

//    __m128 _s = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {

//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        _p = _mm_mul_ps(_p, _q);
//        _p = _mm_sqrt_ps(_p);

//        _s = _mm_add_ps(_s,_p);
//    }

//    _mm_stream_ps(buffer, _s);

//    for(int i=0;i<4;++i)
//        s+=buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_s = cv::v_setzero_f32();

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        v_s += cv::v_sqrt(v_p*v_q);
    }

    cv::v_store_aligned(buffer, v_s);

    for(int i=0; i<4;i++)
        s+=buffer[i];


#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        s+= std::sqrt(p0*q0);
        s+= std::sqrt(p1*q1);

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        s+= std::sqrt(p0*q0);
        s+= std::sqrt(p1*q1);
    }

#endif

    for(;l<dims; l++, a++, b++)
        s += std::sqrt(*a * *b);

#if CV_SSE2
   _mm_mfence();
#endif

    return s;

}

float normBhattacharyya(const float*  a,const float*  b,int dims)
{
    return -std::log(simFidelity(a,b,dims));
}

float normHellinger(const float*  a,const float*  b,int dims)
{
    return 2.f*std::sqrt(1.f-simFidelity(a,b,dims));
}

float normMatusita(const float*  a,const float*  b,int dims)
{
#ifdef FP_FAST_FMAF
    return std::sqrt(std::fmaf(-2.f,simFidelity(a,b,dims),2.f));
#else
    return std::sqrt(2-2*simFidelity(a,b,dims));
#endif

}

float simSquaredChord(const float*  a,const float*  b,int dims)
{

#ifdef FP_FAST_FMAF
    return std::fmaf(2.f,simFidelity(a,b,dims),-1.f);
#else
    return 2.f * simFidelity(a,b,dims) - 1.f;
#endif

}

float normSquaredChord(const float*  a,const float*  b,int dims)
{
    return 1.f-simSquaredChord(a,b,dims);
}

float normPearson_ChiSqr(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    float CV_DECL_ALIGNED(0x20) buffer[8];

    __m256 _n = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 _num = _mm256_sub_ps(_p,_q);
        _num = _mm256_mul_ps(_num, _num);
        _num = _mm256_div_ps(_num, _q);

        _n = _mm256_add_ps(_n, _num);
    }

    _mm256_stream_ps(buffer, _n);

    for(int i=0;i<8;++i)
        n+=buffer[i];
#endif

//#elif CV_SSE

//    float CV_DECL_ALIGNED(0x10) buffer[4];

//    __m128 _n = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 _num = _mm_sub_ps(_p,_q);
//        _num = _mm_mul_ps(_num, _num);
//        _num = _mm_div_ps(_num, _q);

//        _n = _mm_add_ps(_n, _num);
//    }

//    _mm_stream_ps(buffer, _n);

//    for(int i=0;i<4;++i)
//        n+=buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_t = v_p - v_q;
        v_t *= v_t;

        v_n += v_t/v_q;
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;i++)
        n+=buffer[i];

#elif CV_ENABLE_UNROLLED

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {

        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float tmp0 = p0 - q0;
        float tmp1 = p1 - q1;

        tmp0 *= tmp0;
        tmp1 *= tmp1;

        tmp0 /= q0;
        tmp1 /= q1;

        n+= tmp0;
        n+= tmp1;



        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        tmp0 = p0 - q0;
        tmp1 = p1 - q1;

        tmp0 *= tmp0;
        tmp1 *= tmp1;

        tmp0 /= q0;
        tmp1 /= q1;

        n+= tmp0;
        n+= tmp1;
    }
#endif

    for(; l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp = p - q;
        tmp*=tmp;

        n += tmp/q;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}

float normNeyman_ChiSqr(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 _num = _mm256_sub_ps(_p,_q);
        _num = _mm256_mul_ps(_num, _num);
        _num = _mm256_div_ps(_num, _p);

        _n = _mm256_add_ps(_n, _num);
    }

    _mm256_stream_ps(buffer, _n);

    for(int i=0;i<8;++i)
        n+=buffer[i];
#endif

//#elif CV_SSE

//    __m128 _n = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 _num = _mm_sub_ps(_p,_q);
//        _num = _mm_mul_ps(_num, _num);
//        _num = _mm_div_ps(_num, _p);

//        _n = _mm_add_ps(_n, _num);
//    }

//    _mm_stream_ps(buffer, _n);

//    for(int i=0;i<4;++i)
//        n+=buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_t = v_p - v_q;
        v_t*=v_t;

        v_n += v_t/v_p;
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;i++)
        n+=buffer[i];

#elif CV_ENABLE_UNROLLED

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {

        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float tmp0 = p0 - q0;
        float tmp1 = p1 - q1;

        tmp0 *= tmp0;
        tmp1 *= tmp1;

        n+= tmp0/p0;
        n+= tmp1/p1;



        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        tmp0 = p0 - q0;
        tmp1 = p1 - q1;

        tmp0 *= tmp0;
        tmp1 *= tmp1;

        n+= tmp0/p0;
        n+= tmp1/p1;
    }

#endif

    for(; l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp = p - q;

        tmp *= tmp;

        n+= tmp/p;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}

float normSquared_ChiSqr(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 _num = _mm256_sub_ps(_p,_q);
        _num = _mm256_mul_ps(_num, _num);

        __m256 _den = _mm256_add_ps(_p,_q);

        _n = _mm256_add_ps(_n, _mm256_div_ps(_num,_den));
    }

    _mm256_stream_ps(buffer, _n);

    for(int i=0;i<8;++i)
        n+=buffer[i];
#endif

//#elif CV_SSE

//    __m128 _n = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 _num = _mm_sub_ps(_p,_q);
//        _num = _mm_mul_ps(_num, _num);

//        __m256 _den = _mm256_add_ps(_p,_q);

//        _n = _mm_add_ps(_n, _mm_div_ps(_num,_den));
//    }

//    _mm_stream_ps(buffer, _n);

//    for(int i=0;i<4;++i)
//        n+=buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_t = v_p-v_q;
        v_t*=v_t;

        v_n += v_t/(v_p+v_q);
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;i++)
        n+=buffer[i];

#elif CV_ENABLE_UNROLLED

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {

        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float tmp0 = p0 - q0;
        float tmp1 = p1 - q1;

        tmp0 *= tmp0;
        tmp1 *= tmp1;

        n+= tmp0/(p0 + q0);
        n+= tmp1/(p1 + q1);



        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        tmp0 = p0 - q0;
        tmp1 = p1 - q1;

        tmp0 *= tmp0;
        tmp1 *= tmp1;

        n+= tmp0/(p0 + q0);
        n+= tmp1/(p1 + q1);
    }

#endif

    for(; l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp = p - q;

        tmp *= tmp;

        n+= tmp/(p + q);
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}


float normProbabilisticSymmetric_ChiSqr(const float*  a,const float*  b,int dims)
{
    return 2.f*normSquared_ChiSqr(a,b,dims);
}

float normDivergence(const float*  a,const float*  b,int dims)
{
    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 _num = _mm256_sub_ps(_p,_q);
        _num = _mm256_mul_ps(_num, _num);

        __m256 _den = _mm256_add_ps(_p,_q);
        _den = _mm256_mul_ps(_den, _den);

        _n = _mm256_add_ps(_n, _mm256_div_ps(_num,_den));
    }

    _mm256_stream_ps(buffer, _n);

    for(int i=0;i<8;++i)
        n+=buffer[i];
#endif

//#elif CV_SSE

//    __m128 _n = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 _num = _mm_sub_ps(_p,_q);
//        _num = _mm_mul_ps(_num, _num);

//        __m256 _den = _mm256_add_ps(_p,_q);
//        _den = _mm256_mul_ps(_den, _den);

//        _n = _mm_add_ps(_n, _mm_div_ps(_num,_den));
//    }

//    _mm_stream_ps(buffer, _n);

//    for(int i=0;i<4;++i)
//        n+=buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_t1 = v_p - v_q;
        cv::v_float32x4 v_t2 = v_p + v_q;

        v_n += (v_t1*v_t1) / (v_t2*v_t2);
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;++i)
        n+=buffer[i];

#elif CV_ENABLE_UNROLLED

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {

        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float num0 = p0 - q0;
        float num1 = p1 - q1;

        float den0 = p0 + q0;
        float den1 = p1 + q1;

        num0 *= num0;
        num1 *= num1;

        den0 *= den0;
        den1 *= den1;

        n+= num0/den0;
        n+= num1/den1;



        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        num0 = p0 - q0;
        num1 = p1 - q1;

        den0 = p0 + q0;
        den1 = p1 + q1;


        num0 *= num0;
        num1 *= num1;

        den0 *= den0;
        den1 *= den1;

        n+= num0/den0;
        n+= num1/den1;
    }

#endif

    for(; l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float num = p - q;
        float den = p + q;

        num *= num;
        den *= den;

        n+= num/den;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return 2.f*n;

}

float normClark(const float*  a,const float*  b,int dims)
{
    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    __m256 _n = _mm256_setzero_ps();

    for(;l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 _num = _mm256_sub_ps(_p,_q);
        _num = _mm256_and_ps(absmask, _num);

        __m256 _den = _mm256_add_ps(_p,_q);

        _num = _mm256_div_ps(_num,_den);
        _num = _mm256_mul_ps(_num,_num);

        _n = _mm256_add_ps(_n, _num);
    }

    _mm256_stream_ps(buffer, _n);

    for(int i=0;i<8;++i)
        n+=buffer[i];
#endif

//#elif CV_SSE2

//    static const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));

//    __m128 _n = _mm_setzero_ps();

//    for(;l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 _num = _mm_sub_ps(_p,_q);
//        _num = _mm_and_ps(absmask,_num);

//        __m256 _den = _mm256_add_ps(_p,_q);

//        _num = _mm_div_ps(_num, _den);
//        _num = _mm_mul_ps(_num,_num);

//        _n = _mm_add_ps(_n, _num);
//    }

//    _mm_stream_ps(buffer, _n);

//    for(int i=0;i<4;++i)
//        n+=buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_t = cv::v_abs(v_p-v_q)/(v_p+v_q);

        v_n = cv::v_muladd(v_t, v_t, v_n);
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;++i)
        n+=buffer[i];

#elif CV_ENABLE_UNROLLED

    for(; l<=dims-4; l+=4, a+=4, b+=4)
    {

        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float num0 = std::abs(p0 - q0);
        float num1 = std::abs(p1 - q1);

        float den0 = p0 + q0;
        float den1 = p1 + q1;

        num0 /= den0;
        num1 /= den1;

        num0 *= num0;
        num1 *= num1;

        n+= num0;
        n+= num1;



        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        num0 = std::abs(p0 - q0);
        num1 = std::abs(p1 - q1);

        den0 = p0 + q0;
        den1 = p1 + q1;

        num0 /= den0;
        num1 /= den1;

        num0 *= num0;
        num1 *= num1;

        n+= num0;
        n+= num1;

    }

#endif

    for(; l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float num = std::abs(p - q);
        float den = p + q;

        num /= den;
        num *= num;

        n+= num;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return std::sqrt(n);
}

float normAdditiveSymmetric_ChiSqr(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();


    for(; l<=dims-8; l+=8, a+=8, b+=8)
    {

        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp = _mm256_sub_ps(_p, _q);

        __m256 _n1 = _mm256_mul_ps(tmp,tmp);
        __m256 _n2 = _mm256_add_ps(_p, _q);
        __m256 _d = _mm256_add_ps(_d, _mm256_mul_ps(_p,_q) );


        _n1 = _mm256_mul_ps(_n1, _n2);
        _n1 = _mm256_div_ps(_n1,_d);

        _n = _mm256_add_ps(_n, _n1);
    }

    _mm256_stream_ps(buffer, _n);

    for(int i=0; i<8; ++i)
        n += buffer[i];
#endif
//#elif CV_SSE

//    __m128 _n = _mm_setzero_ps();

//    for(; l<=dims-4; l+=4, a+=4, b+=4)
//    {

//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp = _mm_sub_ps(_p, _q);

//        __m128 _n1 = _mm_mul_ps(tmp,tmp);
//        __m128 _n2 = _mm_add_ps(_p,_q);
//        __m128 _d = _mm_mul_ps(_p,_q);

//        _n1 = _mm_mul_ps(_n1, _n2);
//        _n1 = _mm_div_ps(_n1,_d);

//        _n = _mm_add_ps(_n, _n1);
//    }

//    _mm_stream_ps(buffer, _n);

//    for(int i=0; i<4; ++i)
//        n += buffer[i];

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_t = v_p - v_q;

        v_n += (v_t * v_t * (v_p + v_q)) / (v_p * v_q);
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0; i<4; ++i)
        n += buffer[i];

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float tmp0 = p0 - q0;
        float tmp1 = p1 - q1;

        float n10 = tmp0 * tmp0;
        float n11 = tmp1 * tmp1;

        float n20 = p0 + q0;
        float n21 = p1 + q1;

        float d0 = p0 * q0;
        float d1 = p1 * q1;

        n += (n10*n20)/d0;
        n += (n11*n21)/d1;



        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        tmp0 = p0 - q0;
        tmp1 = p1 - q1;

        n10 = tmp0 * tmp0;
        n11 = tmp1 * tmp1;

        n20 = p0 + q0;
        n21 = p1 + q1;

        d0 = p0 * q0;
        d1 = p1 * q1;

        n += (n10*n20)/d0;
        n += (n11*n21)/d1;
    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float n1 = p - q;

        float n2 = p + q;

        float d = p * q;

        n1*=n1;

        n += (n1*n2)/d;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}

float normEntKullbackLeibler(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        n += p0 * std::log(p0/q0);
        n += p1 * std::log(p1/q1);

        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        n += p0 * std::log(p0/q0);
        n += p1 * std::log(p1/q1);
    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        n+= p * std::log(p / q);
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}

float normEntJeffreys(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

#ifdef FP_FAST_FMAF
        n = std::fmaf(p0-q0,std::log(p0/q0), n);
        n = std::fmaf(p1-q1,std::log(p1/q1), n);
#else
        n += (p0 - q0) * std::log(p0/q0);
        n += (p1 - q1) * std::log(p1/q1);
#endif
        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

#ifdef FP_FAST_FMAF
        n = std::fmaf(p0-q0,std::log(p0/q0), n);
        n = std::fmaf(p1-q1,std::log(p1/q1), n);
#else
        n += (p0 - q0) * std::log(p0/q0);
        n += (p1 - q1) * std::log(p1/q1);
#endif

    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

#ifdef FP_FAST_FMAF
        n = std::fmaf(p-q,std::log(p/q), n);
#else
        n += (p-q) * std::log(p/q);
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}

float normEntKDivergence(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

#ifdef FP_FAST_FMAF
        n = std::fmaf(p0, std::log((2.f*p0)/(p0 + q0)), n);
        n = std::fmaf(p1, std::log((2.f*p1)/(p1 + q1)), n);
#else
        n += p0 * std::log((2.f*p0)/(p0 + q0));
        n += p1 * std::log((2.f*p1)/(p1 + q1));
#endif

        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

#ifdef FP_FAST_FMAF
        n = std::fmaf(p0, std::log((2.f*p0)/(p0 + q0)), n);
        n = std::fmaf(p1, std::log((2.f*p1)/(p1 + q1)), n);
#else
        n += p0 * std::log((2.f*p0)/(p0 + q0));
        n += p1 * std::log((2.f*p1)/(p1 + q1));
#endif

    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

#ifdef FP_FAST_FMAF
        n = std::fmaf(p, std::log((2.f*p)/(p + q)), n);
#else
        n+= p * std::log( (2.f * p) / (p + q) );
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}

float normEntTopsoe(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float den0 = (p0 + q0);
        float den1 = (p1 + q1);

#ifdef FP_FAST_FMA
        n = std::fmaf(p0, std::log((2.f*p0) / den0 ), std::fmaf(q0, std::log((2.f*q0) / den0), n ) );
        n = std::fmaf(p1, std::log((2.f*p1) / den1 ), std::fmaf(q1, std::log((2.f*q1) / den1), n ) );
#else
        n += p0 * std::log((2.f*p0) / den0 ) + q0 * std::log((2.f*q0) / den0);
        n += p1 * std::log((2.f*p1) / den1) + q1 * std::log((2.f*q1) / den1);
#endif



        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        den0 = (p0 + q0);
        den1 = (p1 + q1);

#ifdef FP_FAST_FMA
        n = std::fmaf(p0, std::log((2.f*p0) / den0 ), std::fmaf(q0, std::log((2.f*q0) / den0), n ) );
        n = std::fmaf(p1, std::log((2.f*p1) / den1 ), std::fmaf(q1, std::log((2.f*q1) / den1), n ) );
#else
        n += p0 * std::log((2.f*p0) / den0 ) + q0 * std::log((2.f*q0) / den0);
        n += p1 * std::log((2.f*p1) / den1) + q1 * std::log((2.f*q1) / den1);
#endif

    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float den = p+q;

#ifdef FP_FAST_FMAF
        n = std::fmaf(p, std::log((2.f*p) / den0 ), std::fmaf(q, std::log((2.f*q) / den), n ) );
#else
        n+= p * std::log( (2.f * p) / den ) + q * std::log( (2.f * q) / den );
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}

float normEntJensen_Shannon(const float*  a,const float*  b,int dims)
{

    float n1(0.f);
    float n2(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float den0 = (p0 + q0);
        float den1 = (p1 + q1);

#ifdef FP_FAST_FMA
        n1 = std::fmaf(p0, std::log((2.f*p0) / den0 ), n1 );
        n1 = std::fmaf(p1, std::log((2.f*p1) / den1 ), n1 );

        n2 = std::fmaf(q0, std::log((2.f*q0) / den0 ), n1 );
        n2 = std::fmaf(q1, std::log((2.f*q1) / den1 ), n1 );
#else
        n1 += p0 * std::log((2.f*p0) / den0 );
        n1 += p1 * std::log((2.f*p1) / den1) ;

        n2 += q0 * std::log((2.f*q0) / den0);
        n2 += q1 * std::log((2.f*q1) / den1);
#endif
        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        den0 = (p0 + q0);
        den1 = (p1 + q1);

#ifdef FP_FAST_FMA
        n1 = std::fmaf(p0, std::log((2.f*p0) / den0 ), n1 );
        n1 = std::fmaf(p1, std::log((2.f*p1) / den1 ), n1 );

        n2 = std::fmaf(q0, std::log((2.f*q0) / den0 ), n1 );
        n2 = std::fmaf(q1, std::log((2.f*q1) / den1 ), n1 );
#else
        n1 += p0 * std::log((2.f*p0) / den0 );
        n1 += p1 * std::log((2.f*p1) / den1) ;

        n2 += q0 * std::log((2.f*q0) / den0);
        n2 += q1 * std::log((2.f*q1) / den1);
#endif

    }
#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float den = p+q;

#ifdef FP_FAST_FMA
        n1 = std::fmaf(p, std::log((2.f*p) / den ), n1 );
        n2 = std::fmaf(q, std::log((2.f*q) / den ), n2 );
#else
        n1 += p * std::log( (2.f * p) / den );
        n2 += q * std::log( (2.f * q) / den );
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return 0.5f * (n1 + n2);

}

float normEntJensenDifference(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float tmp0 = (p0+q0);
        float tmp1 = (p1+q1);

        tmp0*=0.5f;
        tmp1*=0.5f;

#ifdef FP_FAST_FMAF
        n = std::fmaf(0.5f, std::fmaf(p0, std::log(p0), q0 * std::log(q0)), std::fmaf(-tmp0, std::log(tmp0), n));
        n = std::fmaf(0.5f, std::fmaf(p1, std::log(p1), q1 * std::log(q1)), std::fmaf(-tmp1, std::log(tmp1), n));
#else
        n += 0.5f * (p0 * std::log(p0) + q0 * std::log(q0)) - tmp0 * std::log(tmp0);
        n += 0.5f * (p1 * std::log(p1) + q1 * std::log(q1)) - tmp1 * std::log(tmp1);
#endif

        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        tmp0 = (p0+q0);
        tmp1 = (p1+q1);

        tmp0*=0.5f;
        tmp1*=0.5f;


#ifdef FP_FAST_FMAF
        n = std::fmaf(0.5f, std::fmaf(p0, std::log(p0), q0 * std::log(q0)), std::fmaf(-tmp0, std::log(tmp0), n));
        n = std::fmaf(0.5f, std::fmaf(p1, std::log(p1), q1 * std::log(q1)), std::fmaf(-tmp1, std::log(tmp1), n));
#else
        n += 0.5f * (p0 * std::log(p0) + q0 * std::log(q0)) - tmp0 * std::log(tmp0);
        n += 0.5f * (p1 * std::log(p1) + q1 * std::log(q1)) - tmp1 * std::log(tmp1);
#endif
    }
#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp = 0.5f*(p + q);

#ifdef FP_FAST_FMA
        n = std::fmaf(0.5f, std::fmaf(p, std::log(p), q * std::log(q)), std::fmaf(-tmp, std::log(tmp), n));
#else
        n+= 0.5f * (p * std::log(p) + q * std::log(q)) - tmp * std::log(tmp);
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}


float normTaneja(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float tmp0 = p0 + q0;
        float tmp1 = p1 + q1;

#ifdef FP_FAST_FMAF
        tmp0*=0.5f;
        tmp1*=0.5f;

        n = std::fma(tmp0, std::log( tmp0/( 2.f * std::sqrt(p0*q0) ) ), n);
        n = std::fma(tmp1, std::log( tmp1/( 2.f * std::sqrt(p1*q1) ) ), n);
#else
        n += 0.5f * tmp0 * std::log( tmp0/( 2.f * std::sqrt(p0*q0) ) );
        n += 0.5f * tmp1 * std::log( tmp1/( 2.f * std::sqrt(p1*q1) ) );
#endif
        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        tmp0 = p0 + q0;
        tmp1 = p1 + q1;

#ifdef FP_FAST_FMAF
        tmp0*=0.5f;
        tmp1*=0.5f;

        n = std::fma(tmp0, std::log( tmp0/( 2.f * std::sqrt(p0*q0) ) ), n);
        n = std::fma(tmp1, std::log( tmp1/( 2.f * std::sqrt(p1*q1) ) ), n);
#else
        n += 0.5f * tmp0 * std::log( tmp0/( 2.f * std::sqrt(p0*q0) ) );
        n += 0.5f * tmp1 * std::log( tmp1/( 2.f * std::sqrt(p1*q1) ) );
#endif

    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp = p+q;

#ifdef FP_FAST_FMAF
        tmp*=0.5f;

        n = std::fma(tmp, std::log( tmp/( 2.f * std::sqrt(p*q) ) ), n);
#else
        n += 0.5f * tmp * std::log( tmp/( 2.f * std::sqrt(p*q) ) );
#endif
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}

float normKumarJohnson(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float tmp_n0 = q0*q0;
        float tmp_n1 = q1*q1;

        float tmp_d0 = p0*q0;
        float tmp_d1 = p1*q1;

#ifdef FP_FAST_FMA
        tmp_n0 = std::fma(p0,p0,-tmp_n0);
        tmp_n1 = std::fma(p1,p1,-tmp_n1);
#else
        tmp_n0 = p0*p0 - tmp_n0;
        tmp_n1 = p1*p1 - tmp_n1;

#endif
        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        tmp_d0 = 2.f * std::pow(tmp_d0,1.5f);
        tmp_d1 = 2.f * std::pow(tmp_d1,1.5f);

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;

        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        tmp_n0 = q0*q0;
        tmp_n1 = q1*q1;

        tmp_d0 = p0*q0;
        tmp_d1 = p1*q1;

#ifdef FP_FAST_FMA

        tmp_n0 = std::fma(p0,p0,-tmp_n0);
        tmp_n1 = std::fma(p1,p1,-tmp_n1);

#else

        tmp_n0 = p0*p0 - tmp_n0;
        tmp_n1 = p1*p1 - tmp_n1;

#endif

        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        tmp_d0 = 2.f * std::pow(tmp_d0,1.5f);
        tmp_d1 = 2.f * std::pow(tmp_d1,1.5f);

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;
    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp_n = q*q;

        float tmp_d = p*q;

#ifdef FP_FAST_FMA
        tmp_n = std::fma(p,p,-tmp_n);
#else
        tmp_n = p * p - tmp_n;
#endif

        tmp_n *= tmp_n;

        tmp_d = 2.f * std::pow(tmp_d,1.5f);

        n += tmp_n / tmp_d;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}

//float normAvg(const float*  a,const float*  b,int dims);



float normVicisWaveHedges(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];

    __m256 _n = _mm256_setzero_ps();

    static const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for(; l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_and_ps(tmp_n,absmask);

        __m256 tmp_d = _mm256_min_ps(_p,_q);

        tmp_n = _mm256_div_ps(tmp_n,tmp_d);

        _n = _mm256_add_ps(_n,tmp_n);
    }

    _mm256_stream_ps(buffer,_n);

    for(int i=0;i<8;++i)
        n += buffer[i];

#endif
//#elif CV_SSE2

//    __m256 _n = _mm_setzero_ps();

//    static const __m128 absmask = _mm_castsi256_ps(_mm_set1_epi32(0x7fffffff));

//    for(; l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_and_ps(tmp_n,absmask);

//        __m128 tmp_d = _mm_min_ps(_p,_q);

//        tmp_n = _mm_div_ps(tmp_n,tmp_d);

//        _mm_add_ps(n,tmp_n);
//    }

//    _mm_stream_ps(buffer,_n);

//    for(int i=0;i<4;++i)
//        n += buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_tn = cv::v_abs(v_p-v_q);
        cv::v_float32x4 v_td = cv::v_min(v_p, v_q);

        v_n += v_tn / v_td;
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;++i)
        n += buffer[i];

#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = std::abs(a0 - b0);
        float tmp_n1 = std::abs(a1 - b1);

        float tmp_d0 = std::min(a0, b0);
        float tmp_d1 = std::min(a1, b1);

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;


        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = std::abs(a0 - b0);
        tmp_n1 = std::abs(a1 - b1);

        tmp_d0 = std::min(a0, b0);
        tmp_d1 = std::min(a1, b1);

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;
    }
#endif

    for(; l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp_n = std::abs(p - q);

        float tmp_d = std::min(p, q);

        n += tmp_n / tmp_d;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}

float normVicisSymmetric_ChiSqr(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX

    __m256 _n = _mm256_setzero_ps();


    for(; l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_mul_ps(tmp_n, tmp_n);

        __m256 tmp_d = _mm256_min_ps(_p,_q);
        tmp_d = _mm256_mul_ps(tmp_d, tmp_d);

        tmp_n = _mm256_div_ps(tmp_n,tmp_d);

        _n = _mm256_add_ps(_n,tmp_n);
    }

    _mm256_stream_ps(buffer,_n);

    for(int i=0;i<8;++i)
        n += buffer[i];
#endif

//#elif CV_SSE

//    __m256 _n = _mm_setzero_ps();

//    for(; l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_mul_ps(tmp_n, tmp_n);

//        __m128 tmp_d = _mm_min_ps(_p,_q);
//        tmp_d = _mm_mul_ps(tmp_d, tmp_d);

//        tmp_n = _mm_div_ps(tmp_n,tmp_d);

//        _mm_add_ps(n,tmp_n);
//    }

//    _mm_stream_ps(buffer,_n);

//    for(int i=0;i<4;++i)
//        n += buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_tn = cv::v_abs(v_p-v_q);
        cv::v_float32x4 v_td = cv::v_min(v_p, v_q);

        v_td*=v_td;

        v_n += v_tn / v_td;
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;++i)
        n += buffer[i];

#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = a0 - b0;
        float tmp_n1 = a1 - b1;

        float tmp_d0 = std::min(a0, b0);
        float tmp_d1 = std::min(a1, b1);

        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        tmp_d0 *= tmp_d0;
        tmp_d1 *= tmp_d1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;


        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = a0 - b0;
        tmp_n1 = a1 - b1;

        tmp_d0 = std::min(a0, b0);
        tmp_d1 = std::min(a1, b1);

        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        tmp_d0 *= tmp_d0;
        tmp_d1 *= tmp_d1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;
    }
#endif

    for(; l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp_n = p - q;

        float tmp_d = std::min(p, q);

        tmp_n*=tmp_n;

        tmp_d*=tmp_d;

        n += tmp_n / tmp_d;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;
}

float normVicisSymmetric_ChiSqr_2(const float*  a,const float*  b,int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX
    __m256 _n = _mm256_setzero_ps();


    for(; l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_mul_ps(tmp_n, tmp_n);

        __m256 tmp_d = _mm256_min_ps(_p,_q);

        tmp_n = _mm256_div_ps(tmp_n,tmp_d);

        _n = _mm256_add_ps(_n,tmp_n);
    }

    _mm256_stream_ps(buffer,_n);

    for(int i=0;i<8;++i)
        n += buffer[i];
#endif

//#elif CV_SSE

//    __m256 _n = _mm_setzero_ps();

//    for(; l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_mul_ps(tmp_n, tmp_n);

//        __m128 tmp_d = _mm_min_ps(_p,_q);

//        tmp_n = _mm_div_ps(tmp_n,tmp_d);

//        _mm_add_ps(n,tmp_n);
//    }

//    _mm_stream_ps(buffer,_n);

//    for(int i=0;i<4;++i)
//        n += buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();

    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_tn = v_p-v_q;
        cv::v_float32x4 v_td = cv::v_min(v_p, v_q);

        v_n += (v_tn * v_tn) / v_td;
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;++i)
        n += buffer[i];

#elif CV_ENABLE_UNROLLED
    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = a0 - b0;
        float tmp_n1 = a1 - b1;

        float tmp_d0 = std::min(a0, b0);
        float tmp_d1 = std::min(a1, b1);

        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;


        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = a0 - b0;
        tmp_n1 = a1 - b1;

        tmp_d0 = std::min(a0, b0);
        tmp_d1 = std::min(a1, b1);

        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;
    }
#endif

    for(; l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp_n = p - q;

        float tmp_d = std::min(p, q);

        tmp_n*=tmp_n;

        n += tmp_n / tmp_d;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}


float normVicisSymmetric_ChiSqr_3(const float*  a, const float*  b, int dims)
{

    float n(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[8];
#else
    float CV_DECL_ALIGNED(0x10) buffer[4];
#endif

#if CV_AVX
    __m256 _n = _mm256_setzero_ps();


    for(; l<=dims-8; l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 tmp_n = _mm256_sub_ps(_p,_q);
        tmp_n = _mm256_mul_ps(tmp_n, tmp_n);

        __m256 tmp_d = _mm256_max_ps(_p,_q);

        tmp_n = _mm256_div_ps(tmp_n,tmp_d);

        _n = _mm256_add_ps(_n,tmp_n);
    }

    _mm256_stream_ps(buffer,_n);

    for(int i=0;i<8;++i)
        n += buffer[i];
#endif

//#elif CV_SSE

//    __m256 _n = _mm_setzero_ps();

//    for(; l<=dims-4; l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 tmp_n = _mm_sub_ps(_p,_q);
//        tmp_n = _mm_mul_ps(tmp_n, tmp_n);

//        __m128 tmp_d = _mm_max_ps(_p,_q);

//        tmp_n = _mm_div_ps(tmp_n,tmp_d);

//        _mm_add_ps(n,tmp_n);
//    }

//    _mm_stream_ps(buffer,_n);

//    for(int i=0;i<4;++i)
//        n += buffer[i];
//#endif

#if CV_SIMD128

    cv::v_float32x4 v_n = cv::v_setzero_f32();


    for(;l<=dims-4;l+=4, a+=4, b+=4)
    {
        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_tn = v_p - v_q;
        cv::v_float32x4 v_td = cv::v_max(v_p, v_q);

        v_n += (v_tn*v_tn)/v_td;
    }

    cv::v_store_aligned(buffer, v_n);

    for(int i=0;i<4;++i)
        n += buffer[i];

#elif CV_ENABLE_UNROLLED
    for(;l<dims-4;l+=4, a+=4, b+=4)
    {
        float a0 = a[0];
        float a1 = a[1];

        float b0 = b[0];
        float b1 = b[1];

        float tmp_n0 = a0 - b0;
        float tmp_n1 = a1 - b1;

        float tmp_d0 = std::max(a0, b0);
        float tmp_d1 = std::max(a1, b1);

        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;


        a0 = a[2];
        a1 = a[3];

        b0 = b[2];
        b1 = b[3];

        tmp_n0 = a0 - b0;
        tmp_n1 = a1 - b1;

        tmp_d0 = std::max(a0, b0);
        tmp_d1 = std::max(a1, b1);

        tmp_n0 *= tmp_n0;
        tmp_n1 *= tmp_n1;

        n += tmp_n0 / tmp_d0;
        n += tmp_n1 / tmp_d1;
    }
#endif

    for(; l<dims;l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float tmp_n = p - q;

        float tmp_d = std::max(p, q);

        tmp_n*=tmp_n;

        n += tmp_n / tmp_d;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return n;

}

float normMaxSymmetric_ChiSqr(const float*  a,const float*  b,int dims)
{

    float t1(0.f);
    float t2(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[8];
#endif

#if CV_AVX

    float CV_DECL_ALIGNED(0x20) buffer[16];

    __m256 _t1 = _mm256_setzero_ps();
    __m256 _t2 = _t1;

    for(;l<=dims-8;l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 num = _mm256_sub_ps(_p,_q);
        num = _mm256_mul_ps(num, num);

        _t1 = _mm256_add_ps(_t1,_mm256_div_ps(num,_p));
        _t2 = _mm256_add_ps(_t2,_mm256_div_ps(num,_q));
    }

    _mm256_stream_ps(buffer,_t1);
    _mm256_stream_ps(buffer+8,_t2);

    for(int i=0, j=8;i<8;i++, j++)
    {
        t1+=buffer[i];
        t2+=buffer[j];
    }

#endif

//#elif CV_SSE

//    float CV_DECL_ALIGNED(0x20) buffer[8];

//    __m128 _t1 = _mm_setzero_ps();
//    __m128 _t2 = _t1;

//    for(;l<dims-4;l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 num = _mm_sub_ps(_p,_q);
//        num = _mm_mul_ps(num, num);

//        _t1 = _mm_add_ps(_t1,_mm_div_ps(num,_p));
//        _t2 = _mm_add_ps(_t2,_mm_div_ps(num,_q));
//    }

//    _mm_stream_ps(buffer,_t1);
//    _mm_stream_ps(buffer,_t2);

//    for(int i=0, j=4;i<4;i++, j++)
//    {
//        t1+=buffer[i];
//        t2+=buffer[j];
//    }

//#endif

#if CV_SIMD128

    cv::v_float32x4 v_t1 = cv::v_setzero_f32();
    cv::v_float32x4 v_t2 = v_t1;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_num = v_p-v_q;
        v_num*=v_num;

        v_t1 += v_num/v_p;
        v_t2 += v_num/v_q;
    }

    cv::v_store_aligned(buffer, v_t1);
    cv::v_store_aligned(buffer+4, v_t2);

    for(int i=0, j=4;i<4;i++, j++)
    {
        t1+=buffer[i];
        t2+=buffer[j];
    }

#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float num0 = p0-q0;
        float num1 = p1-q1;

        num0 *= num0;
        num1 *= num1;

        t1+=num0/p0;
        t1+=num1/p1;

        t2+=num0/q0;
        t2+=num1/q1;

        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        num0 = p0-q0;
        num1 = p1-q1;

        num0 *= num0;
        num1 *= num1;

        t1+=num0/p0;
        t1+=num1/p1;

        t2+=num0/q0;
        t2+=num1/q1;
    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float p = *a;
        float q = *b;

        float num = p-q;
        num *= num;

        t1+=num/p;
        t2+=num/q;
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return std::max(t1,t2);
}

float normMinSymmetric_ChiSqr(const float*  a,const float* b,int dims)
{

    float t1(0.f);
    float t2(0.f);

    int l(0);

#if CV_AVX
    float CV_DECL_ALIGNED(0x20) buffer[16];
#elif CV_SIMD128
    float CV_DECL_ALIGNED(0x10) buffer[8];
#endif

#if CV_AVX

    float CV_DECL_ALIGNED(0x20) buffer[16];

    __m256 _t1 = _mm256_setzero_ps();
    __m256 _t2 = _t1;

    for(;l<=dims-8;l+=8, a+=8, b+=8)
    {
        __m256 _p = _mm256_loadu_ps(a);
        __m256 _q = _mm256_loadu_ps(b);

        __m256 num = _mm256_sub_ps(_p,_q);
        num = _mm256_mul_ps(num, num);

        _t1 = _mm256_add_ps(_t1,_mm256_div_ps(num,_p));
        _t2 = _mm256_add_ps(_t2,_mm256_div_ps(num,_q));
    }

    _mm256_stream_ps(buffer,_t1);
    _mm256_stream_ps(buffer+8,_t2);

    for(int i=0;i<8;++i)
    {
        t1+=buffer[i];
        t2+=buffer[i+8];
    }
#endif

//#elif CV_SSE


//    __m128 _t1 = _mm_setzero_ps();
//    __m128 _t2 = _t1;

//    for(;l<dims-4;l+=4, a+=4, b+=4)
//    {
//        __m128 _p = _mm_loadu_ps(a);
//        __m128 _q = _mm_loadu_ps(b);

//        __m128 num = _mm_sub_ps(_p,_q);
//        num = _mm_mul_ps(num, num);

//        _t1 = _mm_add_ps(_t1,_mm_div_ps(num,_p));
//        _t2 = _mm_add_ps(_t2,_mm_div_ps(num,_q));
//    }

//    _mm_stream_ps(buffer,_t1);
//    _mm_stream_ps(buffer,_t2);

//    for(int i=0, j=4;i<4; i++, j++)
//    {
//        t1+=buffer[i];
//        t2+=buffer[j];
//    }

//#endif

#if CV_SIMD128


    cv::v_float32x4 v_t1 = cv::v_setzero_f32();
    cv::v_float32x4 v_t2 = v_t1;

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {

        cv::v_float32x4 v_p = cv::v_load(a);
        cv::v_float32x4 v_q = cv::v_load(b);

        cv::v_float32x4 v_num = v_p-v_q;
        v_num*=v_num;

        v_t1 += v_num/v_p;
        v_t2 += v_num/v_q;
    }

    cv::v_store_aligned(buffer, v_t1);
    cv::v_store_aligned(buffer+4, v_t2);

    for(int i=0, j=4;i<4;i++, j++)
    {
        t1+=buffer[i];
        t2+=buffer[j];
    }


#elif CV_ENABLE_UNROLLED

    for(;l<=dims-4; l+=4, a+=4, b+=4)
    {
        float p0 = a[0];
        float p1 = a[1];

        float q0 = b[0];
        float q1 = b[1];

        float num0 = p0-q0;
        float num1 = p1-q1;

        num0 *= num0;
        num1 *= num1;

        t1+=num0/p0;
        t1+=num1/p1;

        t2+=num0/q0;
        t2+=num1/q1;

        //

        p0 = a[2];
        p1 = a[3];

        q0 = b[2];
        q1 = b[3];

        num0 = p0-q0;
        num1 = p1-q1;

        num0 *= num0;
        num1 *= num1;

        t1+=num0/p0;
        t1+=num1/p1;

        t2+=num0/q0;
        t2+=num1/q1;
    }

#endif

    for(;l<dims; l++, a++, b++)
    {
        float num = a[0]-b[0];
        num *= num;

        t1+=num/a[0];
        t2+=num/b[0];
    }

#if CV_SSE2
   _mm_mfence();
#endif

    return std::min(t1,t2);
}

} // hal

} // clustering

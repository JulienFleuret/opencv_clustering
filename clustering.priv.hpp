#ifndef CLUSTERING_PRIV_HPP
#define CLUSTERING_PRIV_HPP

#include <opencv2/core.hpp>

namespace clustering
{

void generateRandomCenter(const std::vector<cv::Vec2f>& box, float* center, cv::RNG& rng);

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

    inline KMeansPPDistanceComputer( float *_tdist2,
                              const float *_data,
                              const float *_dist,
                              int _dims,
                              size_t _step,
                              size_t _stepci)
        : tdist2(_tdist2),
          data(_data),
          dist(_dist),
          dims(_dims),
          step(_step),
          stepci(_stepci),
          fun(nullptr)
    {}


    virtual ~KMeansPPDistanceComputer() = default;

    virtual void operator()( const cv::Range& range ) const;

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
void generateCentersPP(const cv::Mat& _data,
                       cv::Mat& _out_centers,
                       int K,
                       cv::RNG& rng,
                       int trials,
                       float(*fun)(const float*, const float*, int));


void apply_scale(const float* src, const float& scale, float* dst, const int& dims);
void update_new_old_centres(const float* samples, float* nc, float* oc, const int& dims);

}

#endif // CLUSTERING_PRIV_HPP

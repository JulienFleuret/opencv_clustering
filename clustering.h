#ifndef CLUSTERING_H
#define CLUSTERING_H

//Copyright 2018, University Laval, CVSL-MIVIM

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: Julien FLEURET, julien.fleuret.1@ulaval.ca

// This file does contains the declaration of several clustering function based on OpenCV's library.

#include <opencv2/core.hpp>
#include <functional>

#include <opencv2/cudev.hpp>

namespace clustering
{

class DistanceBody
{
public:

//    typedef float(*function_type)(const float*, const float*, int);
    typedef std::function<float(const float*, const float*, int)> function_type;
    typedef float(*function_ptr)(const float*, const float*, int);

    inline DistanceBody():
        fun(nullptr)
    {}

    inline DistanceBody(std::nullptr_t):
        fun(nullptr)
    {}

    inline DistanceBody(const function_type _fun):
        fun(_fun)
    {}

    inline DistanceBody(const DistanceBody& obj):
        fun(obj.fun)
    {}

    inline DistanceBody(DistanceBody&& obj):
        fun(std::move(obj.fun))
    {}

    virtual ~DistanceBody() = default;

    inline DistanceBody& operator=(const DistanceBody& obj)
    {
        if(std::addressof(obj) != this)
            this->fun = obj.fun;

        return (*this);
    }

    inline DistanceBody& operator=(DistanceBody&& obj)
    {
        if(std::addressof(obj) != this)
            this->fun = std::move(obj.fun);

        return (*this);
    }

    virtual float operator()(const float* a, const float* b, int dims)const
    {
        return this->fun(a,b,dims);
    }

    virtual float operator ()(const float& a, const float& b)const
    {
        return this->fun(&a, &b, 1);
    }

    template<class T=function_ptr>
    inline const T* target()const
    {
        return this->fun.target<T>();
    }

    template<class T=function_ptr>
    inline T* target()
    {
        return this->fun.target<T>();
    }

    inline operator bool()const
    {
        return (bool)this->fun;
    }

protected:

    function_type fun;

};

/** @brief Finds centers of clusters and groups input samples around the clusters.

The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters
and groups the input samples around the clusters. As an output, \f$\texttt{labels}_i\f$ contains a
0-based cluster index for the sample stored in the \f$i^{th}\f$ row of the samples matrix.

@param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
Examples of this array can be:
-   Mat points(count, 2, CV_32F);
-   Mat points(count, 1, CV_32FC2);
-   Mat points(1, count, CV_32FC2);
-   std::vector\<cv::Point2f\> points(sampleCount);
@param K Number of clusters to split the set by.
@param bestLabels Input/output integer array that stores the cluster indices for every sample.
@param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
@param attempts Flag to specify the number of times the algorithm is executed using different
initial labellings. The algorithm returns the labels that yield the best compactness (see the last
function parameter).
@param flags Flag that can take values of cv::KmeansFlags
@param centers Output matrix of the cluster centers, one row per each cluster center.
@param fun metric to use for clustering, if null or nullptr, the norm L2 is used.
@return The function returns the compactness measure that is computed as
\f[\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2\f]
after every attempt. The best (minimum) value is chosen and the corresponding labels and the
compactness value are returned by the function. Basically, you can use only the core of the
function, set the number of attempts to 1, initialize labels each time using a custom algorithm,
pass them with the ( flags = KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best
(most-compact) clustering.
*/
double kmeans( cv::InputArray _data,
               int K,
               cv::InputOutputArray _bestLabels,
               cv::TermCriteria criteria,
               int attempts,
               int flags,
               cv::OutputArray _centers,
               const DistanceBody& norm = nullptr
//               const std::function<float(const float*, const float*, int)>& fun = nullptr
               );

/** @brief Finds centers of clusters and groups input samples around the clusters.

The function mini_batch_kmeans implements the mini batch k-means algorithm that finds the centers of cluster_count clusters
and groups the input samples around the clusters. As an output, \f$\texttt{labels}_i\f$ contains a
0-based cluster index for the sample stored in the \f$i^{th}\f$ row of the samples matrix.

@param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
Examples of this array can be:
-   Mat points(count, 2, CV_32F);
-   Mat points(count, 1, CV_32FC2);
-   Mat points(1, count, CV_32FC2);
-   std::vector\<cv::Point2f\> points(sampleCount);
@param K Number of clusters to split the set by.
@param B Batch size.
@param bestLabels Input/output integer array that stores the cluster indices for every sample.
@param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
@param attempts Flag to specify the number of times the algorithm is executed using different
initial labellings. The algorithm returns the labels that yield the best compactness (see the last
function parameter).
@param flags Flag that can take values of cv::KmeansFlags
@param centers Output matrix of the cluster centers, one row per each cluster center.
@param fun metric to use for clustering, if null or nullptr, the norm L2 is used.
@return The function returns the compactness measure that is computed as
\f[\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2\f]
after every attempt. The best (minimum) value is chosen and the corresponding labels and the
compactness value are returned by the function. Basically, you can use only the core of the
function, set the number of attempts to 1, initialize labels each time using a custom algorithm,
pass them with the ( flags = KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best
(most-compact) clustering.
*/
double mini_batch_kmeans(cv::InputArray _data,
                          int K,
                          int B,
                          cv::InputOutputArray _bestLabels,
                          cv::TermCriteria criteria,
                          int attempts,
                          int flags,
                          cv::OutputArray _centers,
//                          const std::function<float(const float*, const float*, int)>& fun = nullptr
                          const DistanceBody& norm = nullptr
                          );


double dbscan(cv::InputArray _data,
              const float& neighbourhoud_distance,
              const int& minPts,
              cv::OutputArray _labels,
              cv::OutputArray _centres = cv::noArray());



namespace cuda
{

/** @brief Finds centers of clusters and groups input samples around the clusters.

The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters
and groups the input samples around the clusters. As an output, \f$\texttt{labels}_i\f$ contains a
0-based cluster index for the sample stored in the \f$i^{th}\f$ row of the samples matrix.
This implementation use cuda

@param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
Examples of this array can be:
-   Mat points(count, 2, CV_32F);
-   Mat points(count, 1, CV_32FC2);
-   Mat points(1, count, CV_32FC2);
-   std::vector\<cv::Point2f\> points(sampleCount);
@param K Number of clusters to split the set by.
@param bestLabels Input/output integer array that stores the cluster indices for every sample.
@param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
@param attempts Flag to specify the number of times the algorithm is executed using different
initial labellings. The algorithm returns the labels that yield the best compactness (see the last
function parameter).
@param flags Flag that can take values of cv::KmeansFlags
@param centers Output matrix of the cluster centers, one row per each cluster center.
@param fun metric to use for clustering, if null or nullptr, the norm L2 is used.
@return The function returns the compactness measure that is computed as
\f[\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2\f]
after every attempt. The best (minimum) value is chosen and the corresponding labels and the
compactness value are returned by the function. Basically, you can use only the core of the
function, set the number of attempts to 1, initialize labels each time using a custom algorithm,
pass them with the ( flags = KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best
(most-compact) clustering.
*/
double kmeans( cv::InputArray _data,
               int K,
               cv::InputOutputArray _bestLabels,
               cv::TermCriteria criteria,
               int attempts,
               int flags,
               cv::OutputArray _centers,
               const std::function<float(const float*, const float*, int)>& fun = nullptr
               );


}

}

#endif // CLUSTERING_H

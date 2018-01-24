#ifndef METRICS_H
#define METRICS_H

#include "clustering.h"

namespace clustering
{
namespace hal
{
float normL2(const float*  a, const float*  b, int dims);

float normLp(const float*  a,const float*  b,int dims,const float& p);

float normSorensen(const float*  a,const float*  b,int dims);

float normCzekanowski(const float*  a,const float*  b,int dims);

float normGower(const float*  a,const float*  b,int dims);

float normSoergel(const float*  a,const float*  b,int dims);

float normKulczynski(const float*  a,const float*  b,int dims);

float normCanberra(const float*  a,const float*  b,int dims);

float normLorentzian(const float*  a,const float*  b,int dims);

float normIntersection(const float*  a,const float*  b,int dims);

float simIntersection(const float*  a,const float*  b,int dims);

float normWaveHedges(const float*  a,const float*  b,int dims);

float simCzekanowski(const float*  a,const float*  b,int dims);

float simMotyka(const float*  a,const float*  b,int dims);

float normMotyka(const float*  a,const float*  b,int dims);

float simKulczynski(const float*  a,const float*  b,int dims);

float simRuzicka(const float*  a,const float*  b,int dims);

float normTanimoto(const float*  a,const float*  b,int dims);

float simInnerProduct(const float*  a,const float*  b,int dims);

float simHarmonicMean(const float*  a,const float*  b,int dims);

float simCosine(const float*  a,const float*  b,int dims);

float simKumarHassebrook(const float*  a,const float*  b,int dims);

float simJaccard(const float*  a,const float*  b,int dims);

float normJaccard(const float*  a,const float*  b,int dims);

float simDice(const float*  a,const float*  b,int dims);

float normDice(const float*  a,const float*  b,int dims);

float simFidelity(const float*  a,const float*  b,int dims);

float normBhattacharyya(const float*  a,const float*  b,int dims);

float normHellinger(const float*  a,const float*  b,int dims);

float normMatusita(const float*  a,const float*  b,int dims);

float simSquaredChord(const float*  a,const float*  b,int dims);

float normSquaredChord(const float*  a,const float*  b,int dims);

float normPearson_ChiSqr(const float*  a,const float*  b,int dims);

float normNeyman_ChiSqr(const float*  a,const float*  b,int dims);

float normSquared_ChiSqr(const float*  a,const float*  b,int dims);

float normProbabilisticSymmetric_ChiSqr(const float*  a,const float*  b,int dims);

float normDivergence(const float*  a,const float*  b,int dims);

float normClark(const float*  a,const float*  b,int dims);

float normAdditiveSymmetric_ChiSqr(const float*  a,const float*  b,int dims);

float normEntKullbackLeibler(const float*  a,const float*  b,int dims);

float normEntJeffreys(const float*  a,const float*  b,int dims);

float normEntKDivergence(const float*  a,const float*  b,int dims);

float normEntTopsoe(const float*  a,const float*  b,int dims);

float normEntJensen_Shannon(const float*  a,const float*  b,int dims);

float normEntJensenDifference(const float*  a,const float*  b,int dims);

float normTaneja(const float*  a,const float*  b,int dims);

float normKumarJohnson(const float*  a,const float*  b,int dims);

//float normAvg(const float*  a,const float*  b,int dims);

float normVicisWaveHedges(const float*  a,const float*  b,int dims);

float normVicisSymmetric_ChiSqr(const float*  a,const float*  b,int dims);

float normVicisSymmetric_ChiSqr_2(const float*  a,const float*  b,int dims);

float normVicisSymmetric_ChiSqr_3(const float*  a,const float*  b,int dims);

float normMaxSymmetric_ChiSqr(const float*  a,const float*  b,int dims);

float normMinSymmetric_ChiSqr(const float*  a,const float* b,int dims);



float normMahalanobis(const float* a,const float* b,int dims);

float normBhattacharyyaMv(const float* a,const float* b,int dims);

} // hal


#define INSTANCE_DISTANCE_CLASS(name)\
class name : public DistanceBody\
{\
    public: \
    \
    typedef DistanceBody MyBase;\
    \
    inline name():\
            MyBase(hal::name)\
    {} \
    \
    inline name(const name& obj): \
            MyBase(obj)\
    {}\
    \
    inline name(name&& obj):\
            MyBase(std::move(obj))\
    {}\
    \
    virtual ~name() = default;\
    \
};

class normLp : public DistanceBody
{
private:

    int p;

public:

    typedef DistanceBody MyBase;

    inline normLp(const int& _p):
        MyBase(),
        p(_p)
    {}

    inline normLp(const normLp& obj):
        MyBase(obj),
        p(obj.p)
    {}

    inline normLp(normLp&& obj):
        MyBase(std::move(obj)),
        p(obj.p)
    {}

    virtual ~normLp() = default;

    inline normLp& operator=(const int& _p)
    {
        this->p = _p;

        return (*this);
    }

    inline normLp& operator =(const normLp& obj)
    {
        if(std::addressof(obj) != this)
        {
            this->fun = obj.fun;
            this->p = obj.p;
        }

        return (*this);
    }

    inline normLp& operator =(normLp&& obj)
    {
        if(std::addressof(obj) != this)
        {
            this->fun = obj.fun;
            this->p = obj.p;
        }

        return (*this);
    }

};

INSTANCE_DISTANCE_CLASS(normL2)
//INSTANCE_DISTANCE_CLASS(normLp)
INSTANCE_DISTANCE_CLASS(normSorensen)
INSTANCE_DISTANCE_CLASS(normCzekanowski)
INSTANCE_DISTANCE_CLASS(normGower)
INSTANCE_DISTANCE_CLASS(normSoergel)
INSTANCE_DISTANCE_CLASS(normKulczynski)
INSTANCE_DISTANCE_CLASS(normCanberra)
INSTANCE_DISTANCE_CLASS(normLorentzian)
INSTANCE_DISTANCE_CLASS(normIntersection)
INSTANCE_DISTANCE_CLASS(simIntersection)
INSTANCE_DISTANCE_CLASS(normWaveHedges)
INSTANCE_DISTANCE_CLASS(simCzekanowski)
INSTANCE_DISTANCE_CLASS(simMotyka)
INSTANCE_DISTANCE_CLASS(normMotyka)
INSTANCE_DISTANCE_CLASS(simKulczynski)
INSTANCE_DISTANCE_CLASS(simRuzicka)
INSTANCE_DISTANCE_CLASS(normTanimoto)
INSTANCE_DISTANCE_CLASS(simInnerProduct)
INSTANCE_DISTANCE_CLASS(simHarmonicMean)
INSTANCE_DISTANCE_CLASS(simCosine)
INSTANCE_DISTANCE_CLASS(simKumarHassebrook)
INSTANCE_DISTANCE_CLASS(simJaccard)
INSTANCE_DISTANCE_CLASS(normJaccard)
INSTANCE_DISTANCE_CLASS(simDice)
INSTANCE_DISTANCE_CLASS(normDice)
INSTANCE_DISTANCE_CLASS(simFidelity)
INSTANCE_DISTANCE_CLASS(normBhattacharyya)
INSTANCE_DISTANCE_CLASS(normHellinger)
INSTANCE_DISTANCE_CLASS(normMatusita)
INSTANCE_DISTANCE_CLASS(simSquaredChord)
INSTANCE_DISTANCE_CLASS(normSquaredChord)
INSTANCE_DISTANCE_CLASS(normPearson_ChiSqr)
INSTANCE_DISTANCE_CLASS(normNeyman_ChiSqr)
INSTANCE_DISTANCE_CLASS(normSquared_ChiSqr)
INSTANCE_DISTANCE_CLASS(normProbabilisticSymmetric_ChiSqr)
INSTANCE_DISTANCE_CLASS(normDivergence)
INSTANCE_DISTANCE_CLASS(normClark)
INSTANCE_DISTANCE_CLASS(normAdditiveSymmetric_ChiSqr)
INSTANCE_DISTANCE_CLASS(normEntKullbackLeibler)
INSTANCE_DISTANCE_CLASS(normEntJeffreys)
INSTANCE_DISTANCE_CLASS(normEntKDivergence)
INSTANCE_DISTANCE_CLASS(normEntTopsoe)
INSTANCE_DISTANCE_CLASS(normEntJensen_Shannon)
INSTANCE_DISTANCE_CLASS(normEntJensenDifference)
INSTANCE_DISTANCE_CLASS(normTaneja)
INSTANCE_DISTANCE_CLASS(normKumarJohnson)
INSTANCE_DISTANCE_CLASS(normVicisWaveHedges)
INSTANCE_DISTANCE_CLASS(normVicisSymmetric_ChiSqr)
INSTANCE_DISTANCE_CLASS(normVicisSymmetric_ChiSqr_2)
INSTANCE_DISTANCE_CLASS(normVicisSymmetric_ChiSqr_3)
INSTANCE_DISTANCE_CLASS(normMaxSymmetric_ChiSqr)
INSTANCE_DISTANCE_CLASS(normMinSymmetric_ChiSqr)
INSTANCE_DISTANCE_CLASS(normMahalanobis)
INSTANCE_DISTANCE_CLASS(normBhattacharyyaMv)


}

#endif // METRICS_H

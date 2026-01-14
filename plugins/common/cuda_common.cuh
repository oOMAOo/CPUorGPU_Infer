#include <iostream>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

static __forceinline__ __device__
__half operator*(const __half& a, const int& b)
{
    return a * __int2half_rn(b);
}

static __forceinline__ __device__
__half operator*(const float& a, const half& b)
{
    return __float2half(a) * b;
}

static __forceinline__ __device__
__half operator+=(const float& a, const half& b)
{
    return __float2half(a) += b;
}

static __forceinline__ __device__
__half operator/(const __half& a, const int& b)
{
    return a / __int2half_rn(b);
}

static __forceinline__ __device__
__half operator+(const __half& a, const float& b)
{
    return a + __float2half(b);
}

static __forceinline__ __device__
__half operator-(const __half& a, const int& b)
{
    return a - __int2half_rn(b);
}

static __forceinline__ __device__
__half operator-(const int& a, const __half& b)
{
    return __int2half_rn(a) - b;
}

static __forceinline__ __device__
__half operator+=(const __half& a, const __half& b)
{
    return a + b;
}

static __forceinline__ __device__
__half min(const __half& a, const half& b)
{
    return __float2half(min(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half max(const __half& a, const half& b)
{
    return __float2half(max(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half fabs(const __half& a)
{
    //TODO return __habs(a); what happened.
    return __float2half(fabs(__half2float(a)));
}

static __forceinline__ __device__
__half floor(const __half& a)
{
    return __float2half(floorf(__half2float(a)));
}

static __forceinline__ __device__
__half round(const __half& a)
{
    return __float2half(roundf(__half2float(a)));
}

static __forceinline__ __device__
__half fmod(const __half& a, const __half& b)
{
  return __float2half(fmodf(__half2float(a), __half2float(b)));
}

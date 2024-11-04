#pragma once

struct functor_mul
{
    __device__
    storage operator()(const storage& a, const storage& b)
    {
        auto result = fd_q::mul(a, b);
        return result;
    }
};

typedef thrust::tuple<storage, storage> Tuple2;
struct functor_mul_2
{
    __device__
    storage operator()(const Tuple2& t)
    {
        storage a, b;
        thrust::tie(a, b) = t;

        storage result = fd_q::mul(a, b);
        return result;
    }
};

struct functor_add
{
    __device__
    fd_q::storage operator()(const storage& a, const storage& b)
    {
        storage result = fd_q::add(a, b);
        return result;
    }
};
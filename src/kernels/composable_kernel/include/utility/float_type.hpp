#ifndef CK_FLOAT_TYPE_AMD_HPP
#define CK_FLOAT_TYPE_AMD_HPP

namespace ck {

// For some reason, HIP compiler need this definition to generate optimal ISA
// float
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float4_t __attribute__((ext_vector_type(4)));
typedef float float8_t __attribute__((ext_vector_type(8)));
typedef float float16_t __attribute__((ext_vector_type(16)));
typedef float float32_t __attribute__((ext_vector_type(32)));

// float16
typedef _Float16 half_t;
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));

// bfloat16
typedef ushort ushort2_t __attribute__((ext_vector_type(2)));
typedef ushort ushort4_t __attribute__((ext_vector_type(4)));
typedef ushort ushort8_t __attribute__((ext_vector_type(8)));
typedef ushort ushort16_t __attribute__((ext_vector_type(16)));

struct c_vec32_4_t
{
    union VecType
    {
        struct
        {
            float32_t x;
            float32_t y;
            float32_t z;
            float32_t w;
        } s;
        float n[128];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        c.s.y = 0;
        c.s.z = 0;
        c.s.w = 0;
        return c;
    }
};

struct c_vec32_2_t
{
    union VecType
    {
        struct
        {
            float32_t x;
            float32_t y;
        } s;
        float n[64];
    } l;

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        c.s.y = 0;
        return c;
    }
};

struct c_vec32_2_2_t
{
    union VecType
    {
        struct
        {
            c_vec32_2_t x;
            c_vec32_2_t y;
        } s;
        float n[128];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x.l.s.x = 0;
        c.s.x.l.s.y = 0;
        c.s.y.l.s.x = 0;
        c.s.y.l.s.y = 0;
        return c;
    }
};

struct c_vec32_1_t
{
    union VecType
    {
        struct
        {
            float32_t x;
        } s;
        float n[32];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        return c;
    }
};

struct c_vec16_1_t
{
    union VecType
    {
        struct
        {
            float16_t x;
        } s;
        float n[16];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        return c;
    }
};

struct c_vec4_2_t
{
    union VecType
    {
        struct
        {
            float4_t x;
            float4_t y;
        } s;
        float n[8];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        c.s.y = 0;
        return c;
    }
};

struct c_vec4_1_t
{
    union VecType
    {
        struct
        {
            float4_t x;
        } s;
        float n[4];
    };

    __host__ __device__ static VecType CreateVecZero()
    {
        VecType c;
        c.s.x = 0;
        return c;
    }
};

template <class T, index_t N>
struct vector_type
{
    //  typedef T MemoryType __attribute__((ext_vector_type(N)));
};

template <>
struct vector_type<float, 1>
{
    using MemoryType = float;
};

template <>
struct vector_type<float, 2>
{
    using MemoryType = float2_t;
};

template <>
struct vector_type<float, 4>
{
    using MemoryType = float4_t;
};

template <>
struct vector_type<float, 8>
{
    using MemoryType = float8_t;
};

template <>
struct vector_type<float, 16>
{
    using MemoryType = float16_t;
};

template <>
struct vector_type<half_t, 1>
{
    using MemoryType = half_t;
};

template <>
struct vector_type<half_t, 2>
{
    using MemoryType = half2_t;
};

template <>
struct vector_type<half_t, 4>
{
    using MemoryType = half4_t;
};

template <>
struct vector_type<half_t, 8>
{
    using MemoryType = half8_t;
};

template <>
struct vector_type<half_t, 16>
{
    using MemoryType = half16_t;
};

template <>
struct vector_type<ushort, 1>
{
    using MemoryType = ushort;
};

template <>
struct vector_type<ushort, 2>
{
    using MemoryType = ushort2_t;
};

template <>
struct vector_type<ushort, 4>
{
    using MemoryType = ushort4_t;
};

template <>
struct vector_type<ushort, 8>
{
    using MemoryType = ushort8_t;
};

template <>
struct vector_type<ushort, 16>
{
    using MemoryType = ushort16_t;
};

// data type conversion
template <typename T>
struct type_convert
{
    template <typename X>
    __device__ T operator()(X x) const
    {
        return static_cast<T>(x);
    }
};

template <>
template <>
__device__ float type_convert<float>::operator()<ushort>(ushort x) const
{
    return bfloat16_to_float(x);
}

template <>
template <>
__device__ ushort type_convert<ushort>::operator()<float>(float x) const
{
    return float_to_bfloat16(x);
}

template <typename T>
struct inner_product_with_conversion
{
    static constexpr auto convert = type_convert<T>();

    __device__ T operator()(float4_t a, float4_t b) const
    {
        const float* p_a_float = reinterpret_cast<const float*>(&a);
        const float* p_b_float = reinterpret_cast<const float*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_float[v]) * convert(p_b_float[v]);
        }

        return acc;
    }

    __device__ T operator()(float2_t a, float2_t b) const
    {
        const float* p_a_float = reinterpret_cast<const float*>(&a);
        const float* p_b_float = reinterpret_cast<const float*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_float[v]) * convert(p_b_float[v]);
        }

        return acc;
    }

    __device__ T operator()(float a, float b) const { return convert(a) * convert(b); }

    __device__ T operator()(half2_t a, half2_t b) const
    {
        const half_t* p_a_half = reinterpret_cast<const half_t*>(&a);
        const half_t* p_b_half = reinterpret_cast<const half_t*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }

        return acc;
    }

    __device__ T operator()(half4_t a, half4_t b) const
    {
        const half_t* p_a_half = reinterpret_cast<const half_t*>(&a);
        const half_t* p_b_half = reinterpret_cast<const half_t*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }
        return acc;
    }

    __device__ T operator()(half8_t a, half8_t b) const
    {
        const half_t* p_a_half = reinterpret_cast<const half_t*>(&a);
        const half_t* p_b_half = reinterpret_cast<const half_t*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 8; ++v)
        {
            acc += convert(p_a_half[v]) * convert(p_b_half[v]);
        }
        return acc;
    }

    __device__ T operator()(ushort2_t a, ushort2_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 2; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }

        return acc;
    }

    __device__ T operator()(ushort4_t a, ushort4_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 4; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }
        return acc;
    }

    __device__ T operator()(ushort8_t a, ushort8_t b) const
    {
        const ushort* p_a_bfloat16 = reinterpret_cast<const ushort*>(&a);
        const ushort* p_b_bfloat16 = reinterpret_cast<const ushort*>(&b);

        T acc = 0;
        for(index_t v = 0; v < 8; ++v)
        {
            acc += convert(p_a_bfloat16[v]) * convert(p_b_bfloat16[v]);
        }
        return acc;
    }
};

} // namespace ck
#endif

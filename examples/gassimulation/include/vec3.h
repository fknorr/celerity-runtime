#ifndef GASSIMULATION_VEC3_H
#define GASSIMULATION_VEC3_H
#define USERFUNC __host__ __device__
template<typename T>
struct vec3 {

    T x, y, z;

	USERFUNC void operator+=(const vec3<T>& other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }

	USERFUNC friend vec3<T> operator+(vec3<T> v, const vec3<T>& other) {
        v += other;
        return v;
    }

	USERFUNC void operator*=(const T& val) {
        x *= val;
        y *= val;
        z *= val;
    }

	USERFUNC friend vec3<T> operator*(vec3<T> v, const T& val) {
        v *= val;
        return v;
    }

	USERFUNC friend T operator*(const vec3<T>& v1, const vec3<T>& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
};

#endif //GASSIMULATION_VEC3_H

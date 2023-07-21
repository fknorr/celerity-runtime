#ifndef GASSIMULATION_ARRAY_H
#define GASSIMULATION_ARRAY_H

template<typename T, unsigned int N>
struct array {
    T data[N];

    USERFUNC T operator[](size_t n) const {
        return data[n];
    }

    USERFUNC T& operator[](size_t n) {
        return data[n];
    }
};

#endif //GASSIMULATION_ARRAY_H

#include <vector>
#include <random>
#include <mutex>
#include <atomic>
#include <string>
#include <unordered_map>
#include <memory>
#include <random>
#include <limits>

#include <filesystem>

namespace fs = std::filesystem;


using Float = float;
constexpr Float Pi = 3.1415926535f;
constexpr Float Pi2 = Pi * 0.5f;
constexpr Float Pi4 = Pi * 0.25f;
constexpr Float InvPi = 1.0f / Pi;
constexpr Float Inv4Pi = 1.0f / (4.0f * Pi);
constexpr Float MaxFloat = std::numeric_limits<Float>::max();
constexpr Float MinFloat = std::numeric_limits<Float>::min();
constexpr Float MachineEpsilon = std::numeric_limits<Float>::epsilon();
Float RayBias = 0.05f;


constexpr Float gamma(int n) {
    return n * MachineEpsilon / (1 - n * MachineEpsilon);
}

inline Float RadiansToDegrees(Float x) {
    return x * InvPi * 180.0f;
}

inline Float DegreesToRadians(Float x) {
    return x / 180.0f * Pi;
}

// https://en.wikipedia.org/wiki/Permuted_congruential_generator
class Rng {
    uint64_t state;
    static uint64_t const multiplier = 6364136223846793005u;
    static uint64_t const increment = 1442695040888963407u;

    static uint32_t rotr32(uint32_t x, unsigned r) {
        return x >> r | x << (-r & 31);
    }

    uint32_t pcg32() {
        uint64_t x = state;
        auto count = (unsigned) (x >> 59ULL);        // 59 = 64 - 5

        state = x * multiplier + increment;
        x ^= x >> 18ULL;                                // 18 = (64 - 27)/2
        return rotr32((uint32_t) (x >> 27ULL), count);    // 27 = 32 - 5
    }

public:
    explicit Rng(uint64_t state = 0) : state(state + increment) {
        pcg32();
    }

    uint32_t uniformUint32() {
        return pcg32();
    }

    float uniformFloat() {
        return float(uniformUint32()) / std::numeric_limits<uint32_t>::max();
    }

};

template<class T, size_t N>
struct VecBase {
    static constexpr size_t _N = N;
    T _v[_N];

    VecBase() {
        for (auto &i:_v) {
            i = T();
        }
    }
};

template<class T>
struct VecBase<T, 2> {
    static constexpr size_t _N = 2;
    union {
        T _v[_N];
        struct {
            T x, y;
        };
    };

    VecBase() : x(T()), y(T()) {}

    VecBase(const T &x, const T &y) : x(x), y(y) {}
};

template<class T>
struct VecBase<T, 3> {
    static constexpr size_t _N = 3;
    union {
        T _v[_N];
        struct {
            T x, y, z;
        };
    };

    VecBase() : x(T()), y(T()), z(T()) {}

    VecBase(const T &x, const T &y, const T &z) : x(x), y(y), z(z) {}
};

template<class T>
struct VecBase<T, 4> {
    static constexpr size_t _N = 4;
    union {
        T _v[_N];
        struct {
            T x, y, z, w;
        };
    };

    VecBase() : x(T()), y(T()), z(T()), w(T()) {}

    VecBase(const T &x, const T &y, const T &z, const T &w) : x(x), y(y), z(z), w(w) {}
};

template<class T, size_t N>
struct Vec : VecBase<T, N> {
    using VecBase<T, N>::VecBase;
    static constexpr size_t _N = VecBase<T, N>::_N;

    Vec &operator+=(const Vec &rhs) {
        for (size_t i = 0; i < _N; i++) {
            this->_v[i] += rhs._v[i];
        }
        return *this;
    }

    Vec &operator-=(const Vec &rhs) {
        for (size_t i = 0; i < _N; i++) {
            this->_v[i] -= rhs._v[i];
        }
        return *this;
    }

    Vec &operator/=(const Vec &rhs) {
        for (size_t i = 0; i < _N; i++) {
            this->_v[i] /= rhs._v[i];
        }
        return *this;
    }

    Vec &operator*=(const Vec &rhs) {
        for (size_t i = 0; i < _N; i++) {
            this->_v[i] *= rhs._v[i];
        }
        return *this;
    }

    Vec &operator/=(const T &rhs) {
        for (size_t i = 0; i < _N; i++) {
            this->_v[i] /= rhs;
        }
        return *this;
    }

    Vec &operator*=(const T &rhs) {
        for (size_t i = 0; i < _N; i++) {
            this->_v[i] *= rhs;
        }
        return *this;
    }

    Vec operator+(const Vec &rhs) const {
        Vec tmp = *this;
        tmp += rhs;
        return tmp;
    }

    Vec operator-(const Vec &rhs) const {
        Vec tmp = *this;
        tmp -= rhs;
        return tmp;
    }

    Vec operator*(const Vec &rhs) const {
        Vec tmp = *this;
        tmp *= rhs;
        return tmp;
    }

    Vec operator/(const Vec &rhs) const {
        Vec tmp = *this;
        tmp /= rhs;
        return tmp;
    }

    Vec operator*(const T &rhs) const {
        Vec tmp = *this;
        tmp *= rhs;
        return tmp;
    }

    Vec operator/(const T &rhs) const {
        Vec tmp = *this;
        tmp /= rhs;
        return tmp;
    }

    T dot(const Vec &rhs) const {
        T sum = this->_v[0] * rhs._v[0];
        for (size_t i = 1; i < N; i++) {
            sum += this->_v[i] * rhs._v[i];
        }
        return sum;
    }

    T absDot(const Vec &rhs) const {
        return std::abs(dot(rhs));
    }

    T lengthSquared() const {
        return dot(*this);
    }

    T length() const {
        return std::sqrt(lengthSquared());
    }

    void normalize() {
        (*this) /= length();
    }

    Vec<T, N> normalized() const {
        auto t = *this;
        t.normalize();
        return t;
    }

    Vec<T, N> operator-() const {
        auto tmp = *this;
        for (auto i = 0; i < N; i++) {
            tmp._v[i] = -tmp._v[i];
        }
        return tmp;
    }

    const T &operator[](size_t i) const {
        return this->_v[i];
    }

    T &operator[](size_t i) {
        return this->_v[i];
    }

    T max() const {
        T v = this->_v[0];
        for (int i = 1; i < N; i++) {
            v = std::max(v, this->_v[i]);
        }
        return v;
    }

    T min() const {
        T v = this->_v[0];
        for (int i = 1; i < N; i++) {
            v = std::min(v, this->_v[i]);
        }
        return v;
    }

    friend Vec<T, N> operator*(T k, const Vec<T, N> &rhs) {
        auto t = rhs;
        t *= k;
        return t;
    }
};


template<class T, size_t N>
Vec<T, N> min(const Vec<T, N> &a, const Vec<T, N> &b) {
    Vec<T, N> r;
    for (int i = 0; i < N; i++) {
        r[i] = std::min(a[i], b[i]);
    }
    return r;
}

template<class T, size_t N>
Vec<T, N> max(const Vec<T, N> &a, const Vec<T, N> &b) {
    Vec<T, N> r;
    for (int i = 0; i < N; i++) {
        r[i] = std::max(a[i], b[i]);
    }
    return r;
}

using Point3f = Vec<float, 3>;
using Point3i = Vec<int, 3>;

using Point2f = Vec<float, 2>;
using Point2i = Vec<int, 2>;

template<class T, size_t N>
struct BoundBox {
    Vec<T, N> pMin, pMax;

    BoundBox unionOf(const BoundBox &box) const {
        return BoundBox{min(pMin, box.pMin), max(pMax, box.pMax)};
    }

    BoundBox unionOf(const Vec<T, N> &rhs) const {
        return BoundBox{min(pMin, rhs), max(pMax, rhs)};
    }

    Vec<T, N> centroid() const {
        return size() * 0.5 + pMin;
    }

    Vec<T, N> size() const {
        return pMax - pMin;
    }

    T surfaceArea() const {
        auto a = (size()[0] * size()[1] + size()[0] * size()[2] + size()[1] * size()[2]);
        return a + a;
    }

    bool intersects(const BoundBox &rhs) const {
        for (size_t i = 0; i < N; i++) {
            if (pMin[i] > rhs.pMax[i] || pMax[i] < rhs.pMin[i]);
            else {
                return true;
            }
        }
        return false;
    }

    Vec<T, N> offset(const Vec<T, N> &p) const {
        auto o = p - pMin;
        return o / size();
    }
};

using Bounds3f = BoundBox<float, 3>;

struct Vec3f : Vec<float, 3> {

    Vec3f(const Vec<float, 3> &v) : Vec(v.x, v.y, v.z) {}

    Vec3f(float x, float y, float z) : Vec(x, y, z) {}

    Vec3f(float x = 0) : Vec(x, x, x) {}

    Vec3f cross(const Vec3f &v) const {
        return Vec3f(
                y * v.z - z * v.y,
                z * v.x - x * v.z,
                x * v.y - y * v.x
        );
    }
};

inline void ComputeLocalFrame(const Vec3f &v1, Vec3f *v2, Vec3f *v3) {
    if (std::abs(v1.x) > std::abs(v1.y))
        *v2 = Vec3f(-v1.z, 0, v1.x) /
              std::sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        *v2 = Vec3f(0, v1.z, -v1.y) /
              std::sqrt(v1.y * v1.y + v1.z * v1.z);
    *v3 = v1.cross(*v2).normalized();
}

struct CoordinateSystem {
    CoordinateSystem() = default;

    explicit CoordinateSystem(const Vec3f &v) : normal(v) {
        ComputeLocalFrame(v, &localX, &localZ);
    }

    Vec3f worldToLocal(const Vec3f &v) const {
        return Vec3f(localX.dot(v), normal.dot(v), localZ.dot(v));
    }

    Vec3f localToWorld(const Vec3f &v) const {
        return Vec3f(v.x * localX + v.y * normal + v.z * localZ);
    }

private:
    Vec3f normal;
    Vec3f localX, localZ;
};

struct Matrix4 {
    Matrix4() = default;

    static Matrix4 identity() {
        float i[4][4] = {
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1},
        };
        return Matrix4(i);
    }

    static Matrix4 translate(const Vec3f &v) {
        float t[4][4] = {
                {1, 0, 0, v.x},
                {0, 1, 0, v.y},
                {0, 0, 1, v.z},
                {0, 0, 0, 1},
        };
        return Matrix4(t);
    }

    static Matrix4 scale(const Vec3f &v) {
        float s[4][4] = {
                {v.x, 0,   0,   0},
                {0,   v.y, 0,   0},
                {0,   0,   v.z, 0},
                {0,   0,   0,   1},
        };
        return Matrix4(s);
    }

    static Matrix4 rotate(const Vec3f &x, const Vec3f &axis, const Float angle) {
        const Float s = sin(angle);
        const Float c = cos(angle);
        const Float oc = Float(1.0) - c;
        float r[4][4] = {
                {oc * axis.x * axis.x + c,
                        oc * axis.x * axis.y - axis.z * s,
                           oc * axis.z * axis.x + axis.y * s, 0},
                {oc * axis.x * axis.y + axis.z * s,
                        oc * axis.y * axis.y + c,
                           oc * axis.y * axis.z - axis.x * s, 0},
                {oc * axis.z * axis.x - axis.y * s,
                        oc * axis.y * axis.z + axis.x * s,
                           oc * axis.z * axis.z + c,          0},
                {0,     0, 0,                                 1}
        };
        return Matrix4(r);
    }

    static Matrix4 lookAt(const Vec3f &from, const Vec3f &to) {
        Vec3f up(0, 1, 0);
        Vec3f d = to - from;
        d.normalize();
        Vec3f xAxis = up.cross(d).normalized();
        Vec3f yAxis = xAxis.cross(d).normalized();
        float m[4][4] = {
                {xAxis.x, yAxis.x, d.x, 0},
                {xAxis.y, yAxis.y, d.y, 0},
                {xAxis.z, yAxis.z, d.z, 0},
                {0,       0,       0,   1}
        };
        return Matrix4(m);
    }

    explicit Matrix4(float data[4][4]) {
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                _rows[i][j] = data[i][j];
            }
        }
    }

    Matrix4 operator*(const Matrix4 &rhs) const {
        Matrix4 m = *this;
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                m._rows[i][j] = _rows[i].dot(rhs.column(j));
            }
        }
        return m;
    }

    Matrix4 &operator*=(const Matrix4 &rhs) {
        auto m = *this * rhs;
        for (int i = 0; i < 4; i++)
            _rows[i] = m._rows[i];
        return *this;
    }

    Vec<float, 4> operator*(const Vec<float, 4> &v) const {
        return Vec<float, 4>{
                _rows[0].dot(v),
                _rows[1].dot(v),
                _rows[2].dot(v),
                _rows[3].dot(v)};
    }

    Vec<float, 4> column(size_t i) const {
        return Vec<float, 4>{_rows[0][i], _rows[1][i], _rows[2][i], _rows[3][i]};
    }

    Matrix4 inverse(bool *suc = nullptr) const {
        auto m = reinterpret_cast<const float *>(_rows);
        float inv[16], det;
        int i;

        inv[0] = m[5] * m[10] * m[15] -
                 m[5] * m[11] * m[14] -
                 m[9] * m[6] * m[15] +
                 m[9] * m[7] * m[14] +
                 m[13] * m[6] * m[11] -
                 m[13] * m[7] * m[10];

        inv[4] = -m[4] * m[10] * m[15] +
                 m[4] * m[11] * m[14] +
                 m[8] * m[6] * m[15] -
                 m[8] * m[7] * m[14] -
                 m[12] * m[6] * m[11] +
                 m[12] * m[7] * m[10];

        inv[8] = m[4] * m[9] * m[15] -
                 m[4] * m[11] * m[13] -
                 m[8] * m[5] * m[15] +
                 m[8] * m[7] * m[13] +
                 m[12] * m[5] * m[11] -
                 m[12] * m[7] * m[9];

        inv[12] = -m[4] * m[9] * m[14] +
                  m[4] * m[10] * m[13] +
                  m[8] * m[5] * m[14] -
                  m[8] * m[6] * m[13] -
                  m[12] * m[5] * m[10] +
                  m[12] * m[6] * m[9];

        inv[1] = -m[1] * m[10] * m[15] +
                 m[1] * m[11] * m[14] +
                 m[9] * m[2] * m[15] -
                 m[9] * m[3] * m[14] -
                 m[13] * m[2] * m[11] +
                 m[13] * m[3] * m[10];

        inv[5] = m[0] * m[10] * m[15] -
                 m[0] * m[11] * m[14] -
                 m[8] * m[2] * m[15] +
                 m[8] * m[3] * m[14] +
                 m[12] * m[2] * m[11] -
                 m[12] * m[3] * m[10];

        inv[9] = -m[0] * m[9] * m[15] +
                 m[0] * m[11] * m[13] +
                 m[8] * m[1] * m[15] -
                 m[8] * m[3] * m[13] -
                 m[12] * m[1] * m[11] +
                 m[12] * m[3] * m[9];

        inv[13] = m[0] * m[9] * m[14] -
                  m[0] * m[10] * m[13] -
                  m[8] * m[1] * m[14] +
                  m[8] * m[2] * m[13] +
                  m[12] * m[1] * m[10] -
                  m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] -
                 m[1] * m[7] * m[14] -
                 m[5] * m[2] * m[15] +
                 m[5] * m[3] * m[14] +
                 m[13] * m[2] * m[7] -
                 m[13] * m[3] * m[6];

        inv[6] = -m[0] * m[6] * m[15] +
                 m[0] * m[7] * m[14] +
                 m[4] * m[2] * m[15] -
                 m[4] * m[3] * m[14] -
                 m[12] * m[2] * m[7] +
                 m[12] * m[3] * m[6];

        inv[10] = m[0] * m[5] * m[15] -
                  m[0] * m[7] * m[13] -
                  m[4] * m[1] * m[15] +
                  m[4] * m[3] * m[13] +
                  m[12] * m[1] * m[7] -
                  m[12] * m[3] * m[5];

        inv[14] = -m[0] * m[5] * m[14] +
                  m[0] * m[6] * m[13] +
                  m[4] * m[1] * m[14] -
                  m[4] * m[2] * m[13] -
                  m[12] * m[1] * m[6] +
                  m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] +
                 m[1] * m[7] * m[10] +
                 m[5] * m[2] * m[11] -
                 m[5] * m[3] * m[10] -
                 m[9] * m[2] * m[7] +
                 m[9] * m[3] * m[6];

        inv[7] = m[0] * m[6] * m[11] -
                 m[0] * m[7] * m[10] -
                 m[4] * m[2] * m[11] +
                 m[4] * m[3] * m[10] +
                 m[8] * m[2] * m[7] -
                 m[8] * m[3] * m[6];

        inv[11] = -m[0] * m[5] * m[11] +
                  m[0] * m[7] * m[9] +
                  m[4] * m[1] * m[11] -
                  m[4] * m[3] * m[9] -
                  m[8] * m[1] * m[7] +
                  m[8] * m[3] * m[5];

        inv[15] = m[0] * m[5] * m[10] -
                  m[0] * m[6] * m[9] -
                  m[4] * m[1] * m[10] +
                  m[4] * m[2] * m[9] +
                  m[8] * m[1] * m[6] -
                  m[8] * m[2] * m[5];

        det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        if (det == 0) {
            if (suc) {
                *suc = false;
            }
        }

        det = 1.0 / det;

        Matrix4 out;
        auto invOut = reinterpret_cast<float *>(out._rows);
        for (i = 0; i < 16; i++)
            invOut[i] = inv[i] * det;
        if (suc) {
            *suc = true;
        }
        return out;
    }

private:
    Vec<float, 4> _rows[4];
    static_assert(sizeof(_rows) == sizeof(float) * 16, "Matrix4 must have packed 16 floats");
};

using Spectrum = Vec3f;

inline bool IsBlack(const Spectrum &s) {
    return s.x <= 0 || s.y <= 0 || s.z <= 0;
}

template<class T>
using Ref = std::shared_ptr<T>;

template<class T, class... Args>
Ref<T> MakeRef(Args &&... args) {
    return std::make_shared<T>(args...);
}

struct Ray {
    Vec3f o, d;
    float tMin, tMax;

    Ray() : tMin(-1), tMax(-1) {}

    Ray(const Vec3f &o, const Vec3f &d, Float tMin, Float tMax = MaxFloat)
            : o(o), d(d), tMin(tMin), tMax(tMax) {}
};


class Component;


class Sampler {
public:
    virtual void startPixel(const Point2i &, const Point2i &filmDimension) = 0;

    virtual Float next1D() = 0;

    virtual Point2f next2D() {
        return Point2f(next1D(), next1D());
    }

    virtual Ref<Sampler> clone() const = 0;
};

struct CameraSample {
    Point2f pLens;
    Point2f pFilm;
    Ray ray;
};

class Camera {
public:
    virtual Vec3f worldToCamera(const Vec3f &v) const = 0;

    virtual Vec3f cameraToWorld(const Vec3f &v) const = 0;

    virtual void generateRay(const Point2f &u1,
                             const Point2f &u2,
                             const Point2i &raster,
                             Point2i filmDimension,
                             CameraSample &sample) const = 0;
};


struct ShadingPoint {
    Point2f uv;
    Vec3f Ns;
    Vec3f Ng;
};

class Shader {
public:
    virtual Spectrum evaluate(const ShadingPoint &) const = 0;
};

class RGBShader : public Shader {
    Vec3f value;
public:
    RGBShader(const Vec3f &v) : value(v) {}

    Spectrum evaluate(const ShadingPoint &) const override {
        return value;
    }
};

class BSDF;

struct BSDFSample;

class BSDF {
public:
    enum Type {
        ENone = 0,
        EDiffuse = 1,
        EGlossy = 1 << 1,
        EReflection = 1 << 2,
        ETransmission = 1 << 3,
        ESpecular = 1 << 4,
        EAll = EDiffuse | EGlossy | ETransmission | ESpecular | EReflection,
        EAllButSpecular = EAll & ~ESpecular
    };

    virtual Type getType() const = 0;

    virtual Spectrum evaluate(const ShadingPoint &, const Vec3f &wo, const Vec3f &wi) const = 0;

    virtual Float evaluatePdf(const ShadingPoint &, const Vec3f &wo, const Vec3f &wi) const = 0;

    virtual void sample(Point2f u, const ShadingPoint &, BSDFSample &sample) const = 0;

    virtual bool isSpecular() const {
        return getType() & ESpecular;
    }
};

struct BSDFSample {
    Vec3f wo;
    Vec3f wi;
    float pdf = 0;
    Spectrum f;
    BSDF::Type sampledType = BSDF::ENone;
};

class Triangle;

class Mesh;

class Primitive;

class Shape;

struct Intersection {
    const Shape *shape = nullptr;
    const BSDF *bsdf = nullptr;
    float distance = MaxFloat;
    Vec3f p;
    Vec3f Ns, Ng;
    Point2f uv;
    CoordinateSystem localFrame;

    bool hit() const {
        return shape != nullptr;
    }

    void computeLocalFrame() {
        localFrame = CoordinateSystem(Ng);
        auto v = worldToLocal(Ng);
    }

    Vec3f worldToLocal(const Vec3f &v) const {
        return localFrame.worldToLocal(v);
    }

    Vec3f localToWorld(const Vec3f &v) const {
        return localFrame.localToWorld(v);
    }

    // w should be normalized
    Ray spawnRay(const Vec3f &w) const {
        auto t = RayBias / w.absDot(Ng);
        return Ray(p, w, t, MaxFloat);
    }

    Ray spawnTo(const Point3f &p) const {
        return Ray(this->p, (p - this->p), RayBias, 1);
    }
};

struct VisibilityTester;

struct LightSample {
    Vec3f wi;
    Spectrum Li;
    float pdf;
};

struct LightRaySample {
    Ray ray;
    Spectrum Le;
    float pdfPos, pdfDir;
};

class Light {
public:
    virtual Spectrum Li(ShadingPoint &sp) const = 0;

    virtual void sampleLi(const Point2f &u, Intersection &isct, LightSample &sample, VisibilityTester &) const = 0;

    virtual Float pdfLi(const Intersection &intersection, const Vec3f &wi) const = 0;

    virtual void sampleLe(const Point2f &u1, const Point2f &u2, LightRaySample &sample) = 0;

};

class AreaLight : public Light {
public:
    virtual void setShape(Shape *shape) = 0;
};

struct SurfaceSample {
    Point3f p;
    Float pdf;
    Vec3f normal;
};

class Primitive {
public:
    virtual bool intersect(const Ray &ray, Intersection &isct) const = 0;

    virtual Bounds3f getBoundingBox() const = 0;

    virtual AreaLight *getAreaLight() const { return nullptr; }

    virtual void sample(const Point2f &u, SurfaceSample &sample) const = 0;

    virtual Float area() const = 0;
};

class Shape : public Primitive {
public:
    virtual BSDF *getBSDF() const = 0;

    // split shape according to given axis
    // returns true iff split can be performed
    virtual bool split(Float coord, int axis, std::vector<Ref<Shape>> &splitResults) const { return false; }

};

class Triangle : public Shape {
    BSDF *bsdf = nullptr;
    Mesh *mesh = nullptr;
    uint32_t indexOffset = -1;
public:
    void sample(const Point2f &u, SurfaceSample &sample) const override {
        Point2f uv = u;
        if (uv.x + uv.y > 1.0f) {
            uv.x = 1.0f - uv.x;
            uv.y = 1.0f - uv.y;
        }
        sample.pdf = 1 / area();
        sample.p = (1 - uv.x - uv.y) * vertex(0) + uv.x * vertex(1) + uv.y * vertex(1);
        Vec3f e1 = (vertex(1) - vertex(0));
        Vec3f e2 = (vertex(2) - vertex(0));
        sample.normal = e1.cross(e2).normalized();
    }

    Vec3f vertex(uint32_t i) const;

    Float area() const override {
        return Vec3f(vertex(1) - vertex(0)).cross(vertex(2) - vertex(0)).length();
    }

    BSDF *getBSDF() const {
        return bsdf;
    }

    // returns true iff its a nearer intersection
    bool intersect(const Ray &ray, Intersection &isct) const override {
        float u, v;
        Vec3f e1 = (vertex(1) - vertex(0));
        Vec3f e2 = (vertex(2) - vertex(0));
        auto Ng = e1.cross(e2).normalized();
        float denom = (ray.d.dot(Ng));
        float t = -(ray.o - vertex(0)).dot(Ng) / denom;
        if (denom == 0)
            return false;
        if (t < ray.tMin)
            return false;
        Vec3f p = ray.o + t * ray.d;
        double det = e1.cross(e2).length();
        auto u0 = e1.cross(p - vertex(0));
        auto v0 = Vec3f(p - vertex(0)).cross(e2);
        if (u0.dot(Ng) < 0 || v0.dot(Ng) < 0)
            return false;
        v = u0.length() / det;
        u = v0.length() / det;
        if (u < 0 || v < 0 || u > 1 || v > 1)
            return false;
        if (u + v <= 1) {
            if (t < isct.distance) {
                isct.distance = t;
                isct.Ng = Ng;
                isct.Ns = Ng;
                isct.shape = this;
                return true;
            }
        }
        return false;
    }

    Bounds3f getBoundingBox() const override {
        return Bounds3f{
                min(vertex(0), min(vertex(1), vertex(2))),
                max(vertex(0), max(vertex(1), vertex(2)))
        };
    }
};

class Sphere : public Shape {
    Ref<BSDF> bsdf = nullptr;
    Vec3f center;
    Float radius;
    Ref<AreaLight> light;
public:
    Sphere(Float r, const Vec3f &c, const Ref<BSDF> &bsdf, const Ref<AreaLight> &light = nullptr)
            : bsdf(bsdf), center(c), radius(r), light(light) {
        if (light) {
            light->setShape(this);
        }
    }

    BSDF *getBSDF() const {
        return bsdf.get();
    }

    AreaLight *getAreaLight() const override {
        return light.get();
    }

    void sample(const Point2f &u, SurfaceSample &sample) const override {
        float theta = 2 * Pi * u[0];
        float v = 2 * u[1] - 1;
        auto t = std::sqrt(1 - v * v);
        sample.p = Point3f(v * std::sin(theta), v * std::cos(theta), v) * radius + center;
        sample.pdf = 1.0f / area();
        sample.normal = (sample.p - center).normalized();
    }

    Float area() const override {
        return 4 * Pi * radius * radius;
    }

    // returns true iff its a nearer intersection
    bool intersect(const Ray &ray, Intersection &isct) const override {
        auto oc = ray.o - center;
        auto a = ray.d.dot(ray.d);
        auto b = 2 * ray.d.dot(oc);
        auto c = oc.dot(oc) - radius * radius;
        auto delta = b * b - 4 * a * c;
        if (delta < 0) {
            return false;
        }
        auto t1 = (-b - std::sqrt(delta)) / (2 * a);
        if (t1 >= ray.tMin) {
            if (t1 < isct.distance) {
                isct.distance = t1;
                auto p = ray.o + t1 * ray.d;
                isct.Ng = p - center;
                isct.Ng.normalize();
                isct.Ns = isct.Ng;
                isct.shape = this;
                return true;
            }
        }
        auto t2 = (-b + std::sqrt(delta)) / (2 * a);
        if (t2 >= ray.tMin) {
            if (t2 < isct.distance) {
                isct.distance = t2;
                auto p = ray.o + t2 * ray.d;
                isct.Ng = p - center;
                isct.Ng.normalize();
                isct.Ns = isct.Ng;
                isct.shape = this;
                return true;
            }
        }
        return false;
    }

    Bounds3f getBoundingBox() const override {
        return Bounds3f{
                {center[0] - radius, center[1] - radius, center[2] - radius},
                {center[0] + radius, center[1] + radius, center[2] + radius}
        };
    }
};

struct CurrentPathGuard {
    CurrentPathGuard() : current(fs::current_path()) {}

    ~CurrentPathGuard() {
        fs::current_path(current);
    }

private:
    fs::path current;
};


class Mesh {
    friend class Triangle;

    std::vector<Triangle> triangles;
    std::vector<int> indices;
    std::vector<Vec3f> vertices;
public:
    static Ref<Mesh> loadObjFile(const std::string &filename, std::vector<Ref<Triangle>> &triangles) {
        CurrentPathGuard _guard;

    }
};

Vec3f Triangle::vertex(uint32_t i) const {
    return mesh->vertices[mesh->indices[indexOffset + i]];
}

class Accelerator : public Primitive {
public:
};


struct Scene {
    Ref<Accelerator> accelerator;
    std::vector<Light *> lights;
    std::vector<Ref<Primitive>> primitives;

    bool intersect(const Ray &ray, Intersection &isct) {
        rayCounter++;
//        bool hit = false;
//        for (auto &i:primitives) {
//            if (i->intersect(ray, isct))
//                hit = true;
//        }

        auto hit = accelerator->intersect(ray, isct);
        if (hit) {
            isct.p = ray.o + ray.d * isct.distance;
        }
        return hit;
    }

    void preprocess() {
        for (auto &i: primitives) {
            if (i->getAreaLight()) {
                lights.push_back(i->getAreaLight());
            }
        }
    }

    Light *sampleLight(Ref<Sampler> &sampler) const {
        auto i = std::min<int>(sampler->next1D() * lights.size(), lights.size() - 1);
        if (i >= 0) {
            return lights[i];
        }
        return nullptr;
    }

    size_t getRayCounter() const {
        return rayCounter;
    }

private:
    std::atomic<size_t> rayCounter = 0;
};

struct VisibilityTester {
    Ray shadowRay;

    bool visible(Scene &scene) {
        Intersection isct;
        if (!scene.intersect(shadowRay, isct) || isct.distance >= shadowRay.tMax - RayBias) {
            return true;
        }
        return false;
    }
};

class AtomicFloat {
    float value() const {
        return bitsToFloat(bits);
    }

    explicit operator float() const {
        return value();
    }

    void add(float v) {
        uint32_t old, newBits;
        do {
            old = bits;
            float val = bitsToFloat(old) + v;
            newBits = floatToBits(val);
        } while (!bits.compare_exchange_weak(old, newBits, std::memory_order::memory_order_relaxed));
    }

    explicit AtomicFloat(float v = 0) : bits(floatToBits(v)) {}

    AtomicFloat(const AtomicFloat &rhs) : bits(uint32_t(rhs.bits)) {}

private:
    static float bitsToFloat(uint32_t bits) {
        union {
            uint32_t i;
            float f;
        } v;
        v.i = bits;
        return v.f;
    }

    static uint32_t floatToBits(float f) {
        union {
            uint32_t i;
            float f;
        } v;
        v.f = f;
        return v.i;
    }

    std::atomic<uint32_t> bits;
};


using AtomicVec3f = Vec<AtomicFloat, 3>;


Point2f ConcentricSampleDisk(const Point2f &u) {
    Point2f uOffset = 2.f * u - Point2f(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
        return Point2f(0, 0);

    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = Pi4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = Pi2 - Pi4 * (uOffset.x / uOffset.y);
    }
    return r * Point2f(std::cos(theta), std::sin(theta));
}

Vec3f CosineHemisphereSampling(const Point2f &u) {
    auto uv = ConcentricSampleDisk(u);
    auto r = uv.lengthSquared();
    auto h = std::sqrt(1 - r);
    return Vec3f(uv.x, h, uv.y);
}

class GGXMicrofacetModel {
public:

};

class DiffuseBSDF : public BSDF {
    Ref<Shader> shader;
public:
    DiffuseBSDF(const Ref<Shader> &shader) : shader(shader) {}


    Spectrum evaluate(const ShadingPoint &point, const Vec3f &wo, const Vec3f &wi) const override {
        if (wo.y * wi.y > 0)
            return shader->evaluate(point) * InvPi;
        return {};
    }

    void sample(Point2f u, const ShadingPoint &sp, BSDFSample &sample) const override {
        sample.wi = CosineHemisphereSampling(u);
        sample.sampledType = BSDF::Type(sample.sampledType | getType());
        if (sample.wo.y * sample.wi.y < 0) {
            sample.wi.y = -sample.wi.y;
        }
        sample.pdf = std::abs(sample.wi.y) * InvPi;
        sample.f = evaluate(sp, sample.wo, sample.wi);
    }

    Float evaluatePdf(const ShadingPoint &point, const Vec3f &wo, const Vec3f &wi) const override {
        if (wo.y * wi.y > 0)
            return std::abs(wi.y) * InvPi;
        return 0;
    }

    Type getType() const override {
        return Type(EDiffuse | EReflection);
    }

};

template<class T, class F>
void ParallelFor(T begin, T end, F _f) {
    auto f = std::move(_f);
#pragma omp parallel for schedule(dynamic, 1)
    for (auto i = begin; i < end; i++) {
        f(i);
    }
}


class BVHAccelerator : public Accelerator {
    struct BVHNode {
        Bounds3f box;
        uint32_t first;
        uint32_t count;
        int left, right;

        bool isLeaf() const {
            return left < 0 && right < 0;
        }
    };

    std::vector<Ref<Primitive>> primitive;
    std::vector<BVHNode> nodes;

    Bounds3f boundBox;

    static Float intersectAABB(const Bounds3f &box, const Ray &ray, const Vec3f &invd) {
        Vec3f t0 = (box.pMin - ray.o) * invd;
        Vec3f t1 = (box.pMax - ray.o) * invd;
        Vec3f tMin = min(t0, t1), tMax = max(t0, t1);
        if (tMin.max() <= tMax.min()) {
            auto t = std::max(ray.tMin + RayBias, tMin.max());
            if (t >= ray.tMax + RayBias) {
                return -1;
            }
            return t;
        }
        return -1;
    }

    int recursiveBuild(int begin, int end, int depth) {
        Bounds3f box{{MaxFloat, MaxFloat, MaxFloat},
                     {MinFloat, MinFloat, MinFloat}};
        if (depth == 0) {
            boundBox = box;
        }

        if (end == begin)return -1;
        for (auto i = begin; i < end; i++) {
            box = box.unionOf(primitive[i]->getBoundingBox());
        }


        if (end - begin <= 4 || depth >= 20) {
            BVHNode node;

            node.box = box;
            node.first = begin;
            node.count = end - begin;
            node.left = node.right = -1;
            nodes.push_back(node);
            return nodes.size() - 1;
        } else {

            int axis = 0;
            auto size = box.size();
            if (size.x > size.y) {
                if (size.x > size.z) {
                    axis = 0;
                } else {
                    axis = 2;
                }
            } else {
                if (size.y > size.z) {
                    axis = 1;
                } else {
                    axis = 2;
                }
            }
            constexpr size_t nBuckets = 12;
            struct Bucket {
                size_t count = 0;
                Bounds3f bound;

                Bucket() = default;
            };
            Bucket buckets[nBuckets];
            for (int i = begin; i < end; i++) {
                int b = std::min<int>(nBuckets - 1,
                                      int(box.offset(primitive[i]->getBoundingBox().centroid())[axis] * nBuckets));
                buckets[b].count++;
                buckets[b].bound = buckets[b].bound.unionOf(primitive[i]->getBoundingBox());
            }
            Float cost[nBuckets - 1] = {0};
            for (int i = 0; i < nBuckets - 1; i++) {
                Bounds3f b0, b1;
                int count0 = 0, count1 = 0;
                for (int j = 0; j <= i; j++) {
                    b0 = b0.unionOf(buckets[j].bound);
                    count0 += buckets[j].count;
                }
                for (int j = i + 1; j < nBuckets; j++) {
                    b1 = b1.unionOf(buckets[j].bound);
                    count1 += buckets[j].count;
                }
                cost[i] = 0.125 + (count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) / box.surfaceArea();
            }
            int splitBuckets = 0;
            Float minCost = MaxFloat;
            for (int i = 0; i < nBuckets - 1; i++) {
                if (cost[i] < minCost) {
                    minCost = cost[i];
                    splitBuckets = i;
                }
            }
            auto mid = std::partition(&primitive[begin], &primitive[end - 1] + 1, [&](Ref<Primitive> &p) {
                int b = box.offset(p->getBoundingBox().centroid())[axis] * nBuckets;
                if (b == nBuckets) {
                    b = nBuckets - 1;
                }
                return b < splitBuckets;
            });
            auto ret = nodes.size();
            nodes.emplace_back();

            BVHNode &node = nodes.back();
            node.box = box;
            node.count = -1;
            nodes.push_back(node);
            nodes[ret].left = recursiveBuild(begin, mid - &primitive[0], depth + 1);
            nodes[ret].right = recursiveBuild(mid - &primitive[0], end, depth + 1);

            return ret;
        }
    }

public:
    BVHAccelerator(const std::vector<Ref<Primitive>> &primitives) : primitive(primitives) {
        recursiveBuild(0, primitive.size(), 0);
        printf("BVH nodes:%d\n", nodes.size());
    }

    bool intersect(const Ray &ray, Intersection &isct) const override {
        bool hit = false;
        auto invd = Vec3f(1) / ray.d;
        constexpr int maxDepth = 40;
        const BVHNode *stack[maxDepth];
        int sp = 0;
        stack[sp++] = &nodes[0];
        while (sp > 0) {
            auto p = stack[--sp];
            auto t = intersectAABB(p->box, ray, invd);
            //         printf("%f\n", t);
            if (t < 0 || t > isct.distance) {
                continue;
            }
            if (p->isLeaf()) {
                for (int i = p->first; i < p->first + p->count; i++) {
                    if (primitive[i]->intersect(ray, isct)) {
                        hit = true;
                    }
                }
            } else {
                if (p->left >= 0)
                    stack[sp++] = &nodes[p->left];
                if (p->right >= 0)
                    stack[sp++] = &nodes[p->right];
            }
        }
        return hit;
    }


    Bounds3f getBoundingBox() const override {
        return boundBox;
    }

    void sample(const Point2f &u, SurfaceSample &sample) const override {

    }

    Float area() const override {
        return 0;
    }
};


struct Film {
    std::vector<Spectrum> pixels;
    float weight = 0;
    const size_t width, height;

    Film(size_t w, size_t h) : width(w), height(h), pixels(w * h) {}

    static float gamma(float x, float k = 1.0f / 2.2f) {
        return std::pow(std::clamp(x, 0.0f, 1.0f), k);
    }

    static int toInt(float x) {
        return std::max<uint32_t>(0, std::min<uint32_t>(255, std::lroundf(gamma(x) * 255)));
    }

    void writePPM(const std::string &filename) {
        auto f = fopen(filename.c_str(), "w");
        fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
        for (int i = 0; i < width * height; i++)
            fprintf(f, "%d %d %d ",
                    toInt(pixels[i][0] * weight),
                    toInt(pixels[i][1] * weight),
                    toInt(pixels[i][2] * weight));
    }

    Spectrum &operator()(const Point2f &p) {
        return (*this)(p.x, p.y);
    }

    Spectrum &operator()(float x, float y) {
        int px = std::clamp<int>(std::lround(x * width), 0, width - 1);
        int py = std::clamp<int>(std::lround(y * height), 0, height - 1);
        return pixels.at(px + py * width);
    }

    Spectrum &operator()(int x, int y) {
        return pixels.at(x + y * width);
    }
};

class PerspectiveCamera : public Camera {
    Matrix4 transform, invTransform;
    Vec3f viewpoint;
    float fov;
public:
    PerspectiveCamera(const Vec3f &p1, const Vec3f &p2, Float fov) : fov(fov) {
        viewpoint = p1;
        transform = Matrix4::lookAt(p1, p2);
        invTransform = transform.inverse();
    }

    Vec3f worldToCamera(const Vec3f &v) const override {
        auto r = invTransform * Vec<Float, 4>(v.x, v.y, v.z, 1);
        return {r.x, r.y, r.z};
    }

    Vec3f cameraToWorld(const Vec3f &v) const override {
        auto r = transform * Vec<Float, 4>(v.x, v.y, v.z, 1);
        return {r.x, r.y, r.z};
    }

public:
    void generateRay(const Point2f &u1,
                     const Point2f &u2,
                     const Point2i &raster,
                     Point2i filmDimension,
                     CameraSample &sample) const override {
        float x = float(raster.x) / filmDimension.x;
        float y = 1 - float(raster.y) / filmDimension.y;

        Point2f pixelWidth(1.0 / filmDimension.x, 1.0 / filmDimension.y);
        sample.pFilm = {x, y};
        sample.pFilm += u1 * pixelWidth - 0.5f * pixelWidth;
        sample.pLens = {0, 0};
        x = 2 * x - 1;
        y = 2 * y - 1;
        y *= float(filmDimension.y) / filmDimension.x;
        float z = 1 / std::atan(fov / 2);
        auto d = Vec3f(x, y, 0) - Vec3f(0, 0, -z);
        d.normalize();
        auto o = Vec3f(sample.pLens.x, sample.pLens.y, 0);
        o = cameraToWorld(o) + viewpoint;
        d = cameraToWorld(d);
        sample.ray = Ray(o, d, RayBias);
    }

};

class RandomSampler : public Sampler {
    Rng rng;
public:
    RandomSampler(uint32_t seed = 0) : rng(seed) {}

    void startPixel(const Point2i &i, const Point2i &filmDimension) override {
        rng = Rng(i.x + i.y * filmDimension.x);
    }

    Float next1D() override {
        return rng.uniformFloat();
    }

    Ref<Sampler> clone() const override {
        return MakeRef<RandomSampler>();
    }
};


class Integrator {
public:
    virtual void render(Scene &scene, Ref<Camera> camera, Ref<Sampler> sampler, Film &film) = 0;
};

class DiffuseAreaLight : public AreaLight {
    Spectrum color;
    Shape *shape = nullptr;
public:
    DiffuseAreaLight(const Spectrum &color) : color(color) {

    }

    Float pdfLi(const Intersection &intersection, const Vec3f &wi) const override {
        Intersection _isct;
        Ray ray(intersection.p, wi, RayBias);
        if (!shape->intersect(ray, _isct)) {
            return 0.0f;
        }
        Float SA = shape->area() * wi.absDot(_isct.Ng) / (_isct.distance * _isct.distance);
        return 1.0f / SA;
    }

    void setShape(Shape *shape) override {
        this->shape = shape;
    }

    Spectrum Li(ShadingPoint &sp) const override {
        return color;
    }

    void sampleLi(const Point2f &u, Intersection &isct, LightSample &sample, VisibilityTester &tester) const override {
        SurfaceSample surfaceSample;
        shape->sample(u, surfaceSample);
        auto wi = surfaceSample.p - isct.p;
        tester.shadowRay = Ray(isct.p, wi, RayBias, 1);
        sample.Li = color;
        sample.wi = wi.normalized();
        sample.pdf = wi.lengthSquared() / sample.wi.absDot(surfaceSample.normal) * surfaceSample.pdf;

    }

    void sampleLe(const Point2f &u1, const Point2f &u2, LightRaySample &sample) override {

    }

};

class RTAO : public Integrator {
    int spp;
public:
    RTAO(int spp) : spp(spp) {}

    void render(Scene &scene, Ref<Camera> camera, Ref<Sampler> sampler, Film &film) override {
        for (int i = 0; i < film.width; i++) {
            for (int j = 0; j < film.height; j++) {
                for (int s = 0; s < spp; s++) {
                    CameraSample sample;
                    camera->generateRay(sampler->next2D(), sampler->next2D(), {i, j}, Point2i{film.width, film.height},
                                        sample);
                    Intersection isct;
                    if (scene.intersect(sample.ray, isct)) {
                        isct.computeLocalFrame();
                        auto wo = isct.worldToLocal(-sample.ray.d);
                        auto w = CosineHemisphereSampling(sampler->next2D());
                        if (wo.y * w.y < 0) {
                            w = -w;
                        }
                        auto ray = isct.spawnRay(w);
                        isct = Intersection();
                        if (!scene.intersect(ray, isct) || isct.distance >= 30) {
                            film(sample.pFilm) += Spectrum(1);
                        }
                    }
                }
            }
        }
        film.weight = 1.0f / spp;
    }
};

// MIS Path Tracer
class PathTracer : public Integrator {
    int spp;
public:
    PathTracer(int spp) : spp(spp) {}

    static Float MISWeight(Float p1, Float p2) {
        p1 *= p1;
        p2 *= p2;
        return p1 / (p1 + p2);
    }

    Spectrum Li(Scene &scene, const Ray &_ray, Ref<Sampler> &sampler, Intersection *_isct) {
        Spectrum L;
        Spectrum beta(1);
        const int maxDepth = 4;
        Intersection isct;
        if (_isct) {
            isct = *_isct;
        }
        int depth = 0;
        Ray ray = _ray;
        Intersection prevIsct;
        BSDFSample sample;
        bool isSpecular = false;
        while (true) {
            if (!isct.hit())break;
            isct.computeLocalFrame();
            auto shape = isct.shape;
            auto light = shape->getAreaLight();
            ShadingPoint shadingPoint;
            shadingPoint.uv = isct.uv;
            shadingPoint.Ng = isct.Ng;
            shadingPoint.Ns = isct.Ns;
            if (light) {
                if (depth == 0 || isSpecular) {
                    L += beta * light->Li(shadingPoint);
                } else {
                    Float scatteringPdf = sample.pdf;
                    Float lightPdf = 1.0f / scene.lights.size() * light->pdfLi(prevIsct, ray.d);
                    auto w = MISWeight(scatteringPdf, lightPdf);
                    L += beta * light->Li(shadingPoint) * w;
                }
            }
            if (++depth > maxDepth) {
                break;
            }
            auto bsdf = shape->getBSDF();
            if (!bsdf) {
                break;
            }
            sample = BSDFSample();
            sample.wo = -isct.worldToLocal(ray.d).normalized();
            bsdf->sample(sampler->next2D(), shadingPoint, sample);
            if (sample.pdf <= 0)break;

            isSpecular = sample.sampledType & BSDF::ESpecular;

            auto sampledLight = scene.sampleLight(sampler);
            if (sampledLight) {
                LightSample lightSample;
                VisibilityTester visibilityTester;
                sampledLight->sampleLi(sampler->next2D(), isct, lightSample, visibilityTester);
                auto wi = isct.worldToLocal(lightSample.wi);
                auto f = bsdf->evaluate(shadingPoint, sample.wo, wi) * lightSample.wi.absDot(isct.Ns);
                Float lightPdf = lightSample.pdf / scene.lights.size();
                Float scatteringPdf = bsdf->evaluatePdf(shadingPoint, sample.wo, wi);
                if (!IsBlack(f) && visibilityTester.visible(scene)) {
                    if (isSpecular) {
                        L += beta * f * lightSample.Li / lightPdf;
                    } else {
                        auto w = MISWeight(lightPdf, scatteringPdf);
                        L += beta * f * lightSample.Li / lightPdf * w;
                    }
                }
            }

            auto wi = isct.localToWorld(sample.wi);
            beta *= sample.f / sample.pdf * std::abs(wi.dot(isct.Ng));
            ray = isct.spawnRay(wi);
            prevIsct = isct;
            isct = Intersection();
            scene.intersect(ray, isct);
        }
        return L;
    }

    void render(Scene &scene, Ref<Camera> camera, Ref<Sampler> _sampler, Film &film) override {
        auto beginTime = std::chrono::steady_clock::now();
        ParallelFor<int>(0, film.height, [&](int j) {
            auto sampler = _sampler->clone();
            for (int i = 0; i < film.width; i++) {
                sampler->startPixel({i, j}, Point2i(film.width, film.height));
                for (int s = 0; s < spp; s++) {
                    CameraSample sample;
                    camera->generateRay(sampler->next2D(), sampler->next2D(), {i, j}, Point2i{film.width, film.height},
                                        sample);
                    Intersection isct;
                    if (scene.intersect(sample.ray, isct)) {
                        film(sample.pFilm) += Li(scene, sample.ray, sampler, &isct);
                    }
                }
            }
        });
        film.weight = 1.0f / spp;
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = (endTime - beginTime);
        printf("%llu rays traced, %f M rays/sec\n", scene.getRayCounter(),
               scene.getRayCounter() / elapsed.count() / 1e6);
    }

};

int main() {
    Ref<Camera> camera(new PerspectiveCamera(Vec3f(50.0, 40.8, 220.0), Vec3f(50.0, 40.8, 0.0), DegreesToRadians(60)));
    Film film(1080 / 2, 720 / 2);
    film.weight = 1;
    Scene scene;
    scene.primitives = {
            MakeRef<Sphere>(6.0, Vec3f(10, 70, 51.6), nullptr, MakeRef<DiffuseAreaLight>(Vec3f(100., 100., 100.))),
            MakeRef<Sphere>(1e5, Vec3f(1e5 + 1, 40.8, 81.6),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(0.7 * Vec3f(0.75, 0.25, 0.25)))),
            MakeRef<Sphere>(1e5, Vec3f(-1e5 + 99, 40.8, 81.6),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(0.7 * Vec3f(0.25, 0.25, 0.75)))),
            MakeRef<Sphere>(1e5, Vec3f(50, 40.8, 1e5),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(0.7 * Vec3f(0.75, 0.65, 0.75)))),
            MakeRef<Sphere>(1e5, Vec3f(50, 40.8, -1e5 + 350),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(0.7 * Vec3f(0.50, 0.50, 0.50)))),
            MakeRef<Sphere>(1e5, Vec3f(50, 1e5, 81.6),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(0.7 * Vec3f(0.65, 0.75, 0.75)))),
            MakeRef<Sphere>(1e5, Vec3f(50, -1e5 + 81.6, 81.6),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(0.7 * Vec3f(0.75, 0.75, 0.65)))),
            MakeRef<Sphere>(20, Vec3f(50, 20, 50),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(Vec3f(0.25, 0.75, 0.25)))),
            MakeRef<Sphere>(16.5, Vec3f(19, 16.5, 25),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(Vec3f(0.99, 0.99, 0.99)))),
            MakeRef<Sphere>(16.5, Vec3f(77, 16.5, 78),
                            MakeRef<DiffuseBSDF>(MakeRef<RGBShader>(Vec3f(0.99, 0.99, 0.99))))
    };
    scene.preprocess();
    scene.accelerator = MakeRef<BVHAccelerator>(scene.primitives);
    Ref<Sampler> sampler(new RandomSampler());
    Ref<Integrator> integrator(new PathTracer(64));
    integrator->render(scene, camera, sampler, film);
    film.writePPM("out.ppm");
}
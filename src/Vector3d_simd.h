/*
 * vector3dSimd.h
 *
 *  Created on: Mar 10, 2017
 *      Author: maxim
 */

#ifndef vector3dSIMD_H_
#define vector3dSIMD_H_

#include "VectorsInternal.h"

/*!
 * \class vector3d_simd
 * \brief Optimized class of a 3d vector. Contains information of three coordinates (x, y, z) and main algebraic methods.
 * All coordinates are stored as double precision floating points. Class requires for AVX2 support. If machine has no AVX2
 * instructions a regular non-optimized class should be used.
 */
class _MM_ALIGN32 vector3d_simd {
public:
    typedef double elt_type;

    /*!
     * Member variables
     */
    union
    {
        struct _MM_ALIGN32 { elt_type x, y, z; };
        __m256d mmvalue;
    };
private:
    uint8_t been_inserted;      //!< Used in comma initializer

public:

    /*!
     * \brief Default constructor. All values will be assignet to zero.
     */
    MUSTINLINE vector3d_simd() : mmvalue(_mm256_setzero_pd()), been_inserted(0) { }

    /*!
     * \brief Constructor takes three doubles of coordinates and assign them to the internal field.
     */
    MUSTINLINE vector3d_simd(elt_type x, elt_type y, elt_type z) :
        mmvalue(_mm256_set_pd(0.0, z, y, x)), been_inserted(0) { }

    /*!
     * \brief Constructor copies data from another register
     */
    MUSTINLINE vector3d_simd(__m256d other) : mmvalue(other), been_inserted(0) { }

    /*!
     * \brief Addition operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator+ (const vector3d_simd &other) const {
        return _mm256_add_pd(mmvalue, other.mmvalue);
    }

    /*!
     * \brief Subtraction operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator- (const vector3d_simd &other) const {
        return _mm256_sub_pd(mmvalue, other.mmvalue);
    }

    /*!
     * \brief Multiplication operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator* (const vector3d_simd &other) const {
        return _mm256_mul_pd(mmvalue, other.mmvalue);
    }

    /*!
     * \brief Division operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator/ (const vector3d_simd &other) const {
        /*
         * FIXME: optimize it! Be aware of division by 0 from the "dummy" variable of b vector!
         */
        __m256d r= other.mmvalue;
        r[3] = 1.0;
        return _mm256_div_pd(mmvalue, r);
    }

    /*!
     * \brief Addition assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd& operator+= (const vector3d_simd &other) {
        mmvalue = _mm256_add_pd(mmvalue, other.mmvalue);
        return *this;
    }

    /*!
     * \brief Subtraction assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd& operator-= (const vector3d_simd &other) {
        mmvalue = _mm256_sub_pd(mmvalue, other.mmvalue);
        return *this;
    }

    /*!
     * \brief Multiplication assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd& operator*= (const vector3d_simd &other) {
        mmvalue = _mm256_mul_pd(mmvalue, other.mmvalue);
        return *this;
    }

    /*!
     * \brief Division assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd& operator/= (const vector3d_simd &other) {
        /*
         * FIXME: optimize it! Be aware of division by 0 from the "dummy" variable of b vector!
         */
        __m256d r= other.mmvalue;
        r[3] = 1.0;
        mmvalue = _mm256_div_pd(mmvalue, r);
        return *this;
    }

    /*!
     * \brief Addition of scalar value to all coordinates of \e this vector
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator+ (elt_type value) const {
        return _mm256_add_pd(mmvalue, _mm256_set_pd(0.0, value, value, value));
    }

    /*!
     * \brief Subtraction of scalar value to all coordinates of \e this vector
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator- (elt_type value) const {
        return _mm256_sub_pd(mmvalue, _mm256_set_pd(0.0, value, value, value));
    }

    /*!
     * \brief Multiplication of all coordinates of \e this vector by scalar value
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator* (elt_type value) const {
        return _mm256_mul_pd(mmvalue, _mm256_set_pd(0.0, value, value, value));
    }
    friend MUSTINLINE vector3d_simd operator*(elt_type value, const vector3d_simd &rhs)  {
        return rhs * value;
    }

    /*!
     * \brief Division of all coordinates of \e this vector by scalar value
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd operator/ (elt_type value) const {
        return _mm256_div_pd(mmvalue, _mm256_set_pd(1.0, value, value, value));
    }

    /*!
     * \brief Addition and assignment operator for scalar value
     * Add given scalar value from all coordinates of \e this vector
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3d_simd& operator+= (elt_type value) {
        mmvalue = _mm256_add_pd(mmvalue, _mm256_set_pd(0.0, value, value, value));
        return *this;
    }

    /*!
     * \brief Subtraction and assignment operator for scalar value
     * Subtract given scalar value from all coordinates of \e this vector
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3d_simd& operator-= (elt_type value) {
        mmvalue = _mm256_sub_pd(mmvalue, _mm256_set_pd(0.0, value, value, value));
        return *this;
    }

    /*!
     * \brief Multiplication assignment operator for scalar value
     * Multiplies all coordinates of \e this vector by scalar value
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3d_simd& operator*= (elt_type value) {
        mmvalue = _mm256_mul_pd(mmvalue, _mm256_set_pd(0.0, value, value, value));
        return *this;
    }

    /*!
     * \brief Division assignment operator for scalar value
     * Divides all coordinates of \e this vector by scalar value
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3d_simd& operator/= (elt_type value) {
        mmvalue = _mm256_div_pd(mmvalue, _mm256_set_pd(1.0, value, value, value));
        return *this;
    }

    /*!
     * \brief Assignment operator for scalars
     * Assigns given scalar to all coordinates
     * @param value Scalar value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3d_simd& operator= (elt_type value) {
        mmvalue = _mm256_set_pd(0.0, value, value, value);
        return *this;
    }

    /*!
     * \brief Assignment operator for vector
     * Assigns given vector of three coordinates to \e this
     * @param other Scalar value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3d_simd& operator= (const vector3d_simd &other) {
        mmvalue = other.mmvalue;
        return *this;
    }

    /*!
     * \brief Explicit set of three coordinates
     * Assigns given coordinates to \e this vector
     * @param x Coordinate
     * @param y Coordinate
     * @param z Coordinate
     */
    MUSTINLINE void set(elt_type x, elt_type y, elt_type z) {
        mmvalue = _mm256_set_pd(0.0, z, y, x);
    }

    /*!
     * \brief Cross product of two vectors
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3d_simd cross(const vector3d_simd &other) const {
        return _mm256_sub_pd(
            _mm256_mul_pd(
                    _mm256_permute4x64_pd(mmvalue, _MM_SHUFFLE(3, 0, 2, 1)),
                    _mm256_permute4x64_pd(other.mmvalue, _MM_SHUFFLE(3, 1, 0, 2))),
            _mm256_mul_pd(
                    _mm256_permute4x64_pd(mmvalue, _MM_SHUFFLE(3, 1, 0, 2)),
                    _mm256_permute4x64_pd(other.mmvalue, _MM_SHUFFLE(3, 0, 2, 1)))
            );
    }

    /*!
     * \brief Dot product of two vectors
     * @param other Other vector
     * @return Result as a scalar
     */
    MUSTINLINE elt_type dot(const vector3d_simd &other) const {
        __m256d r1 = _mm256_mul_pd(mmvalue, mmvalue);
        __m256d r2 = _mm256_hadd_pd(r1, r1);
        return _mm_cvtsd_f64(_mm_add_pd(_mm256_extractf128_pd(r2, 1),
            _mm256_castpd256_pd128(r2)));
    }

    /*!
     * \brief Length (absolute value) of \e this vector
     * @return Result as a scalar
     */
    MUSTINLINE elt_type length() const {
        __m256d r1 = _mm256_mul_pd(mmvalue, mmvalue);
        __m256d r2 = _mm256_hadd_pd(r1, r1);
        return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_add_pd(_mm256_extractf128_pd(r2, 1),
            _mm256_castpd256_pd128(r2))));
    }

    /*!
     * \brief Reciprocal length (absolute value) of \e this vector
     * @return Result as a scalar
     */
    MUSTINLINE elt_type rlength() const {
        return 1./length();
    }

    /*!
     * \brief Normalization of \e this vector
     * @return Vector scaled to unit length
     */
    MUSTINLINE vector3d_simd normalize() const {
        __m256d r = _mm256_set1_pd(rlength());
        return _mm256_mul_pd(mmvalue, r);
    }

    /*!
     * \brief Prints coordinates of vector into the stream
     * @param os Reference to a stream
     * @param v Vector to be printed
     * @return Reference to the stream
     */
    friend MUSTINLINE std::ostream& operator<< (std::ostream& os, const vector3d_simd &v) {
        os << v.x << ' ' << v.y << ' ' << v.z << ' ';
        return os;
    }

    /*!
     * \brief Set vector through the stream
     * @param input Reference to a stream
     * @param v Vector to be printed
     * @return Reference to the stream
     */
    friend MUSTINLINE std::istream &operator>> (std::istream  &input, vector3d_simd &v) {
        input >> v.x >> v.y >> v.z;
        return input;
    }

    /*!
     * \brief Allows to initialize vector in convenient way through operator<<
     */
    MUSTINLINE vector3d_simd& operator<< (const elt_type &value) {
        x = value;
        been_inserted = 1;
        return *this;
    }

    /*!
     * \brief Inserts \e value in the vector
     * \warning There should be no more than 3 values passed into the vector using comma initializer. Otherwise
     * an error will be printed out
     */
    MUSTINLINE vector3d_simd& operator, (const elt_type &value) {
        if (been_inserted == 0) {
            std::cerr << "Error! Bad use of comma initializer. Do not mess up the code! "
                "Assign values to vector using operator<<..." << std::endl;
            return *this;
        }
        if (been_inserted == 1) {
            y = value;
            ++been_inserted;
        }
        else if (been_inserted == 2) {
            z = value;
            ++been_inserted;
        }
        else
            std::cerr << "Error! Too many arguments have been passed to comma initializer "
                "(see operator<<, vector3d_simd)..." << std::endl;
        return *this;
    }
};

#endif /* vector3dSIMD_H_ */

/* ****************************************************************************** *
 * MIT License                                                                    *
 *                                                                                *
 * Copyright (c) 2018 Maxim Masterov                                              *
 *                                                                                *
 * Permission is hereby granted, free of charge, to any person obtaining a copy   *
 * of this software and associated documentation files (the "Software"), to deal  *
 * in the Software without restriction, including without limitation the rights   *
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      *
 * copies of the Software, and to permit persons to whom the Software is          *
 * furnished to do so, subject to the following conditions:                       *
 *                                                                                *
 * The above copyright notice and this permission notice shall be included in all *
 * copies or substantial portions of the Software.                                *
 *                                                                                *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    *
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  *
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  *
 * SOFTWARE.                                                                      *
 * ****************************************************************************** */

#ifndef VECTOR3F_H_
#define VECTOR3F_H_

#include "VectorsInternal.h"

/*!
 * \class vector3f
 * \brief Optimized class of a 3d vector. Contains information of three coordinates (x, y, z) and main algebraic methods.
 * All coordinates are stored as single precision floating points. Class asks for SSE4.2 support. If machine has no SSE4.2
 * instructions a regular non-optimized class should be used
 */
class _MM_ALIGN16 vector3f_simd {
public:
    typedef float elt_type;

    // Member variables
    union
    {
        struct _MM_ALIGN16 { elt_type x, y, z; };
        __m128 mmvalue;
    };

private:
    uint8_t been_inserted;      //!< Used in comma initializer

public:
    /*!
     * \brief Default constructor. All values will be assignet to zero.
     */
    MUSTINLINE vector3f_simd() : mmvalue(_mm_setzero_ps()), been_inserted(0) { }

    /*!
     * \brief Constructor takes three doubles of coordinates and assign them to the internal field.
     */
    MUSTINLINE vector3f_simd(elt_type x, elt_type y, elt_type z) :
        mmvalue(_mm_set_ps(0.0f, z, y, x)), been_inserted(0) { }

    /*!
     * \brief Constructor copies data from another register
     */
    MUSTINLINE vector3f_simd(__m128 other) : mmvalue(other), been_inserted(0) { }

    /*!
     * \brief Addition operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator+(const vector3f_simd &other) const {
        return _mm_add_ps(mmvalue, other.mmvalue);
    }

    /*!
     * \brief Subtraction operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator-(const vector3f_simd &other) const {
        return _mm_sub_ps(mmvalue, other.mmvalue);
    }

    /*!
     * \brief Multiplication operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator*(const vector3f_simd &other) const {
        return _mm_mul_ps(mmvalue, other.mmvalue);
    }

    /*!
     * \brief Division operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator/(const vector3f_simd &other) const {
        return _mm_div_ps(mmvalue, other.mmvalue);
    }

    /*!
     * \brief Addition assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd &operator+=(const vector3f_simd &other) {
        mmvalue = _mm_add_ps(mmvalue, other.mmvalue);
        return *this;
    }

    /*!
     * \brief Subtraction assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd &operator-=(const vector3f_simd &other) {
        mmvalue = _mm_sub_ps(mmvalue, other.mmvalue);
        return *this;
    }

    /*!
     * \brief Multiplication assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd &operator*=(const vector3f_simd &other) {
        mmvalue = _mm_mul_ps(mmvalue, other.mmvalue);
        return *this;
    }

    /*!
     * \brief Division assignment operator
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd &operator/=(const vector3f_simd &other) {
        mmvalue = _mm_div_ps(mmvalue, other.mmvalue);
        return *this;
    }

    /*!
     * \brief Addition of scalar value to all coordinates of \e this vector
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator+(elt_type value) const {
        return _mm_add_ps(mmvalue, _mm_set1_ps(value));
    }

    /*!
     * \brief Subtraction of scalar value to all coordinates of \e this vector
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator-(elt_type value) const {
        return _mm_sub_ps(mmvalue, _mm_set1_ps(value));
    }

    /*!
     * \brief Multiplication of all coordinates of \e this vector by scalar value
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator*(elt_type value) const {
        return _mm_mul_ps(mmvalue, _mm_set1_ps(value));
    }
    friend MUSTINLINE vector3f_simd operator*(elt_type value, const vector3f_simd &rhs)  {
        return rhs * value;
    }

    /*!
     * \brief Division of all coordinates of \e this vector by scalar value
     * @param value value
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd operator/(elt_type value) const {
        return _mm_div_ps(mmvalue, _mm_set1_ps(value));
    }

    /*!
     * \brief Addition and assignment operator for scalar value
     * Add given scalar value from all coordinates of \e this vector
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3f_simd &operator+=(elt_type value) {
        mmvalue = _mm_add_ps(mmvalue, _mm_set1_ps(value));
        return *this;
    }

    /*!
     * \brief Subtraction and assignment operator for scalar value
     * Subtract given scalar value from all coordinates of \e this vector
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3f_simd &operator-=(elt_type value) {
        mmvalue = _mm_sub_ps(mmvalue, _mm_set1_ps(value));
        return *this;
    }

    /*!
     * \brief Multiplication assignment operator for scalar value
     * Multiplies all coordinates of \e this vector by scalar value
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3f_simd &operator*=(elt_type value) {
        mmvalue = _mm_mul_ps(mmvalue, _mm_set1_ps(value));
        return *this;
    }

    /*!
     * \brief Division assignment operator for scalar value
     * Divides all coordinates of \e this vector by scalar value
     * @param value value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3f_simd &operator/=(elt_type value) {
        mmvalue = _mm_div_ps(mmvalue, _mm_set1_ps(value));
        return *this;
    }

    /*!
     * \brief Assignment operator for scalars
     * Assigns given scalar to all coordinates
     * @param value Scalar value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3f_simd &operator=(elt_type value) {
        mmvalue = _mm_set_ps(0, value, value, value);
        return *this;
    }

    /*!
     * \brief Assignment operator for vector
     * Assigns given vector of three coordinates to \e this
     * @param other Scalar value
     * @return Reference to \e this vector
     */
    MUSTINLINE vector3f_simd &operator=(const vector3f_simd &other) {
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
        mmvalue = _mm_set_ps(0.0f, z, y, x);
    }

    /*!
     * \brief Cross product of two vectors
     * @param other Other vector
     * @return Result of \e this type
     */
    MUSTINLINE vector3f_simd cross(const vector3f_simd &other) const
    {
        return _mm_sub_ps(
            _mm_mul_ps(_mm_shuffle_ps(mmvalue, mmvalue, _MM_SHUFFLE(3, 0, 2, 1)),
                _mm_shuffle_ps(other.mmvalue, other.mmvalue, _MM_SHUFFLE(3, 1, 0, 2))),
            _mm_mul_ps(_mm_shuffle_ps(mmvalue, mmvalue, _MM_SHUFFLE(3, 1, 0, 2)),
                _mm_shuffle_ps(other.mmvalue, other.mmvalue, _MM_SHUFFLE(3, 0, 2, 1)))
            );
    }

    /*!
     * \brief Dot product of two vectors
     * @param other Other vector
     * @return Result as a scalar
     */
    MUSTINLINE elt_type dot(const vector3f_simd &other) const {
        return _mm_cvtss_f32(_mm_dp_ps(mmvalue, other.mmvalue, 0x71));
    }

    /*!
     * \brief Length (absolute value) of \e this vector
     * @return Result as a scalar
     */
    MUSTINLINE elt_type length() const {
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(mmvalue, mmvalue, 0x71)));
    }

    /*!
     * \brief Reciprocal length (absolute value) of \e this vector
     * @return Result as a scalar
     */
    MUSTINLINE elt_type rlength() const {
        return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_dp_ps(mmvalue, mmvalue, 0x71)));
    }

    /*!
     * \brief Normalization of \e this vector
     * @return Vector scaled to unit length
     */
    MUSTINLINE vector3f_simd normalize() const {
        return _mm_mul_ps(mmvalue, _mm_rsqrt_ps(_mm_dp_ps(mmvalue, mmvalue, 0x7F)));
    }

    /*!
     * \brief Prints coordinates of vector into the stream
     * @param os Reference to a stream
     * @param v Vector to be printed
     * @return Reference to the stream
     */
    friend MUSTINLINE std::ostream& operator<< (std::ostream& os, const vector3f_simd &v) {
        os << v.x << ' ' << v.y << ' ' << v.z << ' ';
        return os;
    }

    /*!
     * \brief Set vector through the stream
     * @param input Reference to a stream
     * @param v Vector to be printed
     * @return Reference to the stream
     */
    friend MUSTINLINE std::istream &operator>> (std::istream  &input, vector3f_simd &v) {
        input >> v.x >> v.y >> v.z;
        return input;
    }

    /*!
     * \brief Allows to initialize vector in convenient way through operator<<
     */
    MUSTINLINE vector3f_simd& operator<< (const elt_type &value) {
        x = value;
        been_inserted = 1;
        return *this;
    }

    /*!
     * \brief Inserts \e value in the vector
     * \warning There should be no more than 3 values passed into the vector using comma initializer. Otherwise
     * an error will be printed out
     */
    MUSTINLINE vector3f_simd& operator, (const elt_type &value) {
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
#endif /* VECTOR3F_H_ */

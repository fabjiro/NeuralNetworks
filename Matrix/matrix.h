#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

using namespace std;

typedef float(*Activation)(const float&);

class Matrix
{
private:
    float Random() noexcept;
    int32_t GetNumRows() const noexcept;
    int32_t GetNumColumns() const noexcept;

public:
    // operator overload
    vector<vector<float>> data;

    // -
    Matrix operator-(const Matrix&) noexcept;

    // +=
    Matrix& operator+=(Matrix&) noexcept;
    Matrix& operator+=(const float&) noexcept;

    // *=
    Matrix& operator*=(Matrix&) noexcept;
    Matrix& operator*=(const float&) noexcept;

    static Matrix Transpose(const Matrix &) noexcept;
    static Matrix Multiply(const Matrix &, const Matrix &) noexcept;

    Matrix& Map(Activation fun) noexcept;
    static Matrix& Map(Matrix &, Activation fun) noexcept;

    static Matrix FromArray(const vector<float> &) noexcept;
    vector<float> ToArray() noexcept;

    explicit Matrix(const int32_t &, const int32_t &) noexcept;
    void Print() noexcept;
};
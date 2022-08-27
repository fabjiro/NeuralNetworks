#include "matrix.h"

Matrix::Matrix(const int32_t& row, const int32_t& column)  noexcept {
    for (int32_t i = 0; i < row; ++i) {
        this->data.push_back(std::vector<float>(column));
        for (int32_t j = 0; j < column; ++j) {
            this->data[i][j] = this->Random() - this->Random();
        }
    }
}

void Matrix::Print() noexcept {
    for (const vector<float>& item : this->data) {
        for (const float& it : item) {
            cout << it << " ";
        }
        cout << endl;
    }
}

Matrix Matrix::Transpose(const Matrix& matrix) noexcept {
    Matrix temp = Matrix(matrix.GetNumColumns(), matrix.GetNumRows());
    for (int32_t i = 0; i < matrix.GetNumRows(); ++i)
        for (int32_t j = 0; j < matrix.GetNumColumns(); ++j)
            temp.data[j][i] = matrix.data[i][j];

    return temp;
}

Matrix Matrix::Multiply(const Matrix& m1, const Matrix& m2) noexcept {
    auto temp = Matrix{ m1.GetNumRows(), m2.GetNumColumns() };
    float sum{};

    for (int32_t i = 0; i < temp.GetNumRows(); ++i) {
        for (int32_t j = 0; j < temp.GetNumColumns(); ++j) {
            sum = 0.0;
            for (int32_t k = 0; k < m1.GetNumColumns(); ++k) {
                sum += m1.data[i][k] * m2.data[k][j];
            }
            temp.data[i][j] = sum;
        }
    }

    return temp;
}

Matrix& Matrix::Map(Activation fun) noexcept {
    for (auto& row : this->data)
        for (auto& column : row)
            column = fun(column);
    return *this;
}

Matrix& Matrix::Map(Matrix& m, Activation fun) noexcept {
    for (auto& row : m.data)
        for (auto& column : row)
            column = fun(column);
    return m;
}


Matrix Matrix::FromArray(const vector<float>& v) noexcept {
    auto temp = Matrix{ (int32_t)v.size(), 1 };
    for (int32_t i = 0; i < v.size(); ++i)
        temp.data[i][0] = v[i];
    return temp;
}

vector<float> Matrix::ToArray() noexcept {
    vector<float> temp;
    for (const auto& row : this->data)
        for (const auto& column : row)
            temp.push_back(column);
    return temp;
}

float Matrix::Random() noexcept {
    return ((float)rand()) / ((float)RAND_MAX);
}

Matrix Matrix::operator-(const Matrix& M1) noexcept {
    Matrix result = Matrix{ this->GetNumRows(), this->GetNumColumns() };
    for (int32_t i = 0; i < result.GetNumRows(); ++i)
        for (int32_t j = 0; j < result.GetNumColumns(); j++)
            result.data[i][j] = this->data[i][j] - M1.data[i][j];
    return result;
}

// +=
Matrix& Matrix::operator+=(Matrix& M1) noexcept {
    for (auto i1 = this->data.begin(), i2 = M1.data.begin();
         i1 != this->data.end() && i2 != M1.data.end(); ++i1, ++i2)
        for (auto i3 = (*i1).begin(), i4 = (*i2).begin();
             i3 != (*i1).end() && i4 != (*i2).end(); ++i3, ++i4)
            *i3 += *i4; //operation
    return *this;
}

Matrix& Matrix::operator+=(const float& n) noexcept {
    for (auto& row : this->data)
        for (auto& column : row)
            column += n;
    return *this;
}


//*=
Matrix& Matrix::operator*=(Matrix& M1) noexcept {
    for (auto i1 = this->data.begin(), i2 = M1.data.begin();
         i1 != this->data.end() && i2 != M1.data.end(); ++i1, ++i2)
        for (auto i3 = (*i1).begin(), i4 = (*i2).begin();
             i3 != (*i1).end() && i4 != (*i2).end(); ++i3, ++i4)
            *i3 *= *i4; //operation
    return *this;
}

Matrix& Matrix::operator*=(const float& n) noexcept {
    for (auto& row : this->data)
        for (auto& column : row)
            column *= n;
    return *this;
}

int32_t Matrix::GetNumRows() const noexcept {
    return (int32_t)this->data.size();
}

int32_t Matrix::GetNumColumns() const noexcept {
    if (this->GetNumRows() > 0) {
        return (int32_t)this->data[0].size();
    }
    return 0;
}
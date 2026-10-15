//
// Created by Pascal on 05.10.2026.
// B-Spline functions
//

#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <Eigen/dense>

using Vec2 = Eigen::Vector2d;
using VecXd = Eigen::VectorXd;
using Mat2 = Eigen::Matrix2d;
using MatXd = Eigen::MatrixXd;

double B00 ()
{
    return 1;
}
double B10 (double x)
{
    return 1.0 - x;
}

double B11 (double x)
{
    return x;
}

double B20 (double x)
{
    return 0.5*pow((x-1),2);

}
double B21 (double x)
{
    return -(x*x) + x + 0.5;
}

double B22 (double x)
{
    return 0.5*(x*x);
}

double B30 (double x)
{
    return -(1.0/6.0)*pow((x-1), 3);
}

double B31 (double x)
{
    return 0.5*pow(x,3)  - (x*x) + (2.0/3.0) ;
}

double B32 (double x)
{
    return -0.5*pow(x,3) + (1.0/6.0) + 0.5*pow(x, 2) + 0.5*x;
}

double B33 (double x)
{
    return (1.0/6.0)*pow(x, 3);
}

std::array<double, 4> B3(double x)
{
    return {
    B30(x), B31(x), B32(x), B33(x)};
}

double interpolate1D (double x, const VecXd& u)
{
    int i = static_cast<int>(std::floor(x));
    double t = x-i;
    return u(i) * B10(t) + u(i+1) * B11(t);
}


double interpolate2D (double x, double y, const Eigen::Matrix4d& u)
{
    auto bx = B3(x);
    auto by = B3(y);

    double res = 0.0;

    for (int i = 0; i<4; ++i) {
        for (int j = 0; j < 4; ++j)
        {
            res += u(i,j) * bx[i] * by[j];
        }
    }

    return res;
}

int main()
{
    double x = 0.7;
    double y = 0.5;
    VecXd u(9);
    u << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    Eigen::Matrix4d A {{ 1.0, 2.0, 3.0, 4.0 },
        { 2.0, 3.0, 4.0, 5.0 },
        { 3.0, 4.0, 5.0, 6.0 },
        { 4.0, 5.0, 6.0, 7.0 }};

    std::cout << interpolate1D(x,u);

    std::cout << interpolate2D(x,y, A) << std::endl;

    return 0;
}
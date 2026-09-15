/* Fluid Implicit Particles on Coadjoint Orbits
 * Toy example implementation for Algorithm 1 using a simple rotation
*/
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>

using Vec2 = Eigen::Vector2d;
using Mat2 = Eigen::Matrix2d;

struct Particle
{
    Vec2 pos;
    Vec2 u; // impulse
};

using State = std::vector<Particle>;

struct GridVelocity
{
    double omega = 0.0;
};

// Toy Example Grid Velocity
GridVelocity F(const State& state)
{
    double meanX = 0.0;

    for (const auto& p : state)
    {
        meanX += p.pos.x();
    }

    meanX /= state.size();

    return {1.0 + 0.25* meanX };

}

// velocity of rotation

Vec2 velocity(const GridVelocity& v, const Vec2& pos)
{
    return Vec2(-v.omega * pos.y(), v.omega * pos.x());
}

Mat2 gradVelocity(const GridVelocity& v, const Vec2&)
{
    Mat2 gradV;

    gradV << 0.0, -v.omega,
        v.omega, 0.0;

    return gradV;
}

struct Derivative
{
    Vec2 dx;
    Vec2 dm;
};

// Advection, Equation (17)
// du/dt = -(grad v(x))^T u(t)

Derivative derivative(const Particle& p, const GridVelocity& f)
{
    const Vec2 v = velocity(f, p.pos);
    const Mat2 gradV = gradVelocity(f, p.pos);

    Derivative d;

    d.dx = v;
    d.dm = -gradV.transpose() * p.u;

    return d;
}

// 4th order runge kutta

Particle rk4(const Particle& p, const GridVelocity& v, double dt)
{
    Derivative k1 = derivative(p, v);

    Particle p2;
    p2.pos = p.pos + k1.dx * 0.5*dt;
    p2.u = p.u + k1.dm * 0.5*dt;
    Derivative k2 = derivative(p2, v);

    Particle p3;
    p3.pos = p.pos + k2.dx * 0.5*dt;
    p3.u = p.u + k2.dm * 0.5*dt;
    Derivative k3 = derivative(p3, v);

    Particle p4;
    p4.pos = p.pos + k3.dx * dt;
    p4.u = p.u + k3.dm * dt;
    Derivative k4 = derivative(p4, v);

    Particle result;
    result.pos = p.pos + dt / 6.0 * (k1.dx + 2.0*k2.dx + 2.0*k3.dx + k4.dx);

    result.u = p.u + dt / 6.0 * (k1.dm + 2.0*k2.dm + 2.0*k3.dm + k4.dm);

    return result;
}

// Time Integration

State timeStep(
    const State& y_n,
    double dt,
    int maxIterations = 10,
    double tolerance = 1e-9)
{

    GridVelocity f_n = F(y_n);
    GridVelocity f_star = f_n;

    State y_next(y_n.size());

    for (int i = 0; i < maxIterations; i++)
    {
        for (std::size_t j = 0; j < y_n.size(); j++)
        {
            y_next[j] = rk4(y_n[j], f_star, dt); //advect every particle
        }

        GridVelocity f_next = F(y_next);

        GridVelocity f_star_new {
            0.5 * (f_n.omega + f_next.omega) // averaged grid velocity (19b)
        };

        double err =
            std::abs(f_star_new.omega - f_star.omega) / std::max(1e-12, std::abs(f_star.omega));

        f_star = f_star_new;

        if (err < tolerance)
            break;
    }

    return y_next;
}

int main()
{
    // test particle
    Particle p{
        Vec2(1.0, 0.0),
        Vec2(1.0, 0.0)
    };
    GridVelocity v{1.0};
    constexpr double dt = 0.1;

    Particle res = rk4(p, v, dt);

    // exact solution using rotation matrix
    const double t = v.omega * dt;

    Mat2 R;

    R << std::cos(t), -std::sin(t),
        std::sin(t), std::cos(t);

    Vec2 exactPos = R* p.pos;
    Vec2 exactU = R * p.u;

    std::cout << std::setprecision(15);

    std::cout << "RK4 position:\n"
              << res.pos << "\n\n";

    std::cout << "Exact position:\n"
              << exactPos << "\n\n";

    std::cout << "Position error:\n"
              << (res.pos - exactPos).norm()
              << "\n\n";

    std::cout << "RK4 u:\n"
              << res.u << "\n\n";

    std::cout << "Exact u:\n"
              << exactU << "\n\n";

    std::cout << "u error:\n"
              << (res.u - exactU).norm()
              << '\n';

    return 0;
}
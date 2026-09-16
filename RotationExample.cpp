/* Fluid Implicit Particles on Coadjoint Orbits
 * Toy example implementation for Algorithm 1 and 2 using a simple rotation with drift
*/
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include <fstream>

using Vec2 = Eigen::Vector2d;
using Mat2 = Eigen::Matrix2d;

struct Particle
{
    Vec2 pos;
    Vec2 u; // impulse
};


using State = std::vector<Particle>;

// Grid velocity for rotation with drift
struct GridVelocity
{
    Vec2 value = Vec2::Zero();

    double omega() const { return value(0);}
    double drift() const { return value(1); }
};

struct StepResult
{
    State y;
    GridVelocity f;
};


// Toy Example Grid Velocity
/* old version with rotation only
GridVelocity F(const State& state)
{
    double meanX = 0.0;

    for (const auto& p : state)
    {
        meanX += p.pos.x();
    }

    meanX /= state.size();

    return {1.0 + 0.25* meanX };

}*/

GridVelocity F(const State& state)
{
    double meanX = 0.0;
    double meanY = 0.0;

    for (const auto& p : state)
    {
        meanX += p.pos.x();
        meanY += p.pos.y();
    }

    meanX /= state.size();
    meanY /= state.size();

    GridVelocity f;

    f.value << 1.0 + 0.25*meanX, 0.25 * meanY;

    /*GridVelocity f;

    f.value << 1.0, 0.0; */

    return f;

}

// velocity of rotation

Vec2 velocity(const GridVelocity& f, const Vec2& pos)
{
    return Vec2(-f.omega() * pos.y() + f.drift(),
        f.omega() * pos.x());
}

Mat2 gradVelocity(const GridVelocity& f, const Vec2&)
{
    Mat2 gradV;

    gradV << 0.0, -f.omega(),
        f.omega(), 0.0;

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

// Time Integration (Algortihm 1)

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
            0.5 * (f_n.value + f_next.value) // averaged grid velocity (19b)
        };

        double err =
            (f_star_new.value - f_star.value).norm() / std::max(1e-12, f_star.value.norm());

        f_star = f_star_new;

        if (err < tolerance)
            break;
    }

    return y_next;
}

// orthogonal projection for energy correction (for 21c)
Vec2 orthogonalProjection(const Vec2& delta, const Vec2& f_star)
{
    const double denom = f_star.squaredNorm();

    if (denom < 1e-10) return delta;

    return delta - (f_star*f_star.dot(delta) / denom);

}

// Energy based correction (Algortihm 2)

StepResult EnergyCorrection(const State& y_n,
    const GridVelocity& f_n,
    double dt,
    int maxIterations = 10,
    double tolerance = 1e-9) {

GridVelocity f_star = F(y_n);;

State y_next(y_n.size());

    GridVelocity f_next = f_n;
    GridVelocity f_raw_prev = F(y_n);


for (int i = 0; i < maxIterations; i++)
{
    // Advection
    for (std::size_t j = 0; j < y_n.size(); j++)
    {
        y_next[j] = rk4(y_n[j], f_star, dt); //advect every particle (21a)
    }

    GridVelocity f_raw = F(y_next);

    GridVelocity f_star_new {
        0.5 * (f_n.value + f_raw.value) // averaged grid velocity (19b)
    };

    Vec2 delta = f_raw.value - f_n.value;

    f_next.value = f_n.value + orthogonalProjection(delta, f_star_new.value); // correction (21c)

    const double err = (f_raw.value - f_raw_prev.value).norm() / std::max(1e-12, f_next.value.norm());

    if (err < tolerance)
        break;

    f_star = f_star_new;
    f_raw_prev = f_raw;
}

    return {y_next, f_next};

}

/*
void printState(const State& state)
{
    for (int i = 0; i < state.size(); ++i)
    {
        std::cout
            << "pos = "
            << state.at(i).pos.transpose()
            << ", u ="
            << state.at(i).u.transpose()
            << "\n";
    }
} */

double energy(const GridVelocity& f)
{
    return 0.5 * f.value.squaredNorm();
}

int main()
{

    State initialState = {
        {
            Vec2(1.0, 0.0),
            Vec2(1.0, 0.0)
        },
        {
            Vec2(0.0, 1.0),
            Vec2(0.0, 1.0)
        },
        {
        Vec2(-0.5, 0.5),
            Vec2(0.5, 0.5)
            }
    };

    constexpr double dt = 0.1;
    constexpr int numberOfSteps = 100;

    std::ofstream file("results.csv");
    file << "x,y\n";


    // Algo 1 test
    State stateAlg1 = initialState;

    for (int step = 0; step < numberOfSteps; ++step) {
        stateAlg1 = timeStep(
            stateAlg1,
            dt);

        file << stateAlg1[0].pos.x() << ", "
        << stateAlg1[0].pos.y()
        << "\n";
    }

    // Algo 2 test
    State stateAlg2 = initialState;

    GridVelocity fAlg2 = F(stateAlg2);

    for (int step = 0; step < numberOfSteps; ++step)
    {
        const double initialEnergy = energy(fAlg2);

        std::cout << initialEnergy << "\n";

        StepResult result =
            EnergyCorrection(stateAlg2,
                fAlg2,
                dt
                );

        stateAlg2 = result.y;
        fAlg2 = result.f;

        const double finalEnergy = energy(fAlg2);

        std::cout << initialEnergy - finalEnergy << "\n";
    }

    /*
    // RK 4 test

    // test particle
    Particle p{
        Vec2(1.0, 0.0),
        Vec2(1.0, 0.0)
    };
    GridVelocity v;
    v.value << 1.0, 0.0;

    constexpr double dt = 0.1;

    Particle res = rk4(p, v, dt);

    // exact solution using rotation matrix
    const double t = v.omega() * dt;

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
              << '\n';*/



    return 0;
}
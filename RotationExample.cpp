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
using VecXd = Eigen::VectorXd;
using Mat2 = Eigen::Matrix2d;
using MatXd = Eigen::MatrixXd;

struct Particle
{
    Vec2 pos;
    Vec2 u; // impulse
};

using State = std::vector<Particle>;

// Grid velocity for rotation with drift
struct VelocityField
{
    Vec2 value = Vec2::Zero();

    double omega() const { return value(0);}
    double drift() const { return value(1); }
};

// setup for experiments
enum class ModelType
{
    ConstantRotation,
    RotationWithDrift,
    StateDependent
};

enum class Algorithm
{
    TrapezIntegrator,
    EnergyCorrection
};

struct ExperimentConfig
{
    std::string name;

    ModelType model;
    Algorithm algorithm;

    double dt;
    int steps;

    int maxIterations;
    double tolerance;

    double omega;
    double drift;
};

struct StepResult
{
    State y;
    VelocityField f;
};

// Toy Example Grid Velocity

VelocityField F(const State& state, const ExperimentConfig& config)
{
    VelocityField f;

    switch (config.model)
    {
    case ModelType::ConstantRotation:
        {
            f.value << config.omega, 0.0;
            break;
        }

    case ModelType::RotationWithDrift:
        {
            f.value << config.omega,
                        config.drift;
            break;
        }
    case ModelType::StateDependent:
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

            VelocityField f;

            f.value << 1.0 + 0.25*meanX, 0.25 * meanY;

            break;
        }
    }

    return f;

}

// velocity of rotation
/*
Vec2 velocity(const VelocityField& f, const Vec2& pos)
{
    return Vec2(-f.omega() * pos.y() + f.drift(),
        f.omega() * pos.x());
}

*/

// velocity of linear translation (non zero divergence)
Vec2 velocity(const VelocityField& f, const Vec2& pos)
{
    return Vec2(2*pos.x(),
        pos.y());
}


// 2d MACGrid
// divergence is calculated using central differences implied by the grid
// see Fluid Simulation for Computer Graphics 5.2
struct MACGrid
{

    int nx;
    int ny;
    double h;

    MatXd p;
    MatXd v_x;
    MatXd v_y;

    MACGrid(int nx, int ny, double h): nx(nx), ny(ny), h(h), p(ny, nx),
    v_x(ny, nx + 1),
    v_y(ny + 1, nx)
    {
        p.setZero();
        v_x.setZero();
        v_y.setZero();
    }

    void setVelocity(const VelocityField& f)
    {
        // horizontal faces
        for (int j = 0; j < nx; ++j)
        {
            for (int i = 0; i <= ny; ++i)
            {
                Vec2 pos((i+0.5) *h, j*h);
                v_x(j, i) = velocity(f, pos).x();
            }
        }

        // vertical faces
        for (int j = 0; j <= ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                Vec2 pos(i*h, (j+0.5)*h);
                v_y(j, i) = velocity(f, pos).y();
            }
        }
    }

    double divergence(int i, int j) const
    {
        double dv_x_dx = (v_x(j, i + 1) - v_x(j, i)) / h;
        double dv_y_dy = (v_y(j + 1, i) - v_y(j, i)) / h;

        return dv_x_dx + dv_y_dy;
    }

};

// logic for this thing is given by the formula for divergence above
// so for the v_x part the non zero entries are 1/h (-1,1,...,0) for the first and then shifted for every row 1/h(0,-1,1,...,0)
// and for the v_y part the non zero entries are separated by n_y-1 0's, so 1/h(-1,0,1,...,0) for n_y = 2
MatXd divergenceMatrix(MACGrid& grid)
{
    const int nx = grid.nx;
    const int ny = grid.ny;
    const double& h = grid.h;
    const int N = nx*ny;

    MatXd D = MatXd::Zero(N, ny*(nx+1) + (ny+1)*nx);

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int row = j * nx + i;

            // v_x
            D(row, j*(nx+1) + i) = -1.0 / h;
            D(row, j*(nx+1) + (i + 1)) = 1.0 / h;


            //v_y
            D(row, ny * (nx + 1) + j*nx + i) = -1.0 / h;
            D(row, ny * (nx + 1) + (j + 1) *nx + i) = 1.0 / h;
        }
    }

    return D;
}

VecXd velocityVector(MACGrid& grid)
{
    const int nx = grid.nx;
    const int ny = grid.ny;
    MatXd v_x = grid.v_x;
    MatXd v_y = grid.v_y;

    VecXd f(ny * (nx + 1) + (ny + 1) * nx);

    int k = 0;

    // v_x values
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            f(k++) = v_x(j, i);
        }
    }

    // v_y values
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            f(k++) = v_y(j, i);
        }
    }

    return f;
}

// pressure projection (non galerkin)
// uses inverse matrix -> not so good -> how not to use inverse?
VecXd fProj(MatXd& D, VecXd& f)
{

    MatXd A = D * D.transpose();
    MatXd Id = MatXd::Identity(f.size(), f.size());

    MatXd P =  Id - D.transpose() * A.inverse() * D;
    return P * f;

}

Mat2 gradVelocity(const VelocityField& f, const Vec2&)
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

Derivative derivative(const Particle& p, const VelocityField& f)
{
    const Vec2 v = velocity(f, p.pos);
    const Mat2 gradV = gradVelocity(f, p.pos);

    Derivative d;

    d.dx = v;
    d.dm = -gradV.transpose() * p.u;

    return d;
}

// 4th order runge kutta

Particle rk4(const Particle& p, const VelocityField& v, double dt)
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
    const ExperimentConfig& config)
{

    VelocityField f_n = F(y_n, config);
    VelocityField f_star = f_n;

    State y_next(y_n.size());

    for (int i = 0; i < config.maxIterations; i++)
    {
        for (std::size_t j = 0; j < y_n.size(); j++)
        {
            y_next[j] = rk4(y_n[j], f_star, config.dt); //advect every particle
        }

        VelocityField f_next = F(y_next, config);

        VelocityField f_star_new {
            0.5 * (f_n.value + f_next.value) // averaged grid velocity (19b)
        };

        double err =
            (f_star_new.value - f_star.value).norm() / std::max(1e-12, f_star.value.norm());

        f_star = f_star_new;

        if (err < config.tolerance)
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
    const VelocityField& f_n, const ExperimentConfig& config) {

VelocityField f_star = F(y_n, config);;

State y_next(y_n.size());

    VelocityField f_next = f_n;
    VelocityField f_raw_prev = F(y_n, config);


for (int i = 0; i < config.maxIterations; i++)
{
    // Advection
    for (std::size_t j = 0; j < y_n.size(); j++)
    {
        y_next[j] = rk4(y_n[j], f_star, config.dt); //advect every particle (21a)
    }

    VelocityField f_raw = F(y_next, config);

    VelocityField f_star_new {
        0.5 * (f_n.value + f_raw.value) // averaged grid velocity (19b)
    };

    Vec2 delta = f_raw.value - f_n.value;

    f_next.value = f_n.value + orthogonalProjection(delta, f_star_new.value); // correction (21c)

    const double err = (f_raw.value - f_raw_prev.value).norm() / std::max(1e-12, f_next.value.norm());

    if (err < config.tolerance)
        break;

    f_star = f_star_new;
    f_raw_prev = f_raw;
}

    return {y_next, f_next};

}

double energy(const VelocityField& f)
{
    return 0.5 * f.value.squaredNorm();
}

void runExperiment(const ExperimentConfig& config,
    const State& initialState)
{
    State state = initialState;

    std::ofstream ofile (
        "../results/" + config.name + ".csv"
        );

    ofile <<
        "step, time, particle, x, y,"
        "omega, drift, energy\n";

    for (int step = 0; step < config.steps; step++)
    {
        VelocityField f = F(state,config);

        for (std::size_t j = 0; j < state.size(); ++j)
        {
            ofile
            << step << ", "
            << step*config.dt << ", "
            << j << ", "
            << state.at(j).pos.x() << ", "
            << state.at(j).pos.y() << ", "
            << f.omega() << ", "
            << f.drift() << ", "
            << "\n";
        }

        if (step <= config.steps)
        {
            state = timeStep(state, config);
        }

    }
}

int main()
{

    VelocityField f;
    f.value << 1.0, 1.0;

    MACGrid grid(2,2,1.0/2);
    grid.setVelocity(f);

    MatXd D = divergenceMatrix(grid);

    std::cout << D << std::endl;

    VecXd f_vec = velocityVector(grid);

    //std::cout << D << std::endl;
    //std::cout << f_vec << std::endl;


    std::cout << fProj(D, f_vec) <<  std::endl;
    std::cout << D*fProj(D, f_vec) <<  std::endl;


    for (int j = 0; j < grid.ny; ++j)
    {
        for (int i = 0; i<grid.nx; ++i)
        {
           // std::cout << "div(" << i << ", " << j << ") = " << grid.divergence(i,j) << std::endl;
        }
    }
/*
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

    ExperimentConfig exp1 {
        "01_constant_rotation",
        ModelType::ConstantRotation,
        Algorithm::TrapezIntegrator,
        0.1,
        100,
        10,
        1e-9,
        1.0,
        0.0
    };

    ExperimentConfig exp2 = exp1;
    exp2.name = "02_rotation_with_drift_05";
    exp2.model = ModelType::RotationWithDrift;
    exp2.drift = 0.5;

    ExperimentConfig exp3 = exp2;
    exp3.name = "03_rotation_with_drift_3";
    exp3.drift = 3;

    ExperimentConfig exp4 = exp1;
    exp4.name = "04_state_dependent";

    runExperiment(exp1, initialState);
    runExperiment(exp2, initialState);
    runExperiment(exp3, initialState);
    runExperiment(exp4, initialState);



    constexpr double dt = 0.1;
    constexpr int numberOfSteps = 100;

    std::ofstream file("../results.csv");
    file << "x,y\n";


    // Algo 1 test
    State stateAlg1 = initialState;
    GridVelocity fAlg1 = F(stateAlg1);


    for (int step = 0; step < numberOfSteps; ++step) {

        const double initialEnergy = energy(fAlg1);

        stateAlg1 = timeStep(
            stateAlg1,
            dt);

        file << stateAlg1[0].pos.x() << ", "
        << stateAlg1[0].pos.y()
        << "\n";

        fAlg1 = F(stateAlg1, config );
        const double finalEnergy = energy(fAlg1);

        std::cout << initialEnergy - finalEnergy << "\n";
    }

    // Algo 2 test
    State stateAlg2 = initialState;

    GridVelocity fAlg2 = F(stateAlg2);

    for (int step = 0; step < numberOfSteps; ++step)
    {
        const double initialEnergy = energy(fAlg2);

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
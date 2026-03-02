#ifndef LIPPMANN_SCHWINGER_HH
#define LIPPMANN_SCHWINGER_HH LIPPMANN_SCHWINGER_HH

#include <algorithm>
#include "cufft.h"
#include "cublas_v2.h"
#include "common.hh"
#include "derivatives.hh"
#include "fourier.hh"

class LippmannSchwingerSolver
{
public:
    /** @brief Constructor create new instance
     *
     * Initialise all state variables and allocate required memory
     *
     * @param[in] grid_spec specification of computational grid
     * @param[in] verbose verbosity level: 0 = no output, 1 = print summary, >1 = print at every iteration
     */
    LippmannSchwingerSolver(const GridSpec grid_spec, const int verbose = 0);

    /** @brief Destructor
     *
     * Free all allocated memory
     */
    ~LippmannSchwingerSolver();

    /** @brief Compute stress on device
     *
     * Given epsilon and spatially varying Lame parameters lambda, mu compute the
     * stress sigma according to sigma_{ij} = C_{ijkl} epsilon_{kl} where
     *
     *      C_{ijkl} = lambda*delta_{ij}delta_{kl} + mu*(delta_{ik}delta_{jl}+delta_{il}delta_{jk})
     *
     * @param[in] dev_epsilon strain epsilon in real space (device array, size 6*ncells)
     * @param[out] dev_sigma resulting stress sigma in real space (device array, size 6*ncells)
     * @param[in] dev_lambda Lame parameter lambda (device array, size ncells)
     * @param[in] dev_mu Lame parameter mu (device array, size ncells)
     */
    void compute_stress(float *dev_epsilon, float *dev_sigma,
                        float *dev_lambda, float *dev_mu);

    /** @brief Solve for a given set of Lame parameters
     *
     * @param[in]
     * @param[in] lambda Lame parameter lambda (host array, size ncells)
     * @param[in] mu Lame parameter mu (host array, size ncells)
     * @param[in] epsilon_bar average value of epsilon (host array, size 6)
     * @param[out] epsilon Resulting strain (host array, size 6*ncells)
     * @param[out] sigma Resulting stress (host array, size 6*ncells)
     * @param[in] rtol relative tolerance on normalised divergence
     * @param[in] atol absolute tolerance on normalised divergence
     * @param[in] maxiter maximum number of iterations
     */
    int apply(float *lambda, float *mu, float *epsilon_bar,
              float *epsilon, float *sigma,
              float rtol = 1.E-4, float atol = 1.E-7, int maxiter = 100);

    /** @brief Compute normalised divergence for stopping criterion in Fourier space
     *
     * Compute the relative divergence norm
     *
     *      sqrt(<||div(sigma)||^2>) / ||<sigma>||
     *
     * which in Fourier space is given by
     *
     *      sqrt(<||xi.hat(sigma)||^2>) / ||hat(sigma)(0)||
     *
     * @param[in] dev_sigma_hat stress in Fourier space
     */
    float relative_divergence_norm(cufftComplex *dev_sigma_hat);

protected:
    /** @brief Increment solution
     *
     * Increment
     *
     *      epsilon -> epsilon + 1/ncells * r
     *
     * where the factor 1/ncells arises since the inverse Fourier transformation
     * in cuFFT is not normalised.
     *
     * @param[inout] dev_epsilon solution (device array, size 6*ncells)
     * @param[in] dev_increment increment (device array, size 6*ncells)
     */
    void increment_solution(float *dev_epsilon, cufftComplex *dev_increment);

    /* Set the values of epsilon to bar(epsilon) on the device */
    void set_epsilon_bar(float *dev_epsilon, float *epsilon_bar);

    /* Class variables */
    /** @brief specification of computational grid */
    const GridSpec grid_spec;
    /** @brief verbosity level */
    const int verbose;
    /** @brief Fourier vectors */
    float *dev_xi;
    /** @brief normalised Fourier vectors */
    float *dev_xi_zero;
    /** @brief Lame parameter lamba on device */
    float *dev_lambda;
    /** @brief Lame parameter mu on device */
    float *dev_mu;
    /** @brief real-valued strain epsilon on device */
    float *dev_epsilon;
    /** @brief real-valued stress sigma on device */
    float *dev_sigma;
    /** @brief divergence of sigma on device */
    float *dev_div_sigma;
    /** @brief complex-valued Fourier-stress on device */
    cufftComplex *dev_sigma_hat;
    /** @brief complex-valued divergence of Fourier-stress on device */
    cufftComplex *dev_div_sigma_hat;
    /** @brief complex-valued residual on device */
    cufftComplex *dev_residual;
    /** @brief temporary storage for zero mode of sigma in Fourier space */
    cufftComplex *sigma_0;
    /** @brief cuBLAS handle */
    cublasHandle_t handle;
    /** @brief cuFFT plan */
    cufftHandle plan;
};

/** @brief Solve linear elasticity problem with Lippmann-Schwinger iteration
 *
 * Provides an interface which can be called externally
 *
 * @param[in] lambda Lame parameter lambda (host array, size ncells)
 * @param[in] mu Lame parameter mu (host array, size ncells)
 * @param[in] epsilon_bar average value of epsilon (host array, size 6)
 * @param[out] epsilon Resulting strain (host array, size 6*ncells)
 * @param[out] sigma Resulting stress (host array, size 6*ncells)
 * @param[in] cells Number of cells (nx,ny,nz)
 * @param[in] extents Size of domain in each direction (Lx,Ly,Lz)
 * @param[in] rtol relative tolerance on normalised divergence
 * @param[in] atol absolute tolerance on normalised divergence
 * @param[in] maxiter maximum number of iterations
 *
 * Returns the actual number of iterations
 */
extern "C"
{
    int lippmann_schwinger_solve(float *lambda, float *mu, float *epsilon_bar,
                                 float *epsilon, float *sigma,
                                 int *cells,
                                 float *extents,
                                 float rtol, float atol, int maxiter);
}

#endif // LIPPMANN_SCHWINGER_HH
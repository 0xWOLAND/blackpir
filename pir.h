#ifndef PIR_H
#define PIR_H

#include <Eigen/Dense>
#include <random>
#include <optional>
#include <cstdint>

// Using aliases for convenience
using Matrix = Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<uint64_t, Eigen::Dynamic, 1>;
using Int8Matrix = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>;
using Int32Vector = Eigen::Matrix<int32_t, Eigen::Dynamic, 1>;

class SimplePIRParams {
public:
    size_t n;      // LWE dimension
    size_t m;      // Matrix dimension
    uint64_t q;    // LWE modulus
    uint64_t p;    // Plaintext modulus
    double std_dev; // Standard deviation for error

    SimplePIRParams(size_t n_, size_t m_, uint64_t q_, uint64_t p_, 
                   double std_dev_);
};

SimplePIRParams gen_params(size_t m, size_t n, uint32_t mod_power);
Matrix gen_matrix_a(size_t m, size_t n, uint64_t q_bits);
Vector gen_secret(uint64_t q_bits, size_t n);
std::pair<Matrix, Matrix> gen_hint(const SimplePIRParams& params, const Int8Matrix& db);
Vector encrypt(const SimplePIRParams& params, const Vector& v, const Matrix& a, const Vector& s);
std::pair<Vector, Vector> generate_query(const SimplePIRParams& params, const Vector& v, const Matrix& a);
Int32Vector process_query(const Int8Matrix& db, const Vector& query, uint64_t q);
Int32Vector recover(const Matrix& hint, const Vector& s, const Int32Vector& answer, const SimplePIRParams& params);

// Helper function for tests
bool is_approximately_equal(const Int32Vector& v1, const Int32Vector& v2, int32_t tolerance);

#endif
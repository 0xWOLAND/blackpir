#include "pir.h"
#include <random>
#include <stdexcept>
#include <cmath>
#include <iostream>

// Constructor implementation
SimplePIRParams::SimplePIRParams(size_t n_, size_t m_, uint64_t q_, uint64_t p_,
                               double std_dev_)
    : n(n_), m(m_), q(q_), p(p_), std_dev(std_dev_) {}

SimplePIRParams gen_params(size_t m, size_t n, uint32_t mod_power) {
    return SimplePIRParams(n, m, 
        static_cast<uint64_t>(1) << 32,  // q = 2^32
        static_cast<uint64_t>(1) << mod_power,
        0.2);
}

Matrix gen_matrix_a(size_t m, size_t n, uint64_t q_bits) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<uint64_t> dist(0, (static_cast<uint64_t>(1) << q_bits) - 1);
    
    Matrix result(m, n);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(m); ++i) {
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n); ++j) {
            result(i, j) = dist(rng);
        }
    }
    
    return result;
}

Vector gen_secret(uint64_t q_bits, size_t n) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<uint64_t> dist(0, (static_cast<uint64_t>(1) << q_bits) - 1);
    
    Vector result(n);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i) {
        result(i) = dist(rng);
    }
    
    return result;
}

std::pair<Matrix, Matrix> gen_hint(const SimplePIRParams& params, const Int8Matrix& db) {
    Matrix a = gen_matrix_a(params.m, params.n, 32);  // Match q_bits to q (2^32)
    
    // Convert db to uint64_t for matrix multiplication
    Matrix db_u64 = db.cast<uint64_t>();
    Matrix a_mod = a.unaryExpr([&](uint64_t x) { return x % params.q; });
    
    // Use Eigen matrix multiplication
    Matrix hint = (db_u64 * a_mod).unaryExpr([&](uint64_t x) { return x % params.q; });
    
    return {hint, a};
}

Vector encrypt(const SimplePIRParams& params, const Vector& v, const Matrix& a, const Vector& s) {
    uint64_t delta = params.q / params.p;

    // Generate Gaussian error
    std::random_device rd;
    std::normal_distribution<double> normal(0.0, params.std_dev);
    std::mt19937_64 rng(rd());
    
    Vector e(params.m);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(params.m); ++i) {
        int64_t sample = std::round(normal(rng));
        e(i) = (static_cast<uint64_t>(sample) * params.p) % params.q;
    }

    // Use Eigen matrix multiplication for As
    Vector as_prod = (a * s).unaryExpr([&](uint64_t x) { return x % params.q; });

    Vector result = (as_prod + e + delta * v).unaryExpr([&](uint64_t x) { return x % params.q; });

    return result;
}

std::pair<Vector, Vector> generate_query(const SimplePIRParams& params, const Vector& v, const Matrix& a) {
    if (static_cast<size_t>(v.size()) != params.m) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    
    Vector s = gen_secret(32, params.n);  // Match q_bits to q (2^32)
    Vector query = encrypt(params, v, a, s);
    return {s, query};
}

Int32Vector process_query(const Int8Matrix& db, const Vector& query, uint64_t q) {
    Int32Vector result(db.rows());
    for (Eigen::Index i = 0; i < db.rows(); ++i) {
        int32_t sum = 0;
        for (Eigen::Index j = 0; j < db.cols(); ++j) {
            sum += static_cast<int32_t>(db(i, j)) * static_cast<int32_t>(query(j));
        }
        result(i) = sum % q;
    }
    return result;
}

Int32Vector recover(const Matrix& hint, const Vector& s, const Int32Vector& answer, const SimplePIRParams& params) {
    uint64_t delta = params.q / params.p;
    uint64_t half_p = params.p >> 1;

    // Use Eigen matrix multiplication for hint * s
    Vector hint_s = (hint * s).unaryExpr([&](uint64_t x) { return x % params.q; });

    Int32Vector decrypted(answer.size());
    for (Eigen::Index i = 0; i < answer.size(); ++i) {
        uint64_t temp = (static_cast<uint64_t>(answer(i) >= 0 ? answer(i) : answer(i) + params.q) - hint_s(i) + params.q) % params.q;
        uint64_t raw = temp / delta;
        decrypted(i) = (raw >= half_p) ? (raw - params.p) : raw;
    }

    return decrypted;
}

bool is_approximately_equal(const Int32Vector& v1, const Int32Vector& v2, int32_t tolerance) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (Eigen::Index i = 0; i < v1.size(); ++i) {
        int32_t diff = std::abs(v1(i) - v2(i));
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}
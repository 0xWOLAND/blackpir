#include "pir.h"
#include <random>
#include <stdexcept>
#include <cmath>

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
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
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
    for (size_t i = 0; i < n; ++i) {
        result(i) = dist(rng);
    }
    
    return result;
}

std::pair<Matrix, Matrix> gen_hint(const SimplePIRParams& params, const Int8Matrix& db) {
    Matrix a = gen_matrix_a(params.m, params.n, 64);  // Use full 64 bits
    
    Matrix hint(db.rows(), a.cols());
    for (size_t i = 0; i < db.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            int32_t sum = 0;
            for (size_t k = 0; k < db.cols(); ++k) {
                sum += static_cast<int32_t>(db(i, k)) * static_cast<int32_t>(a(k, j) & 0xFFFFFFFF);
            }
            hint(i, j) = static_cast<uint64_t>(sum) % params.q;
        }
    }
    
    return {hint, a};
}

Vector encrypt(const SimplePIRParams& params, const Vector& v, const Matrix& a, const Vector& s) {
    uint64_t delta = params.q / params.p;

    // Generate Gaussian error
    std::random_device rd;
    std::normal_distribution<double> normal(0.0, params.std_dev);
    std::mt19937_64 rng(rd());
    
    Vector e(params.m);
    for (size_t i = 0; i < params.m; ++i) {
        int64_t sample = std::round(normal(rng));
        e(i) = ((static_cast<uint64_t>(sample) * params.p) % params.q);
    }

    // Compute As
    Vector as_prod(params.m);
    for (size_t i = 0; i < params.m; ++i) {
        as_prod(i) = 0;
        for (size_t j = 0; j < params.n; ++j) {
            as_prod(i) = (as_prod(i) + (a(i, j) * s(j)) % params.q) % params.q;
        }
    }

    Vector result(params.m);
    for (size_t i = 0; i < params.m; ++i) {
        uint64_t scaled_v = (delta * v(i)) % params.q;
        result(i) = (as_prod(i) + e(i) + scaled_v) % params.q;
    }

    return result;
}

std::pair<Vector, Vector> generate_query(const SimplePIRParams& params, const Vector& v, const Matrix& a) {
    if (v.size() != params.m) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    
    Vector s = gen_secret(64, params.n);  // Use full 64 bits
    Vector query = encrypt(params, v, a, s);
    return {s, query};
}

Int32Vector process_query(const Int8Matrix& db, const Vector& query, uint64_t q) {
    Int32Vector result(db.rows());
    for (size_t i = 0; i < db.rows(); ++i) {
        int32_t sum = 0;
        for (size_t j = 0; j < db.cols(); ++j) {
            int32_t db_val = static_cast<int32_t>(db(i, j));
            int32_t query_val = static_cast<int32_t>(query(j) & 0xFFFFFFFF);
            sum += db_val * query_val;
        }
        result(i) = sum;
    }
    return result;
}

Int32Vector recover(const Matrix& hint, const Vector& s, const Int32Vector& answer, const SimplePIRParams& params) {
    uint64_t delta = params.q / params.p;
    uint64_t half_p = params.p >> 1;

    Vector hint_s(answer.size());
    for (size_t i = 0; i < answer.size(); ++i) {
        hint_s(i) = 0;
        for (size_t j = 0; j < s.size(); ++j) {
            hint_s(i) = (hint_s(i) + (hint(i, j) * s(j)) % params.q) % params.q;
        }
    }

    Int32Vector decrypted(answer.size());
    for (size_t i = 0; i < answer.size(); ++i) {
        uint64_t temp = (static_cast<uint64_t>(answer(i)) + params.q - hint_s(i)) % params.q;
        uint64_t raw = temp / delta;
        decrypted(i) = (raw >= half_p) ? (raw - params.p) : raw;
    }

    return decrypted;
}

bool is_approximately_equal(const Int32Vector& v1, const Int32Vector& v2, int32_t tolerance) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        int32_t diff = std::abs(v1(i) - v2(i));
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}
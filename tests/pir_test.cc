#include <gtest/gtest.h>
#include "pir.h"
#include <random>

// Test fixture for PIR tests
class PIRTest : public ::testing::Test {
protected:
    void SetUp() override {
        matrix_height = 10;
        matrix_width = 10;
        mod_power = 17;
    }

    size_t matrix_height;
    size_t matrix_width;
    uint32_t mod_power;

    // Helper function to create random database
    Int8Matrix create_random_db() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<int8_t> dist(
            std::numeric_limits<int8_t>::min(),
            std::numeric_limits<int8_t>::max()
        );
        Int8Matrix db(matrix_height, matrix_width);
        for (size_t i = 0; i < matrix_height; ++i) {
            for (size_t j = 0; j < matrix_width; ++j) {
                db(i, j) = dist(rng);
            }
        }
        return db;
    }

    // Helper function to create random query vector
    Vector create_random_query() {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<int32_t> dist(
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max()
        );
        Vector v(matrix_width);
        for (size_t i = 0; i < matrix_width; ++i) {
            v(i) = static_cast<uint64_t>(dist(rng));
        }
        return v;
    }

    // Helper function to create row retrieval query
    Vector create_row_retrieval_query(size_t target_row) {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        Vector v(matrix_width);
        for (size_t i = 0; i < matrix_width; ++i) {
            v(i) = (i == target_row) ? 1 : 0;
        }
        return v;
    }
};

TEST_F(PIRTest, ComprehensiveTest) {
    // Create random database
    Int8Matrix db = create_random_db();
    
    {
        // Select random target row
        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<size_t> row_dist(0, matrix_width - 1);
        size_t target_row = row_dist(rng);
        Vector v = create_row_retrieval_query(target_row);

        // Generate expected result for row retrieval (target row)
        Int32Vector expected_row(matrix_height);
        for (size_t i = 0; i < matrix_height; ++i) {
            expected_row(i) = static_cast<int32_t>(db(i, target_row));
        }

        // Test PIR system with row retrieval query
        SimplePIRParams params = gen_params(matrix_height, 2048, mod_power);
        auto [hint, a] = gen_hint(params, db);
        auto [s, query] = generate_query(params, v, a);
        Int32Vector answer = process_query(db, query, params.q);
        Int32Vector result = recover(hint, s, answer, params);

        // Check results with tolerance for row retrieval
        int32_t tolerance_row = 10;
        ASSERT_TRUE(is_approximately_equal(expected_row, result, tolerance_row))
            << "Row retrieval test failed for row " << target_row;
    }
}
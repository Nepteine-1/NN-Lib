#include <NeuralNetwork/Matrix.hpp>
#include <gtest/gtest.h>

TEST(TestingConstructor, create_same_matrix) {
    std::vector<std::vector<float>> temp_matrix = {
        {5.f,5.f,5.f},
        {5.f,5.f,5.f},
        {5.f,5.f,5.f}};

    std::vector<float> temp = {0.5f, 0.4f, 0.5f};

    Matrix A(3,3,5), B(A), C(temp_matrix);
    EXPECT_EQ((A==B), true);
    EXPECT_EQ((A==C), true);
    EXPECT_EQ((B==C), true);
}

TEST(TestingConstructor, create_rand_matrix) {
    Matrix A(3,3);
    EXPECT_TRUE(A.row() == 3 && A.col() == 3);
    for(int i=0; i<A.row();i++) {
        for(int j=0; j<A.col();j++) {
            EXPECT_TRUE(A.getCoeff(i,j) >= 0.f && A.getCoeff(i,j) <= 1.f);
        }
    }
}

TEST(TestingAddOperator, valid_operation_true) {
    std::vector<std::vector<float>> temp_A = {
        {5.f,4.f,5.f},
        {7.f,0.f,6.f},
        {5.f,2.f,5.f}};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f,1.f},
        {7.f,1.f,7.f},
        {1.f,0.f,1.f}};

    std::vector<std::vector<float>> temp_res1 = {
        {6.f,4.f,6.f},
        {14.f,1.f,13.f},
        {6.f,2.f,6.f}};

    Matrix A(temp_A), B(temp_B);
    Matrix Res1(temp_res1);

    EXPECT_EQ((A+B) == Res1, true);
}

TEST(TestingAddOperator, valid_operation_false) {
    std::vector<std::vector<float>> temp_A = {
        {5.f,4.f,5.f},
        {7.f,0.f,6.f},
        {5.f,2.f,5.f}};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f,1.f},
        {7.f,1.f,7.f},
        {1.f,0.f,1.f}};

    std::vector<std::vector<float>> temp_res = {
        {6.3f,4.f,6.f},
        {14.3f,1.f,13.3f},
        {6.f,2.f,6.3f}};

    Matrix A(temp_A), B(temp_B);
    Matrix Res(temp_res);

    EXPECT_EQ((A+B) == Res, false);
}

TEST(TestingAddOperator, non_valid_operation) {
    std::vector<std::vector<float>> temp_A = {
        {5.f,4.f,5.f},
        {7.f,0.f,6.f}};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f},
        {7.f,1.f},
        {1.f,0.f}};

    Matrix A(temp_A), B(temp_B);

    EXPECT_THROW(A+B, std::runtime_error);
}

TEST(TestingAddOperator, one_empty_matrix) {
    std::vector<std::vector<float>> temp_A = {};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f},
        {7.f,1.f},
        {1.f,0.f}};

    Matrix A(temp_A), B(temp_B);

    EXPECT_THROW(A+B, std::runtime_error);
}

TEST(TestingAddOperator, both_empty_matrix) {
    std::vector<std::vector<float>> temp_A = {};
    std::vector<std::vector<float>> temp_B = {};

    Matrix A(temp_A), B(temp_B);

    EXPECT_NO_THROW(A+B);
    EXPECT_EQ((A+B == A), true);
}

TEST(TestingMulOperator, valid_operation_true) {
    std::vector<std::vector<float>> temp_A = {
        {5.8f,-4.6f,5.5f},
        {-7.4f,0.2f,6.7f},
        {5.4f,2.8f,-5.3f}};
    std::vector<std::vector<float>> temp_B = {
        {-1.8f,0.f,1.8f},
        {7.7f,-1.8f,-7.7f},
        {-1.8f,0.2f,1.3f}};

    std::vector<std::vector<float>> temp_res = {
        {-55.76f, 9.38f,  53.01f},
        {   2.8f, 0.98f,  -6.15f},
        { 21.38f, -6.1f, -18.73f}};

    Matrix A(temp_A), B(temp_B);
    Matrix Res(temp_res);

    EXPECT_EQ((A*B)== Res, true);
}

TEST(TestingMulOperator, valid_operation_false) {
    std::vector<std::vector<float>> temp_A = {
        {5.8f,4.6f,5.5f},
        {7.4f,0.2f,6.7f},
        {5.4f,2.8f,5.3f}};
    std::vector<std::vector<float>> temp_B = {
        {1.8f,0.f,1.8f},
        {7.7f,1.8f,7.7f},
        {1.8f,0.2f,1.3f}};

    std::vector<std::vector<float>> temp_res = {
        {10.46f,0.f,5.9f},
        {52.98f,0.36f,53.59f},
        {9.72f,0.46f,6.79f}};

    Matrix A(temp_A), B(temp_B);
    Matrix Res(temp_res);

    EXPECT_EQ((A*B) == Res, false);
}

TEST(TestingMulOperator, non_valid_operation) {
    std::vector<std::vector<float>> temp_A = {
        {5.f,4.f,5.f},
        {7.f,0.f,6.f}};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f},
        {7.f,1.f}};

    Matrix A(temp_A), B(temp_B);

    EXPECT_THROW(A*B, std::runtime_error);
}

TEST(TestingMulOperator, one_empty_matrix) {
    std::vector<std::vector<float>> temp_A = {};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f},
        {7.f,1.f},
        {1.f,0.f}};

    Matrix A(temp_A), B(temp_B);

    EXPECT_THROW(A*B, std::runtime_error);
}

TEST(TestingMulOperator, both_empty_matrix) {
    std::vector<std::vector<float>> temp_A = {};
    std::vector<std::vector<float>> temp_B = {};

    Matrix A(temp_A), B(temp_B);

    EXPECT_NO_THROW(A*B);
    EXPECT_EQ((A*B == A), true);
}

TEST(TestingHadamard, valid_operation_true) {
    std::vector<std::vector<float>> temp_A = {
        {-5.8f,4.6f,5.5f},
        {7.4f,0.2f,6.7f},
        {5.4f,2.8f,-5.3f}};
    std::vector<std::vector<float>> temp_B = {
        {7.4f,0.f,-6.7f},
        {-5.4f,2.8f,5.3f},
        {5.8f,0.f,5.5f}};

    std::vector<std::vector<float>> temp_res = {
        {-42.92f,0.f,-36.85f},
        {-39.96f,0.56f,35.51f},
        {31.32f,0.f,-29.15f}};

    Matrix A(temp_A), B(temp_B);
    Matrix Res(temp_res);

    EXPECT_EQ(Hadamard(A,B)==Res, true);
}

TEST(TestingHadamard, valid_operation_false) {
    std::vector<std::vector<float>> temp_A = {
        {-5.8f,4.6f,5.5f},
        {2.4f,0.2f,6.7f},
        {5.4f,2.4f,15.3f}};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f,-6.7f},
        {-5.4f,2.8f,5.3f},
        {5.7f,0.f,5.4f}};

    std::vector<std::vector<float>> temp_res = {
        {-42.92f,0.f,-36.85f},
        {-39.96f,0.56f,35.51f},
        {31.32f,0.f,-29.15f}};

    Matrix A(temp_A), B(temp_B);
    Matrix Res(temp_res);

    EXPECT_EQ(Hadamard(A,B)==Res, false);
}

TEST(TestingHadamard, non_valid_operation) {
    std::vector<std::vector<float>> temp_A = {
        {-5.8f,4.6f},
        {2.4f,0.2f},
        {5.4f,2.4f}};
    std::vector<std::vector<float>> temp_B = {
        {1.f,0.f,-6.7f},
        {-5.4f,2.8f,5.3f}};

    Matrix A(temp_A), B(temp_B);

    EXPECT_THROW(Hadamard(A,B), std::runtime_error);
}

TEST(TestingHadamard, one_empty_matrix) {
    std::vector<std::vector<float>> temp_A = {
        {-5.8f,4.6f},
        {2.4f,0.2f},
        {5.4f,2.4f}};
    std::vector<std::vector<float>> temp_B = {};

    Matrix A(temp_A), B(temp_B);

    EXPECT_THROW(Hadamard(A,B), std::runtime_error);
}

TEST(TestingHadamard, both_empty_matrix) {
    std::vector<std::vector<float>> temp_A = {};
    std::vector<std::vector<float>> temp_B = {};

    Matrix A(temp_A), B(temp_B);

    EXPECT_NO_THROW(Hadamard(A,B));
}

TEST(TestingHadamard, big_matrix) {
    Matrix A(50, 50, 4.5f), B(50,50, 3.7f);

    EXPECT_NO_THROW(Hadamard(A,B));
}

TEST(TestingHadamard, matrix_with_big_numbers) {
    Matrix A(3,3, 450000.f), B(3,3, 782000.f);

    EXPECT_NO_THROW(Hadamard(A,B));
    EXPECT_EQ(Hadamard(A,B)==Matrix(3,3,351900000000.f), true);
}

TEST(TestingHadamard, matrix_with_little_numbers) {
    Matrix A(3, 3, 0.005f), B(3,3, 0.004f);

    EXPECT_NO_THROW(Hadamard(A,B));
    EXPECT_EQ(Hadamard(A,B)==Matrix(3,3,0.00002f), true);
}

TEST(TestingHadamard, matrix_little_matrix) {
    Matrix A(1, 1, 7.f), B(1,1, 6.f);

    EXPECT_NO_THROW(Hadamard(A,B)==Matrix(1,1,42.f));
    EXPECT_EQ(Hadamard(A,B)==Matrix(1,1, 42.f), true);
}

TEST(TestingHadamard, zero_matrix) {
    Matrix A(3, 3, 1.f), B(3,3, 0.f);

    EXPECT_EQ(Hadamard(A,B)== B, true);
}

TEST(TestingMerge, valid_operation_true) {
    std::vector<std::vector<float>> temp_A = {
        {-5.8f,4.6f,5.5f},
        {2.4f,0.2f,6.7f},
        {5.4f,2.4f,15.3f}};

    std::vector<std::vector<float>> temp_B = {
        {-1.8f},
        {4.4f},
        {7.1f}};
    
    std::vector<std::vector<float>> temp_res = {
        {-7.6f,2.8f,3.7f},
        {6.8f,4.6f,11.1f},
        {12.5f,9.5f,22.4f}};
    Matrix A(temp_A), B(temp_B);
    Matrix Res(temp_res);
    A.merge(B);

    EXPECT_EQ(A== Res, true);
}

TEST(TestingMerge, non_valid_operation_1) {
    Matrix A(3,3), B(4,1);

    EXPECT_THROW(A.merge(B), std::runtime_error);
}

TEST(TestingMerge, non_valid_operation_2) {
    Matrix A(3,3), B(3,2);

    EXPECT_THROW(A.merge(B), std::runtime_error);
}

TEST(TestingMerge, null_matrix) {
    Matrix A(3,3), B(3,1,0), C(A);
    A.merge(B);

    EXPECT_EQ(A==C, true);
}

TEST(TestingMerge, one_line_matrix) {
    Matrix A(1,100, 5.f), B(1,1,1.f);
    A.merge(B);

    EXPECT_EQ(A==Matrix(1,100,6.f), true);
}

TEST(TestingMerge, one_column_matrix) {
    Matrix A(100,1, 5.f), B(100,1,1.f);
    A.merge(B);

    EXPECT_EQ(A==Matrix(100,1,6.f), true);
}

TEST(TestingTransposee, simple_transposee) {
    std::vector<std::vector<float>> temp_A = {
        {1.f,2.f,2.f},
        {3.f,1.f,2.f},
        {3.f,3.f,1.f}};

    std::vector<std::vector<float>> temp_Res = {
        {1.f,3.f,3.f},
        {2.f,1.f,3.f},
        {2.f,2.f,1.f}};

    Matrix A(temp_A), Res(temp_Res);
    A = A.transposee();

    EXPECT_EQ(A==Res, true);
}

TEST(TestingTransposee, not_square_matrix_1) {
    std::vector<std::vector<float>> temp_A = {
        {1.f,2.f,2.f,2.f},
        {3.f,1.f,2.f,2.f},
        {3.f,3.f,1.f,2.f}};

    std::vector<std::vector<float>> temp_Res = {
        {1.f,3.f,3.f},
        {2.f,1.f,3.f},
        {2.f,2.f,1.f},
        {2.f,2.f,2.f}};

    Matrix A(temp_A), Res(temp_Res);
    A = A.transposee();

    EXPECT_EQ(A==Res, true);
}

TEST(TestingTransposee, not_square_matrix_2) {
    std::vector<std::vector<float>> temp_A = {
        {1.f,2.f,2.f},
        {3.f,1.f,2.f},
        {3.f,3.f,1.f},
        {3.f,3.f,3.f}};

    std::vector<std::vector<float>> temp_Res = {
        {1.f,3.f,3.f,3.f},
        {2.f,1.f,3.f,3.f},
        {2.f,2.f,1.f,3.f}};

    Matrix A(temp_A), Res(temp_Res);
    A = A.transposee();

    EXPECT_EQ(A==Res, true);
}

TEST(TestingTransposee, very_small_matrix) {
    std::vector<std::vector<float>> temp_A = {
        {1.f}};

    std::vector<std::vector<float>> temp_Res = {
        {1.f}};

    Matrix A(temp_A), Res(temp_Res);
    A = A.transposee();

    EXPECT_EQ(A==Res, true);
}

TEST(TestingTransposee, column_matrix) {
    std::vector<std::vector<float>> temp_A = {
        {1.f},
        {5.f},
        {6.f},
        {3.f}};

    std::vector<std::vector<float>> temp_Res = {
        {1.f, 5.f, 6.f, 3.f}};

    Matrix A(temp_A), Res(temp_Res);
    A = A.transposee();

    EXPECT_EQ(A==Res, true);
}

TEST(TestingTransposee, row_matrix) {
    std::vector<std::vector<float>> temp_A = {
        {1.f, 5.f, 6.f, 3.f}};

    std::vector<std::vector<float>> temp_Res = {
        {1.f},
        {5.f},
        {6.f},
        {3.f}};

    Matrix A(temp_A), Res(temp_Res);
    A = A.transposee();

    EXPECT_EQ(A==Res, true);
}

TEST(TestingTransposee, empty_matrix) {
    std::vector<std::vector<float>> temp_A = {};

    std::vector<std::vector<float>> temp_Res = {};

    Matrix A(temp_A), Res(temp_Res);
    A = A.transposee();

    EXPECT_EQ(A==Res, true);
}

TEST(TestingTransposee, symetric_matrix) {
    std::vector<std::vector<float>> temp_A = {
        {1.f,2.f,-5.f},
        {2.f,4.f,3.f},
        {-5.f,3.f,1.f}};

    Matrix A(temp_A);

    EXPECT_EQ(A.transposee()==A, true);
}
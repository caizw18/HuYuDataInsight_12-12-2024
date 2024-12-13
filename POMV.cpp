#include <iostream>
#include <vector>
#include <Eigen/Dense>

double mean(const std::vector<double>& returns) {
    double sum = 0.0;
    for (double r : returns) sum += r;
    return sum / returns.size();
}

Eigen::MatrixXd covarianceMatrix(const std::vector<std::vector<double>>& returns) {
    int n = returns.size();
    int m = returns[0].size();
    Eigen::MatrixXd cov(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double covij = 0.0;
            double mean_i = mean(returns[i]);
            double mean_j = mean(returns[j]);
            for (int k = 0; k < m; k++) {
                covij += (returns[i][k] - mean_i) * (returns[j][k] - mean_j);
            }
            cov(i, j) = covij / (m - 1);
        }
    }

    return cov;
}

int main() {
    std::vector<std::vector<double>> returns = {
        {0.01, 0.02, -0.01},
        {0.03, -0.01, 0.04},
        {-0.02, 0.01, 0.03}
    };

    Eigen::MatrixXd cov = covarianceMatrix(returns);

    std::cout << "Covariance Matrix:\n" << cov << std::endl;

    return 0;
}
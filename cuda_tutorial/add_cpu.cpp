#include <cmath>
#include <iostream>
#include <vector>

// Step 2: Define add function
void add_cpu(std::vector<float> &c, const std::vector<float> &a, const std::vector<float> &b)
{
    // CPU use loop to calculate
    for (size_t i = 0; i < a.size(); i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    // Step 1: Prepare & initialize data
    constexpr size_t N = 1 << 20; // ~1M elements

    // Initialize data
    const std::vector<float> a(N, 1);
    const std::vector<float> b(N, 2);
    std::vector<float> c(N, 0);

    // Step 3: Call the cpu addition function
    add_cpu(c, a, b);

    // Step 4: Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(c[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;
}
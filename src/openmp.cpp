#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[])
{
    long long N = 100000000;
    int threads = 8;

    if (argc >= 2)
        N = stoll(argv[1]);

    if (argc >= 3)
        threads = stoi(argv[2]);

    omp_set_num_threads(threads);

    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N);

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    auto end = chrono::high_resolution_clock::now();

    double seconds =
        chrono::duration<double>(end - start).count();

    double milliseconds = seconds * 1000.0;

    cout << endl;
    cout << "========================================" << endl;
    cout << "       CPU VS GPU DATA PROCESSING" << endl;
    cout << "========================================" << endl;
    cout << "Method              : OpenMP CPU" << endl;
    cout << "Device              : CPU" << endl;
    cout << "Data Size           : " << N << " elements" << endl;
    cout << "CPU Threads         : " << threads << endl;
    cout << "Execution Time      : " << seconds << " seconds" << endl;
    cout << "Execution Time      : " << milliseconds << " ms" << endl;
    cout << "Result Check        : " << C[N - 1] << endl;
    cout << "========================================" << endl;
    cout << endl;

    return 0;
}
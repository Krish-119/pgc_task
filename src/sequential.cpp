#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main(int argc, char* argv[])
{
    long long N = 100000000;

    if (argc >= 2)
        N = atoll(argv[1]);

    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N);

    auto start = chrono::high_resolution_clock::now();

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
    cout << "Method              : Sequential CPU" << endl;
    cout << "Device              : CPU" << endl;
    cout << "Data Size           : " << N << " elements" << endl;
    cout << "CPU Threads         : 1" << endl;
    cout << "Execution Time      : " << seconds << " seconds" << endl;
    cout << "Execution Time      : " << milliseconds << " ms" << endl;
    cout << "Result Check        : " << C[N - 1] << endl;
    cout << "========================================" << endl;
    cout << endl;

    return 0;
}
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
void printCudaInfo();

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char** argv) {
    // default: arrays of 100M numbers
    int N = 100 * 1000 * 1000;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {{"arraysize", 1, 0, 'n'}, {"help", 0, 0, '?'}, {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                break;
            case '?':
            default:
                usage(argv[0]);
                return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    const float alpha = 2.0f;
    float* xarray = new float[N];
    float* yarray = new float[N];
    float* resultarray = new float[N];
    float* expected = new float[N];

    for (int i = 0; i < N; i++) {
        xarray[i] = yarray[i] = i % 10;
        expected[i] = alpha * xarray[i] + yarray[i];
    }

    printCudaInfo();

    printf("Running 3 timing tests (after warmup):\n");
    for (int i = 0; i < 4; i++) {
        // clean up result array before each run
        for (int j = 0; j < N; j++) {
            resultarray[j] = 0.0f;
        }
        if (i == 0) {
            printf("Warmup run (not timed)\n");
        } else {
            printf("Timed run %d\n", i);
        }
        saxpyCuda(N, alpha, xarray, yarray, resultarray);

        // Check results
        for (int i = 0; i < N; i++) {
            float expected = alpha * xarray[i] + yarray[i];
            if (resultarray[i] != expected) {
                fprintf(stderr, "Error: Device saxpy outputs incorrect result. A[%d] = %f, expecting %f.\n", i,
                        resultarray[i], expected);
                exit(1);
            }
        }
    }

    printf("SAXPY outputs are correct!\n");

    delete[] xarray;
    delete[] yarray;
    delete[] resultarray;
    delete[] expected;
    return 0;
}

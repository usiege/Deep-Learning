#include <stdio.h>
#include <stdlib.h>

double power(double base, int exponent)
{
    int result = base;
    int i;

    if(exponent == 0){
        return 1;
    }
    for (size_t i = 1; i < exponent; i++) {
        /* code */
        result = result * base;
    }
    return result;
}

int main(int argc, char const *argv[]) {
    /* code */
    if (argc < 3) {
        printf("Usage: %s base exponent \n", argv[0]);
        return 1;
    }
    double base = atof(argv[1]);
    int exponent = atoi(argv[2]);
    double result = power(base, exponent);
    printf("%g ^ %d is %g\n", base, exponent, result);
    return 0;
}


#include <math.h>
#include <stdlib.h>

int flanger_process_native(
    const double *x, double *out, long long n, long long size,
    double rate, double min_d, double max_d,
    double fb, double mix, double lfo_hz)
{
    double *buf = (double *)calloc((size_t)size, sizeof(double));
    if (buf == (void *)0) return -1;

    double center = 0.5 * (min_d + max_d);
    double amp = 0.5 * (max_d - min_d);
    double dg, wg;
    if (mix <= 0.0) { dg = 1.0; wg = 0.0; }
    else if (mix >= 1.0) { dg = 0.0; wg = 1.0; }
    else {
        double theta = 1.5707963267948966 * mix;  /* pi/2 */
        dg = cos(theta);
        wg = sin(theta);
    }
    double inc = 6.283185307179586 * lfo_hz / rate;  /* 2*pi */
    double phase = 0.0;
    long long w = 0;

    for (long long i = 0; i < n; ++i) {
        double delay = center + amp * sin(phase);
        if (delay < 1.0) delay = 1.0;  /* 不得 <1 采样的最终保险 */
        phase += inc;

        double rp = (double)w - delay;
        long long i0 = (long long)floor(rp);
        double frac = rp - (double)i0;
        long long j0 = i0 % size;
        if (j0 < 0) j0 += size;
        long long j1 = j0 + 1;
        if (j1 == size) j1 = 0;

        double a = buf[j0];
        double wet = a + (buf[j1] - a) * frac;
        double xi = x[i];
        buf[w] = xi + fb * wet;

        double r = dg * xi + wg * wet;
        if (r > 1.0) r = 1.0;
        else if (r < -1.0) r = -1.0;
        out[i] = r;

        ++w;
        if (w == size) w = 0;
    }
    free(buf);
    return 0;
}

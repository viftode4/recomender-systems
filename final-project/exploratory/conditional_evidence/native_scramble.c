/* Sequential same-rating bipartite 2-switches. All arithmetic is integer.
 * PRNG: SplitMix64; bounded draws use rejection, not biased modulo alone.
 * Edge slots start in row-major order. Rows/ratings remain fixed; columns move.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t next_u64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint64_t bounded(uint64_t *state, uint64_t count) {
    uint64_t threshold = (UINT64_C(0) - count) % count;
    uint64_t value;
    do { value = next_u64(state); } while (value < threshold);
    return value % count;
}

int ce_scramble(uint8_t *matrix, size_t height, size_t width,
                uint64_t seed, uint64_t *attempts, uint64_t *successes) {
    size_t cells, count = 0, counts[6] = {0}, starts[6] = {0}, next[6] = {0};
    size_t *rows, *cols, *groups;
    *attempts = *successes = 0;
    if (height && width > SIZE_MAX / height) return 2;
    cells = height * width;
    for (size_t n = 0; n < cells; n++) {
        if (matrix[n] > 5) return 1;
        if (matrix[n]) { count++; counts[matrix[n]]++; }
    }
    if (height < 2 || width < 2 || count < 2) return 0;
    if (count > SIZE_MAX / (3 * sizeof(size_t)) || count > UINT64_MAX / 20) return 2;
    rows = malloc(3 * count * sizeof(size_t));
    if (!rows) return 3;
    cols = rows + count;
    groups = cols + count;
    for (size_t rating = 1; rating < 6; rating++) {
        starts[rating] = starts[rating - 1] + counts[rating - 1];
        next[rating] = starts[rating];
    }
    size_t slot = 0;
    for (size_t n = 0; n < cells; n++) {
        uint8_t rating = matrix[n];
        if (rating) {
            rows[slot] = n / width;
            cols[slot] = n % width;
            groups[next[rating]++] = slot++;
        }
    }
    for (uint64_t step = 0; step < 20 * (uint64_t)count; step++) {
        size_t left = (size_t)bounded(&seed, (uint64_t)count);
        uint8_t rating = matrix[rows[left] * width + cols[left]];
        size_t right = groups[starts[rating] + (size_t)bounded(&seed, (uint64_t)counts[rating])];
        size_t a = rows[left], b = cols[left], c = rows[right], d = cols[right];
        (*attempts)++;
        if (left == right || a == c || b == d || matrix[a * width + d] || matrix[c * width + b]) continue;
        matrix[a * width + b] = matrix[c * width + d] = 0;
        matrix[a * width + d] = matrix[c * width + b] = rating;
        cols[left] = d;
        cols[right] = b;
        (*successes)++;
    }
    free(rows);
    return 0;
}

/* Test-only independent output-head reference contract. */
#ifndef TESTS_REFERENCE_LOGITS_H_INCLUDED
#define TESTS_REFERENCE_LOGITS_H_INCLUDED

#include <stddef.h>

int yvex_test_logits_reference_project(
    unsigned int qtype, const unsigned char *encoded, size_t encoded_bytes,
    unsigned long long rows, unsigned long long width,
    unsigned long long row_bytes, const float *hidden,
    float *logits, unsigned long long logits_capacity);

#endif

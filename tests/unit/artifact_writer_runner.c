/*
 * Returns nonzero on the first owned suite refusal. Fixture validation is not target-scale
 * complete emission.
 */
#include "tests/test.h"

int main(void)
{
    int rc;
    fprintf(stderr, "artifact test: gguf_writer_artifact\n");
    rc = yvex_test_gguf_writer_artifact();
    if (rc != 0)
        fprintf(stderr, "FAIL: gguf_writer_artifact exited %d\n", rc);
    return rc;
}

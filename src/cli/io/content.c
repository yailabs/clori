/* Own the reference client's bounded next-turn local attachment lifecycle. */
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include "src/cli/io/private.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct yvex_cli_content_stage {
    yvex_content_part parts[YVEX_CONTENT_MAX_PARTS - 1u];
    unsigned long long count;
};

static int stage_refuse(yvex_error *err, yvex_status status,
                        const char *reason)
{
    yvex_error_set(err, status, "client.attach", reason);
    return status;
}

static void content_classify(const unsigned char *head, size_t count,
                             yvex_content_kind *kind, const char **media_type)
{
    *kind = YVEX_CONTENT_FILE;
    *media_type = "application/octet-stream";
    if (count >= 8u && !memcmp(head, "\x89PNG\r\n\x1a\n", 8u)) {
        *kind = YVEX_CONTENT_IMAGE; *media_type = "image/png";
    } else if (count >= 3u && head[0] == 0xffu && head[1] == 0xd8u &&
               head[2] == 0xffu) {
        *kind = YVEX_CONTENT_IMAGE; *media_type = "image/jpeg";
    } else if (count >= 6u &&
               (!memcmp(head, "GIF87a", 6u) || !memcmp(head, "GIF89a", 6u))) {
        *kind = YVEX_CONTENT_IMAGE; *media_type = "image/gif";
    } else if (count >= 12u && !memcmp(head, "RIFF", 4u) &&
               !memcmp(head + 8u, "WEBP", 4u)) {
        *kind = YVEX_CONTENT_IMAGE; *media_type = "image/webp";
    } else if (count >= 12u && !memcmp(head, "RIFF", 4u) &&
               !memcmp(head + 8u, "WAVE", 4u)) {
        *kind = YVEX_CONTENT_AUDIO; *media_type = "audio/wav";
    } else if (count >= 4u && !memcmp(head, "fLaC", 4u)) {
        *kind = YVEX_CONTENT_AUDIO; *media_type = "audio/flac";
    } else if (count >= 3u && !memcmp(head, "ID3", 3u)) {
        *kind = YVEX_CONTENT_AUDIO; *media_type = "audio/mpeg";
    } else if (count >= 8u && !memcmp(head + 4u, "ftyp", 4u)) {
        *kind = YVEX_CONTENT_VIDEO; *media_type = "video/mp4";
    } else if (count >= 4u && head[0] == 0x1au && head[1] == 0x45u &&
               head[2] == 0xdfu && head[3] == 0xa3u) {
        *kind = YVEX_CONTENT_VIDEO; *media_type = "video/webm";
    } else if (count >= 5u && !memcmp(head, "%PDF-", 5u)) {
        *media_type = "application/pdf";
    }
}

int yvex_cli_content_stage_open(yvex_cli_content_stage **stage,
                                yvex_error *err)
{
    if (!stage || *stage)
        return stage_refuse(err, YVEX_ERR_INVALID_ARG,
                            "empty attachment stage output is required");
    *stage = calloc(1u, sizeof(**stage));
    if (!*stage)
        return stage_refuse(err, YVEX_ERR_NOMEM,
                            "attachment stage allocation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_cli_content_stage_close(yvex_cli_content_stage **stage)
{
    if (!stage || !*stage) return;
    free(*stage);
    *stage = NULL;
}

void yvex_cli_content_stage_clear(yvex_cli_content_stage *stage)
{
    if (stage) memset(stage, 0, sizeof(*stage));
}

unsigned long long yvex_cli_content_stage_count(
    const yvex_cli_content_stage *stage)
{
    return stage ? stage->count : 0u;
}

const yvex_content_part *yvex_cli_content_stage_parts(
    const yvex_cli_content_stage *stage)
{
    return stage ? stage->parts : NULL;
}

int yvex_cli_content_stage_attach(yvex_cli_content_stage *stage,
                                  const char *path,
                                  yvex_content_part *attached,
                                  yvex_error *err)
{
    unsigned char head[16];
    yvex_content_part *part;
    struct stat facts;
    const char *media_type;
    char *resolved;
    ssize_t got;
    int fd, rc;
    if (!stage || !path || !path[0] ||
        stage->count >= YVEX_CONTENT_MAX_PARTS - 1u)
        return stage_refuse(err, YVEX_ERR_INVALID_ARG,
                            "an existing bounded local attachment path is required");
    resolved = realpath(path, NULL);
    if (!resolved || strlen(resolved) >= YVEX_CONTENT_REFERENCE_CAP) {
        free(resolved);
        return stage_refuse(err, YVEX_ERR_INVALID_ARG,
                            "an existing bounded local attachment path is required");
    }
    fd = open(resolved, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &facts) != 0 || !S_ISREG(facts.st_mode) ||
        facts.st_size <= 0 ||
        (unsigned long long)facts.st_size > YVEX_CONTENT_LOCAL_MAX_BYTES) {
        if (fd >= 0) (void)close(fd);
        free(resolved);
        return stage_refuse(err, YVEX_ERR_BOUNDS,
                            "attachment must be a non-empty bounded regular file");
    }
    got = read(fd, head, sizeof(head));
    if (close(fd) != 0 || got < 0) {
        free(resolved);
        return stage_refuse(err, YVEX_ERR_IO,
                            "attachment metadata could not be read");
    }
    part = stage->parts + stage->count;
    memset(part, 0, sizeof(*part));
    part->schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
    part->storage = YVEX_CONTENT_LOCAL_FILE;
    part->byte_count = (unsigned long long)facts.st_size;
    content_classify(head, (size_t)got, &part->kind, &media_type);
    (void)snprintf(part->media_type, sizeof(part->media_type), "%s",
                   media_type);
    (void)snprintf(part->reference, sizeof(part->reference), "%s", resolved);
    free(resolved);
    rc = yvex_content_part_seal(part, err);
    if (rc != YVEX_OK) {
        memset(part, 0, sizeof(*part));
        return rc;
    }
    stage->count++;
    if (attached) *attached = *part;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cli_content_stage_turn(const yvex_cli_content_stage *stage,
                                const unsigned char *text,
                                unsigned long long text_bytes,
                                yvex_content_part parts[YVEX_CONTENT_MAX_PARTS],
                                unsigned long long *part_count,
                                yvex_error *err)
{
    yvex_content_part *text_part;
    unsigned long long count = yvex_cli_content_stage_count(stage);
    if (!stage || !text || !text_bytes || !parts || !part_count ||
        count >= YVEX_CONTENT_MAX_PARTS)
        return stage_refuse(err, YVEX_ERR_INVALID_ARG,
                            "staged content and non-empty text are required");
    memcpy(parts, stage->parts, (size_t)count * sizeof(*parts));
    text_part = parts + count;
    memset(text_part, 0, sizeof(*text_part));
    text_part->schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
    text_part->kind = YVEX_CONTENT_TEXT;
    text_part->storage = YVEX_CONTENT_INLINE;
    text_part->bytes = text;
    text_part->byte_count = text_bytes;
    (void)snprintf(text_part->media_type, sizeof(text_part->media_type),
                   "text/plain;charset=utf-8");
    if (yvex_content_part_seal(text_part, err) != YVEX_OK) return yvex_error_code(err);
    *part_count = count + 1u;
    yvex_error_clear(err);
    return YVEX_OK;
}

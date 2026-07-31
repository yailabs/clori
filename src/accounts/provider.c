/*
 * Observe configured provider accounts without exposing credentials.
 *
 * Credentials are never persisted and every child is waited. Account facts do not admit source
 * bytes or runtime capability.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/io.h>
#include <yvex/source.h>

static int account_refuse(yvex_error *err,
                          yvex_status status,
                          const char *where,
                          const char *message) {
    yvex_error_set(err, status, where, message);
    return status;
}

static int accounts_path_format(
    char *dst, size_t cap, yvex_error *err, const char *where, const char *a, const char *b) {
    int n;

    if (!dst || cap == 0u || !a) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "path output and base are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (b) {
        n = snprintf(dst, cap, "%s/%s", a, b);
    } else {
        n = snprintf(dst, cap, "%s", a);
    }
    if (n < 0 || (size_t)n >= cap) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, where, "path exceeds capacity");
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}

static void
accounts_json_field(FILE *fp, const char *indent, const char *key, const char *value, int comma) {
    fputs(indent, fp);
    fprintf(fp, "\"%s\": ", key);
    yvex_file_json_write_string(fp, value);
    fprintf(fp, "%s\n", comma ? "," : "");
}

static void
accounts_json_bool(FILE *fp, const char *indent, const char *key, int value, int comma) {
    fputs(indent, fp);
    fprintf(fp, "\"%s\": %s%s\n", key, value ? "true" : "false", comma ? "," : "");
}

static int accounts_state_path(char *out, size_t cap, yvex_error *err) {
    yvex_paths paths;
    int rc;

    memset(&paths, 0, sizeof(paths));
    rc = yvex_paths_default(&paths, err);
    if (rc != YVEX_OK)
        return rc;
    return accounts_path_format(out, cap, err, "accounts", paths.config_dir, "accounts.local.json");
}

typedef struct {
    yvex_account_provider provider;
    const char *name;
    const char *alias;
    const char *token_env;
    const char *cli_env;
    const char *binary;
    const char *legacy_binary;
    const char *missing_cli;
    const char *install_next;
    const char *auth_action;
} account_provider_definition;

static const account_provider_definition account_providers[] = {
    {YVEX_ACCOUNT_PROVIDER_HUGGINGFACE,
     "huggingface",
     "hf",
     "HF_TOKEN",
     "YVEX_HF_CLI",
     "hf",
     "huggingface-cli",
     "missing-huggingface-cli",
     "install Hugging Face CLI and retry",
     "whoami"},
    {YVEX_ACCOUNT_PROVIDER_GITHUB,
     "github",
     "gh",
     "GH_TOKEN",
     "YVEX_GH_CLI",
     "gh",
     "",
     "missing-github-cli",
     "install GitHub CLI and retry",
     "status"},
};

static const account_provider_definition *provider_definition(yvex_account_provider provider) {
    size_t i;

    for (i = 0u; i < sizeof(account_providers) / sizeof(account_providers[0]); ++i) {
        if (account_providers[i].provider == provider)
            return &account_providers[i];
    }
    return NULL;
}

/* Parse a provider name into the canonical provider identity enum. */
int yvex_account_provider_from_name(const char *name, yvex_account_provider *out) {
    size_t i;

    if (!name || !out)
        return 0;
    for (i = 0u; i < sizeof(account_providers) / sizeof(account_providers[0]); ++i) {
        if (strcmp(name, account_providers[i].name) == 0 ||
            strcmp(name, account_providers[i].alias) == 0) {
            *out = account_providers[i].provider;
            return 1;
        }
    }
    return 0;
}

/* Render one canonical provider identity as its stable lowercase name. */
const char *yvex_account_provider_name(yvex_account_provider provider) {
    const account_provider_definition *definition = provider_definition(provider);

    return definition ? definition->name : "unknown";
}

const char *yvex_account_default_token_env(yvex_account_provider provider) {
    const account_provider_definition *definition = provider_definition(provider);

    return definition ? definition->token_env : "";
}

static const char *accounts_provider_env_override(yvex_account_provider provider) {
    const account_provider_definition *definition = provider_definition(provider);

    return definition ? definition->cli_env : "";
}

static const char *accounts_provider_binary(yvex_account_provider provider) {
    const account_provider_definition *definition = provider_definition(provider);

    return definition ? definition->binary : "";
}

static const char *accounts_provider_legacy_binary(yvex_account_provider provider) {
    const account_provider_definition *definition = provider_definition(provider);

    return definition ? definition->legacy_binary : "";
}

static int accounts_find_in_path(const char *binary, char *out, size_t cap) {
    const char *path_env;
    const char *start;

    if (!binary || !binary[0] || !out || cap == 0u)
        return 0;
    if (strchr(binary, '/')) {
        if (access(binary, X_OK) == 0) {
            snprintf(out, cap, "%s", binary);
            return 1;
        }
        return 0;
    }
    path_env = getenv("PATH");
    start = path_env ? path_env : "";
    while (start && *start) {
        char candidate[YVEX_PATH_CAP];
        const char *end = strchr(start, ':');
        size_t len = end ? (size_t)(end - start) : strlen(start);
        const char *dir = len == 0u ? "." : start;
        int n;

        n = snprintf(candidate, sizeof(candidate), "%.*s/%s", (int)len, dir, binary);
        if (n >= 0 && (size_t)n < sizeof(candidate) && access(candidate, X_OK) == 0) {
            snprintf(out, cap, "%s", candidate);
            return 1;
        }
        start = end ? end + 1 : NULL;
    }
    return 0;
}

static int accounts_find_cli(yvex_account_provider provider,
                             const char *explicit_cli,
                             char *out,
                             size_t cap,
                             char *source,
                             size_t source_cap) {
    const char *env_name = accounts_provider_env_override(provider);
    const char *env_value = env_name && env_name[0] ? getenv(env_name) : NULL;
    const char *binary = accounts_provider_binary(provider);
    const char *legacy = accounts_provider_legacy_binary(provider);

    if (!out || cap == 0u || !source || source_cap == 0u)
        return 0;
    out[0] = '\0';
    source[0] = '\0';
    if (explicit_cli && explicit_cli[0]) {
        if (accounts_find_in_path(explicit_cli, out, cap)) {
            snprintf(source, source_cap, "--cli");
            return 1;
        }
        return 0;
    }
    if (env_value && env_value[0]) {
        if (accounts_find_in_path(env_value, out, cap)) {
            snprintf(source, source_cap, "%s", env_name);
            return 1;
        }
        return 0;
    }
    if (accounts_find_in_path(binary, out, cap)) {
        snprintf(source, source_cap, "PATH");
        return 1;
    }
    if (legacy && legacy[0] && accounts_find_in_path(legacy, out, cap)) {
        snprintf(source, source_cap, "PATH-legacy");
        return 1;
    }
    return 0;
}

static void accounts_first_line(char *value) {
    char *p;

    if (!value)
        return;
    for (p = value; *p; ++p) {
        if (*p == '\n' || *p == '\r') {
            *p = '\0';
            break;
        }
    }
    while (*value == ' ' || *value == '\t') {
        memmove(value, value + 1, strlen(value));
    }
    for (p = value + strlen(value); p > value && (p[-1] == ' ' || p[-1] == '\t'); --p) {
        p[-1] = '\0';
    }
}

/* Run one admitted provider CLI invocation and capture its bounded combined output. */
static int accounts_run_capture(
    const char *const *args, char *out, size_t out_cap, int *exit_code, yvex_error *err) {
    int pipefd[2];
    int devnull = -1;
    pid_t pid;
    int status;
    size_t used = 0u;

    if (!args || !args[0] || !out || out_cap == 0u || !exit_code) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "accounts_exec", "arguments and outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    out[0] = '\0';
    *exit_code = 1;
    if (pipe(pipefd) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "accounts_exec", "pipe failed: %s", strerror(errno));
        return YVEX_ERR_IO;
    }
    devnull = open("/dev/null", O_WRONLY);
    pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        if (devnull >= 0)
            close(devnull);
        yvex_error_setf(err, YVEX_ERR_IO, "accounts_exec", "fork failed: %s", strerror(errno));
        return YVEX_ERR_IO;
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        if (devnull >= 0)
            dup2(devnull, STDERR_FILENO);
        close(pipefd[1]);
        if (devnull >= 0)
            close(devnull);
        execv(args[0], (char *const *)args);
        _exit(127);
    }
    close(pipefd[1]);
    if (devnull >= 0)
        close(devnull);
    for (;;) {
        char buf[512];
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            close(pipefd[0]);
            yvex_error_setf(err, YVEX_ERR_IO, "accounts_exec", "read failed: %s", strerror(errno));
            return YVEX_ERR_IO;
        }
        if (n == 0)
            break;
        if (used + 1u < out_cap) {
            size_t take = (size_t)n;
            if (take > out_cap - used - 1u)
                take = out_cap - used - 1u;
            memcpy(out + used, buf, take);
            used += take;
            out[used] = '\0';
        }
    }
    close(pipefd[0]);
    if (waitpid(pid, &status, 0) < 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "accounts_exec", "waitpid failed: %s", strerror(errno));
        return YVEX_ERR_IO;
    }
    if (WIFEXITED(status))
        *exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status))
        *exit_code = 128 + WTERMSIG(status);
    else
        *exit_code = 1;
    return YVEX_OK;
}

int yvex_accounts_run_provider_command(const yvex_account_command_options *options,
                                       yvex_error *err) {
    int stdout_fd = -1;
    int stderr_fd = -1;
    int status;
    pid_t pid;

    if (!options || !options->args[0]) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "accounts_exec", "provider command arguments are required");
        return -1;
    }
    if (options->stdout_path && options->stdout_path[0]) {
        stdout_fd = open(options->stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0664);
        if (stdout_fd < 0) {
            yvex_error_setf(err,
                            YVEX_ERR_IO,
                            "accounts_exec",
                            "cannot open stdout log: %s",
                            options->stdout_path);
            return -1;
        }
    }
    if (options->stderr_path && options->stderr_path[0]) {
        stderr_fd = open(options->stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0664);
        if (stderr_fd < 0) {
            if (stdout_fd >= 0)
                close(stdout_fd);
            yvex_error_setf(err,
                            YVEX_ERR_IO,
                            "accounts_exec",
                            "cannot open stderr log: %s",
                            options->stderr_path);
            return -1;
        }
    }
    pid = fork();
    if (pid < 0) {
        if (stdout_fd >= 0)
            close(stdout_fd);
        if (stderr_fd >= 0)
            close(stderr_fd);
        yvex_error_setf(err, YVEX_ERR_IO, "accounts_exec", "fork failed: %s", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        if (stdout_fd >= 0)
            dup2(stdout_fd, STDOUT_FILENO);
        if (stderr_fd >= 0)
            dup2(stderr_fd, STDERR_FILENO);
        if (stdout_fd >= 0)
            close(stdout_fd);
        if (stderr_fd >= 0)
            close(stderr_fd);
        execv(options->args[0], (char *const *)options->args);
        _exit(127);
    }
    if (stdout_fd >= 0)
        close(stdout_fd);
    if (stderr_fd >= 0)
        close(stderr_fd);
    if (waitpid(pid, &status, 0) < 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "accounts_exec", "waitpid failed: %s", strerror(errno));
        return -1;
    }
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 1;
}

/* Derive non-secret account availability and identity facts from the provider environment. */
int yvex_account_observe(const yvex_account_observe_options *options,
                         yvex_account_observation *out,
                         yvex_error *err) {
    const account_provider_definition *definition;
    const char *token_env;
    const char *token_value;
    const char *provider_args[5];
    char capture[512];
    int exit_code = 1;
    int rc;

    if (!options || !out || options->provider == YVEX_ACCOUNT_PROVIDER_UNKNOWN) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "accounts", "provider and observation output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    definition = provider_definition(options->provider);
    memset(out, 0, sizeof(*out));
    out->provider = options->provider;
    snprintf(out->provider_name,
             sizeof(out->provider_name),
             "%s",
             yvex_account_provider_name(options->provider));
    token_env = options->token_env_name && options->token_env_name[0]
                    ? options->token_env_name
                    : yvex_account_default_token_env(options->provider);
    snprintf(out->token_env_name, sizeof(out->token_env_name), "%s", token_env);
    token_value = token_env && token_env[0] ? getenv(token_env) : NULL;
    out->token_env_present = token_value && token_value[0];
    out->token_value_redacted = out->token_env_present ? 1 : 0;
    out->raw_token_stored_by_yvex = 0;
    yvex_core_timestamp_utc(out->last_checked_at, sizeof(out->last_checked_at));
    if (accounts_state_path(out->state_path, sizeof(out->state_path), err) != YVEX_OK) {
        return yvex_error_code(err);
    }

    if (!accounts_find_cli(options->provider,
                           options->cli_override,
                           out->cli_path,
                           sizeof(out->cli_path),
                           out->cli_source,
                           sizeof(out->cli_source))) {
        snprintf(out->cli_status, sizeof(out->cli_status), "missing");
        snprintf(out->auth_state, sizeof(out->auth_state), "blocked");
        snprintf(out->credential_source,
                 sizeof(out->credential_source),
                 out->token_env_present ? "environment" : "none");
        snprintf(out->account_hint,
                 sizeof(out->account_hint),
                 out->token_env_present ? "env-token" : "unknown");
        snprintf(out->status, sizeof(out->status), "account-provider-blocked");
        snprintf(out->top_blocker, sizeof(out->top_blocker), "missing-%s-cli", out->provider_name);
        snprintf(out->top_blocker,
                 sizeof(out->top_blocker),
                 "%s",
                 definition ? definition->missing_cli : "missing-github-cli");
        snprintf(out->next,
                 sizeof(out->next),
                 "%s",
                 definition ? definition->install_next : "install GitHub CLI and retry");
        return YVEX_OK;
    }

    out->cli_present = 1;
    snprintf(out->cli_status, sizeof(out->cli_status), "present");
    if (out->token_env_present) {
        snprintf(out->auth_state, sizeof(out->auth_state), "env-token-present");
        snprintf(out->credential_source, sizeof(out->credential_source), "environment");
        snprintf(out->account_hint, sizeof(out->account_hint), "env:%s", out->token_env_name);
        snprintf(out->status, sizeof(out->status), "account-provider-pass");
        return YVEX_OK;
    }

    provider_args[0] = out->cli_path;
    provider_args[1] = "auth";
    provider_args[2] = definition ? definition->auth_action : "status";
    provider_args[3] = NULL;
    rc = accounts_run_capture(provider_args, capture, sizeof(capture), &exit_code, err);
    if (rc != YVEX_OK)
        return rc;
    out->command_exit_code = exit_code;
    if (exit_code == 0) {
        accounts_first_line(capture);
        snprintf(out->auth_state, sizeof(out->auth_state), "logged-in");
        snprintf(out->credential_source, sizeof(out->credential_source), "provider-store");
        snprintf(out->account_hint,
                 sizeof(out->account_hint),
                 "%.*s",
                 (int)sizeof(out->account_hint) - 1,
                 capture[0] ? capture : "provider-store");
        snprintf(out->status, sizeof(out->status), "account-provider-pass");
    } else {
        snprintf(out->auth_state, sizeof(out->auth_state), "not-logged-in");
        snprintf(out->credential_source, sizeof(out->credential_source), "none");
        snprintf(out->account_hint, sizeof(out->account_hint), "unknown");
        snprintf(out->status, sizeof(out->status), "account-provider-blocked");
        snprintf(out->top_blocker, sizeof(out->top_blocker), "provider-login-required");
        snprintf(out->next, sizeof(out->next), "yvex accounts login %s", out->provider_name);
    }
    return YVEX_OK;
}

/*
 * Atomically persist the observed non-secret account state.
 *
 * Writes only the explicit accounts provider destination through its transaction.
 */
int yvex_account_write_state(const yvex_account_observation *observations,
                             unsigned long count,
                             yvex_error *err) {
    char path[YVEX_PATH_CAP];
    FILE *fp;
    unsigned long i;
    int rc;

    if (!observations || count == 0u) {
        return account_refuse(err, YVEX_ERR_INVALID_ARG, "accounts", "observations are required");
    }
    rc = accounts_state_path(path, sizeof(path), err);
    if (rc != YVEX_OK)
        return rc;
    rc = yvex_core_mkdir_parent(path, "accounts", err);
    if (rc != YVEX_OK)
        return rc;
    fp = fopen(path, "wb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "accounts", "cannot open account state: %s", path);
        return YVEX_ERR_IO;
    }
    fprintf(fp, "{\n");
    accounts_json_field(fp, "  ", "schema", "yvex.accounts.v1", 1);
    fprintf(fp, "  \"providers\": [\n");
    for (i = 0; i < count; ++i) {
        const yvex_account_observation *obs = &observations[i];
        fprintf(fp, "    {\n");
        accounts_json_field(fp, "      ", "provider", obs->provider_name, 1);
        accounts_json_field(fp, "      ", "cli", obs->cli_path[0] ? obs->cli_path : "missing", 1);
        accounts_json_field(fp, "      ", "cli_status", obs->cli_status, 1);
        accounts_json_field(fp, "      ", "auth_state", obs->auth_state, 1);
        accounts_json_field(fp, "      ", "account_hint", obs->account_hint, 1);
        accounts_json_field(fp, "      ", "credential_source", obs->credential_source, 1);
        accounts_json_field(fp, "      ", "token_env_name", obs->token_env_name, 1);
        accounts_json_bool(fp, "      ", "token_value_redacted", obs->token_value_redacted, 1);
        accounts_json_field(fp, "      ", "last_checked_at", obs->last_checked_at, 1);
        accounts_json_bool(fp, "      ", "raw_token_stored_by_yvex", 0, 0);
        fprintf(fp, "    }%s\n", i + 1u == count ? "" : ",");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "accounts", "cannot close account state: %s", path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

int yvex_account_ensure(const yvex_account_ensure_options *options,
                        yvex_account_observation *out,
                        yvex_error *err) {
    yvex_account_observe_options observe;
    int rc;

    if (!options || !out) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "accounts", "ensure options and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(&observe, 0, sizeof(observe));
    observe.provider = options->provider;
    observe.cli_override = options->cli_override;
    observe.token_env_name = options->token_env_name;
    rc = yvex_account_observe(&observe, out, err);
    if (rc != YVEX_OK)
        return rc;
    if (!out->cli_present)
        return YVEX_OK;
    if (strcmp(out->auth_state, "logged-in") == 0 ||
        strcmp(out->auth_state, "env-token-present") == 0) {
        return yvex_account_write_state(out, 1u, err);
    }
    if (options->required || options->interactive == YVEX_ACCOUNT_INTERACTIVE_NEVER ||
        (options->interactive == YVEX_ACCOUNT_INTERACTIVE_AUTO &&
         (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)))) {
        snprintf(out->status, sizeof(out->status), "account-ensure-blocked");
        snprintf(out->top_blocker, sizeof(out->top_blocker), "provider-login-required");
        snprintf(out->next, sizeof(out->next), "yvex accounts login %s", out->provider_name);
        return YVEX_OK;
    }
    snprintf(out->status, sizeof(out->status), "account-ensure-login-required");
    snprintf(out->top_blocker, sizeof(out->top_blocker), "provider-login-required");
    snprintf(out->next, sizeof(out->next), "yvex accounts login %s", out->provider_name);
    return YVEX_OK;
}

#define _POSIX_C_SOURCE 200809L
#include <yvex/source.h>
#include <yvex/internal/cli_table.h>
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *const literal_pair_1[] = { "raw_token_stored_by_yvex: false",
    "boundary: local provider account state only, no tokens stored by YVEX"};

typedef enum {
    YVEX_ACCOUNTS_OUTPUT_NORMAL = 0,
    YVEX_ACCOUNTS_OUTPUT_TABLE,
    YVEX_ACCOUNTS_OUTPUT_AUDIT,
    YVEX_ACCOUNTS_OUTPUT_JSON
} yvex_accounts_output_mode;

typedef struct {
    yvex_accounts_output_mode output_mode;
    const char *token_env_name;
    const char *cli_override;
    const char *hostname;
    const char *git_protocol;
    const char *scopes;
    int force;
    int web;
    int device;
    int add_to_git_credential;
    int skip_ssh_key;
} yvex_accounts_cli_options;

static int accounts_run_foreground(const char *const *args, yvex_error *err)
{
    yvex_account_command_options command;
    unsigned int i;

    memset(&command, 0, sizeof(command));
    for (i = 0u; i + 1u < YVEX_ACCOUNT_ARG_CAP && args && args[i]; ++i) {
        command.args[i] = args[i];
    }
    command.args[i] = NULL;
    return yvex_accounts_run_provider_command(&command, err);
}

static int accounts_parse_output_mode(const char *value, yvex_accounts_output_mode *mode)
{
    if (!value || !mode) return 0;
    if (strcmp(value, "normal") == 0) {
        *mode = YVEX_ACCOUNTS_OUTPUT_NORMAL;
        return 1;
    }
    if (strcmp(value, "table") == 0) {
        *mode = YVEX_ACCOUNTS_OUTPUT_TABLE;
        return 1;
    }
    if (strcmp(value, "audit") == 0) {
        *mode = YVEX_ACCOUNTS_OUTPUT_AUDIT;
        return 1;
    }
    return 0;
}

static int accounts_parse_value(const char *command,
                                const char *flag,
                                int arg_count,
                                char **args,
                                int *index,
                                const char **value)
{
    if (*index + 1 >= arg_count) {
        yvex_cli_out_writef(stderr, "accounts %s: %s requires a value\n", command, flag);
        return 2;
    }
    *value = args[++(*index)];
    if (!*value || !(*value)[0] || strchr(*value, '\n') || strchr(*value, '\r')) {
        yvex_cli_out_writef(stderr, "accounts %s: %s value is empty or invalid\n", command, flag);
        return 2;
    }
    return 0;
}

static int accounts_parse_common(const char *command,
                                 int arg_count,
                                 char **args,
                                 int start,
                                 yvex_accounts_cli_options *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    options->output_mode = YVEX_ACCOUNTS_OUTPUT_NORMAL;
    for (i = start; i < arg_count; ++i) {
        const char *value = NULL;
        if (strcmp(args[i], "--output") == 0) {
            int rc = accounts_parse_value(command, "--output", arg_count, args, &i, &value);
            if (rc != 0) return rc;
            if (!accounts_parse_output_mode(value, &options->output_mode)) {
                yvex_cli_out_writef(stderr, "accounts %s: unsupported output mode: %s\n", command, value);
                return 2;
            }
        } else if (strcmp(args[i], "--token-env") == 0) {
            int rc = accounts_parse_value(command, "--token-env", arg_count, args, &i, &value);
            if (rc != 0) return rc;
            options->token_env_name = value;
        } else if (strcmp(args[i], "--cli") == 0) {
            int rc = accounts_parse_value(command, "--cli", arg_count, args, &i, &value);
            if (rc != 0) return rc;
            options->cli_override = value;
        } else if (strcmp(args[i], "--json") == 0) {
            options->output_mode = YVEX_ACCOUNTS_OUTPUT_JSON;
        } else {
            yvex_cli_out_writef(stderr, "accounts %s: unknown option: %s\n", command, args[i]);
            return 2;
        }
    }
    return 0;
}

static void accounts_print_observation_audit(const char *prefix,
                                             const yvex_account_observation *obs)
{
    yvex_cli_out_writef(stdout, "%sprovider: %s\n", prefix, obs->provider_name);
    yvex_cli_out_writef(stdout, "%scli: %s\n", prefix, obs->cli_path[0] ? obs->cli_path : "missing");
    yvex_cli_out_writef(stdout, "%scli_status: %s\n", prefix, obs->cli_status);
    yvex_cli_out_writef(stdout, "%sauth_state: %s\n", prefix, obs->auth_state);
    yvex_cli_out_writef(stdout, "%saccount_hint: %s\n", prefix, obs->account_hint);
    yvex_cli_out_writef(stdout, "%scredential_source: %s\n", prefix, obs->credential_source);
    yvex_cli_out_writef(stdout, "%stoken_env_name: %s\n", prefix, obs->token_env_name);
    yvex_cli_out_writef(stdout, "%stoken_value_redacted: %s\n", prefix, obs->token_value_redacted ? "true" : "false");
    yvex_cli_out_writef(stdout, "%sraw_token_stored_by_yvex: false\n", prefix);
    yvex_cli_out_writef(stdout, "%slast_checked_at: %s\n", prefix, obs->last_checked_at);
    if (obs->top_blocker[0]) yvex_cli_out_writef(stdout, "%stop_blocker: %s\n", prefix, obs->top_blocker);
    if (obs->next[0]) yvex_cli_out_writef(stdout, "%snext: %s\n", prefix, obs->next);
}

static void accounts_json_string_field(FILE *fp, const char *name,
                                       const char *value, int comma)
{
    if (comma) yvex_cli_out_fputs(",", fp);
    yvex_cli_out_writef(fp, "\"%s\":", name);
    yvex_cli_out_json_string(fp, value ? value : "");
}

static void accounts_observation_json(FILE *fp,
                                      const yvex_account_observation *obs)
{
    yvex_cli_out_fputs("{", fp);
    accounts_json_string_field(fp, "provider", obs->provider_name, 0);
    accounts_json_string_field(fp, "cli_path", obs->cli_path, 1);
    accounts_json_string_field(fp, "cli_source", obs->cli_source, 1);
    accounts_json_string_field(fp, "cli_status", obs->cli_status, 1);
    accounts_json_string_field(fp, "auth_state", obs->auth_state, 1);
    accounts_json_string_field(fp, "account", obs->account_hint, 1);
    accounts_json_string_field(fp, "credential_source", obs->credential_source, 1);
    accounts_json_string_field(fp, "token_env_name", obs->token_env_name, 1);
    accounts_json_string_field(fp, "login_method", obs->login_method, 1);
    accounts_json_string_field(fp, "status", obs->status, 1);
    accounts_json_string_field(fp, "blocker", obs->top_blocker, 1);
    accounts_json_string_field(fp, "next", obs->next, 1);
    accounts_json_string_field(fp, "state_path", obs->state_path, 1);
    accounts_json_string_field(fp, "last_checked_at", obs->last_checked_at, 1);
    yvex_cli_out_writef(
        fp,
        ",\"cli_present\":%s,\"token_env_present\":%s,"
        "\"token_value_redacted\":%s,\"raw_token_stored_by_yvex\":false,"
        "\"command_exit_code\":%d}",
        obs->cli_present ? "true" : "false",
        obs->token_env_present ? "true" : "false",
        obs->token_value_redacted ? "true" : "false", obs->command_exit_code);
}

static void accounts_collection_json(const char *schema,
                                     const yvex_account_observation *observations,
                                     size_t count)
{
    size_t index;
    yvex_cli_out_writef(stdout, "{\"schema\":\"%s\",\"providers\":[", schema);
    for (index = 0u; index < count; ++index) {
        if (index) yvex_cli_out_fputs(",", stdout);
        accounts_observation_json(stdout, &observations[index]);
    }
    yvex_cli_out_fputs("]}\n", stdout);
}

static void accounts_table(const yvex_account_observation *observations,
                           size_t count)
{
    static const yvex_cli_table_column columns[] = {
        {"PROVIDER", 11u, 16u, YVEX_CLI_TABLE_LEFT, 0},
        {"CLI", 7u, 12u, YVEX_CLI_TABLE_LEFT, 0},
        {"AUTH", 12u, 22u, YVEX_CLI_TABLE_LEFT, 0},
        {"ACCOUNT", 12u, 48u, YVEX_CLI_TABLE_LEFT, 1}
    };
    yvex_cli_table_cell cells[2][4];
    yvex_cli_table_row rows[2];
    size_t index;
    if (!observations || !count || count > 2u) return;
    for (index = 0u; index < count; ++index) {
        yvex_cli_table_tone auth_tone =
            !strcmp(observations[index].auth_state, "logged-in") ||
                    !strcmp(observations[index].auth_state, "env-token-present")
                ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_WARNING;
        cells[index][0] = (yvex_cli_table_cell){observations[index].provider_name,
                                                YVEX_CLI_TABLE_ACCENT};
        cells[index][1] = (yvex_cli_table_cell){observations[index].cli_status,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][2] = (yvex_cli_table_cell){observations[index].auth_state,
                                                auth_tone};
        cells[index][3] = (yvex_cli_table_cell){observations[index].account_hint,
                                                YVEX_CLI_TABLE_DIM};
        rows[index] = (yvex_cli_table_row){cells[index], NULL,
                                           YVEX_CLI_TABLE_PLAIN};
    }
    (void)yvex_cli_table_render(stdout, columns, 4u, rows, count);
}

static void accounts_print_single(const char *surface,
                                  const yvex_account_observation *obs,
                                  yvex_accounts_output_mode mode)
{
    if (mode == YVEX_ACCOUNTS_OUTPUT_JSON) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.provider.account.v1\",\"operation\":", stdout);
        yvex_cli_out_json_string(stdout, surface);
        yvex_cli_out_fputs(",\"provider\":", stdout);
        accounts_observation_json(stdout, obs);
        yvex_cli_out_fputs("}\n", stdout);
        return;
    }
    if (mode == YVEX_ACCOUNTS_OUTPUT_TABLE) {
        accounts_table(obs, 1u);
        yvex_cli_out_writef(stdout, "status: %s\n", obs->status);
        return;
    }
    if (mode == YVEX_ACCOUNTS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "accounts: %s\n", surface);
        accounts_print_observation_audit("", obs);
        yvex_cli_out_writef(stdout, "state_path: %s\n", obs->state_path);
        yvex_cli_out_writef(stdout, "boundary: local provider account state only; no tokens stored by YVEX\n");
        yvex_cli_out_writef(stdout, "status: %s\n", obs->status);
        return;
    }
    yvex_cli_out_writef(stdout, "provider: %s\n", obs->provider_name);
    yvex_cli_out_writef(stdout, "cli: %s\n", obs->cli_status);
    yvex_cli_out_writef(stdout, "auth_state: %s\n", obs->auth_state);
    yvex_cli_out_writef(stdout, "account: %s\n", obs->account_hint);
    if (obs->top_blocker[0]) yvex_cli_out_writef(stdout, "top_blocker: %s\n", obs->top_blocker);
    if (obs->next[0]) yvex_cli_out_writef(stdout, "next: %s\n", obs->next);
    yvex_cli_out_writef(stdout, "boundary: local provider account state only, no tokens stored by YVEX\n");
    yvex_cli_out_writef(stdout, "status: %s\n", obs->status);
}

static int command_accounts_providers(int arg_count, char **args)
{
    yvex_accounts_cli_options options;
    yvex_account_observation observations[2];
    yvex_account_observe_options observe;
    yvex_error err;
    int rc;

    rc = accounts_parse_common("providers", arg_count, args, 3, &options);
    if (rc != 0) return rc;
    yvex_error_clear(&err);
    memset(&observe, 0, sizeof(observe));
    observe.provider = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
    rc = yvex_account_observe(&observe, &observations[0], &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    observe.provider = YVEX_ACCOUNT_PROVIDER_GITHUB;
    rc = yvex_account_observe(&observe, &observations[1], &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));

    if (options.output_mode == YVEX_ACCOUNTS_OUTPUT_JSON) {
        accounts_collection_json("yvex.provider.list.v1", observations, 2u);
        return 0;
    } else if (options.output_mode == YVEX_ACCOUNTS_OUTPUT_TABLE) {
        accounts_table(observations, 2u);
    } else if (options.output_mode == YVEX_ACCOUNTS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "accounts: providers\n");
        accounts_print_observation_audit("provider_0_", &observations[0]);
        accounts_print_observation_audit("provider_1_", &observations[1]);
        yvex_cli_out_writef(stdout,
                           "boundary: provider inventory only; authentication is delegated to installed CLIs; "
                           "no tokens stored by YVEX\n");
    } else {
        yvex_cli_out_writef(stdout, "accounts: providers\n");
        yvex_cli_out_writef(stdout, "provider: huggingface alias=hf cli=%s\n", observations[0].cli_status);
        yvex_cli_out_writef(stdout, "provider: github alias=gh cli=%s\n", observations[1].cli_status);
        yvex_cli_out_writef(stdout,
                           "boundary: provider inventory only; authentication is delegated to installed CLIs; "
                           "no tokens stored by YVEX\n");
    }
    yvex_cli_out_writef(stdout, "status: account-providers\n");
    return 0;
}

static int command_accounts_status(int arg_count, char **args)
{
    yvex_accounts_cli_options options;
    yvex_account_observation observations[2];
    yvex_account_observe_options observe;
    yvex_error err;
    int rc;

    rc = accounts_parse_common("status", arg_count, args, 3, &options);
    if (rc != 0) return rc;
    yvex_error_clear(&err);
    memset(&observe, 0, sizeof(observe));
    observe.provider = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
    rc = yvex_account_observe(&observe, &observations[0], &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    observe.provider = YVEX_ACCOUNT_PROVIDER_GITHUB;
    rc = yvex_account_observe(&observe, &observations[1], &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    (void)yvex_account_write_state(observations, 2u, &err);

    if (options.output_mode == YVEX_ACCOUNTS_OUTPUT_JSON) {
        accounts_collection_json("yvex.provider.status.v1", observations, 2u);
        return 0;
    } else if (options.output_mode == YVEX_ACCOUNTS_OUTPUT_TABLE) {
        accounts_table(observations, 2u);
    } else if (options.output_mode == YVEX_ACCOUNTS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "accounts: status\n");
        accounts_print_observation_audit("provider_0_", &observations[0]);
        accounts_print_observation_audit("provider_1_", &observations[1]);
        yvex_cli_out_writef(stdout, "state_path: %s\n", observations[0].state_path);
        yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
    } else {
        yvex_cli_out_writef(stdout, "accounts: provider status\n");
        yvex_cli_out_writef(stdout, "huggingface: %s cli=%s account=%s\n",
               observations[0].auth_state, observations[0].cli_status, observations[0].account_hint);
        yvex_cli_out_writef(stdout, "github: %s cli=%s account=%s\n",
               observations[1].auth_state, observations[1].cli_status, observations[1].account_hint);
        yvex_cli_out_writef(stdout, "boundary: local provider account state only, no tokens stored by YVEX\n");
    }
    yvex_cli_out_writef(stdout, "status: accounts-status\n");
    return 0;
}

static int command_accounts_whoami(int arg_count, char **args)
{
    yvex_accounts_cli_options options;
    yvex_account_provider provider;
    yvex_account_observe_options observe;
    yvex_account_observation obs;
    yvex_error err;
    int rc;

    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "accounts whoami: PROVIDER is required\n");
        return 2;
    }
    if (!yvex_account_provider_from_name(args[3], &provider)) {
        yvex_cli_out_writef(stderr, "accounts whoami: unsupported provider: %s\n", args[3]);
        return 2;
    }
    rc = accounts_parse_common("whoami", arg_count, args, 4, &options);
    if (rc != 0) return rc;
    memset(&observe, 0, sizeof(observe));
    observe.provider = provider;
    observe.cli_override = options.cli_override;
    observe.token_env_name = options.token_env_name;
    yvex_error_clear(&err);
    rc = yvex_account_observe(&observe, &obs, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (strcmp(obs.auth_state, "logged-in") == 0 ||
        strcmp(obs.auth_state, "env-token-present") == 0) {
        snprintf(obs.status, sizeof(obs.status), "account-whoami-pass");
        (void)yvex_account_write_state(&obs, 1u, &err);
        accounts_print_single("whoami", &obs, options.output_mode);
        return 0;
    }
    snprintf(obs.status, sizeof(obs.status), "account-whoami-blocked");
    accounts_print_single("whoami", &obs, options.output_mode);
    return exit_for_status(YVEX_ERR_UNSUPPORTED);
}

static int accounts_parse_login_options(const char *command,
                                        int arg_count,
                                        char **args,
                                        int start,
                                        yvex_accounts_cli_options *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    options->output_mode = YVEX_ACCOUNTS_OUTPUT_NORMAL;
    for (i = start; i < arg_count; ++i) {
        const char *value = NULL;
        if (strcmp(args[i], "--output") == 0 ||
                   strcmp(args[i], "--token-env") == 0 ||
                   strcmp(args[i], "--cli") == 0 ||
                   strcmp(args[i], "--hostname") == 0 ||
                   strcmp(args[i], "--git-protocol") == 0 ||
                   strcmp(args[i], "--scopes") == 0) {
            const char *flag = args[i];
            int rc = accounts_parse_value(command, flag, arg_count, args, &i, &value);
            if (rc != 0) return rc;
            if (strcmp(flag, "--output") == 0) {
                if (!accounts_parse_output_mode(value, &options->output_mode)) {
                    yvex_cli_out_writef(stderr, "accounts %s: unsupported output mode: %s\n", command, value);
                    return 2;
                }
            } else if (strcmp(flag, "--token-env") == 0) {
                options->token_env_name = value;
            } else if (strcmp(flag, "--cli") == 0) {
                options->cli_override = value;
            } else if (strcmp(flag, "--hostname") == 0) {
                options->hostname = value;
            } else if (strcmp(flag, "--git-protocol") == 0) {
                if (strcmp(value, "https") != 0 && strcmp(value, "ssh") != 0) {
                    yvex_cli_out_writef(stderr, "accounts %s: --git-protocol requires https|ssh\n", command);
                    return 2;
                }
                options->git_protocol = value;
            } else {
                options->scopes = value;
            }
        } else if (strcmp(args[i], "--force") == 0) {
            options->force = 1;
        } else if (strcmp(args[i], "--web") == 0) {
            options->web = 1;
        } else if (strcmp(args[i], "--device") == 0) {
            options->device = 1;
        } else if (strcmp(args[i], "--add-to-git-credential") == 0) {
            options->add_to_git_credential = 1;
        } else if (strcmp(args[i], "--skip-ssh-key") == 0) {
            options->skip_ssh_key = 1;
        } else if (strcmp(args[i], "--json") == 0) {
            options->output_mode = YVEX_ACCOUNTS_OUTPUT_JSON;
        } else {
            yvex_cli_out_writef(stderr, "accounts %s: unknown option: %s\n", command, args[i]);
            return 2;
        }
    }
    return 0;
}

static int command_accounts_login(int arg_count, char **args)
{
    yvex_accounts_cli_options options;
    yvex_account_provider provider;
    yvex_account_observe_options observe;
    yvex_account_observation obs;
    yvex_error err;
    const char *token_env;
    const char *token_value = NULL;
    const char *provider_args[24];
    unsigned int n = 0u;
    int exit_code = 0;
    int rc;

    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "accounts login: PROVIDER is required\n");
        return 2;
    }
    if (!yvex_account_provider_from_name(args[3], &provider)) {
        yvex_cli_out_writef(stderr, "accounts login: unsupported provider: %s\n", args[3]);
        return 2;
    }
    rc = accounts_parse_login_options("login", arg_count, args, 4, &options);
    if (rc != 0) return rc;
    memset(&observe, 0, sizeof(observe));
    observe.provider = provider;
    observe.cli_override = options.cli_override;
    observe.token_env_name = options.token_env_name;
    yvex_error_clear(&err);
    rc = yvex_account_observe(&observe, &obs, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (!obs.cli_present) {
        snprintf(obs.status, sizeof(obs.status), "account-login-blocked");
        accounts_print_single("login", &obs, options.output_mode);
        return exit_for_status(YVEX_ERR_UNSUPPORTED);
    }
    token_env = options.token_env_name && options.token_env_name[0]
        ? options.token_env_name
        : yvex_account_default_token_env(provider);
    token_value = token_env && token_env[0] ? getenv(token_env) : NULL;

    if (provider == YVEX_ACCOUNT_PROVIDER_GITHUB && token_value && token_value[0]) {
        snprintf(obs.auth_state, sizeof(obs.auth_state), "env-token-present");
        snprintf(obs.credential_source, sizeof(obs.credential_source), "environment");
        snprintf(obs.account_hint, sizeof(obs.account_hint), "env:%s", token_env);
        snprintf(obs.login_method, sizeof(obs.login_method), "token-env");
        snprintf(obs.status, sizeof(obs.status), "account-login-pass");
        obs.token_value_redacted = 1;
        (void)yvex_account_write_state(&obs, 1u, &err);
        accounts_print_single("login", &obs, options.output_mode);
        return 0;
    }

    provider_args[n++] = obs.cli_path;
    provider_args[n++] = "auth";
    provider_args[n++] = "login";
    if (provider == YVEX_ACCOUNT_PROVIDER_HUGGINGFACE) {
        if (options.force) provider_args[n++] = "--force";
        if (options.add_to_git_credential) provider_args[n++] = "--add-to-git-credential";
        if (options.device) provider_args[n++] = "--device";
        if (token_value && token_value[0]) {
            provider_args[n++] = "--token";
            provider_args[n++] = token_value;
            obs.token_value_redacted = 1;
        }
    } else {
        if (options.web) provider_args[n++] = "--web";
        if (options.hostname) {
            provider_args[n++] = "--hostname";
            provider_args[n++] = options.hostname;
        }
        if (options.git_protocol) {
            provider_args[n++] = "--git-protocol";
            provider_args[n++] = options.git_protocol;
        }
        if (options.scopes) {
            provider_args[n++] = "--scopes";
            provider_args[n++] = options.scopes;
        }
        if (options.skip_ssh_key) provider_args[n++] = "--skip-ssh-key";
    }
    provider_args[n] = NULL;
    snprintf(obs.login_method, sizeof(obs.login_method), token_value && token_value[0] ? "token-env" : "provider-cli");
    exit_code = accounts_run_foreground(provider_args, &err);
    obs.command_exit_code = exit_code;
    if (exit_code != 0) {
        snprintf(obs.status, sizeof(obs.status), "account-login-fail");
        snprintf(obs.top_blocker, sizeof(obs.top_blocker), "provider-login-failed");
        snprintf(obs.next, sizeof(obs.next), "retry provider login");
        accounts_print_single("login", &obs, options.output_mode);
        return 1;
    }
    rc = yvex_account_observe(&observe, &obs, &err);
    if (rc == YVEX_OK && (strcmp(obs.auth_state, "logged-in") == 0 ||
                          strcmp(obs.auth_state, "env-token-present") == 0)) {
        snprintf(obs.status, sizeof(obs.status), "account-login-pass");
        (void)yvex_account_write_state(&obs, 1u, &err);
        accounts_print_single("login", &obs, options.output_mode);
        return 0;
    }
    snprintf(obs.status, sizeof(obs.status), "account-login-pass");
    snprintf(obs.auth_state, sizeof(obs.auth_state), "unknown");
    (void)yvex_account_write_state(&obs, 1u, &err);
    accounts_print_single("login", &obs, options.output_mode);
    return 0;
}

static int command_accounts_ensure(int arg_count, char **args)
{
    yvex_accounts_cli_options options;
    yvex_account_provider provider;
    yvex_account_ensure_options ensure;
    yvex_account_observation obs;
    yvex_error err;
    int rc;
    int i;

    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "accounts ensure: PROVIDER is required\n");
        return 2;
    }
    if (!yvex_account_provider_from_name(args[3], &provider)) {
        yvex_cli_out_writef(stderr, "accounts ensure: unsupported provider: %s\n", args[3]);
        return 2;
    }
    memset(&options, 0, sizeof(options));
    options.output_mode = YVEX_ACCOUNTS_OUTPUT_NORMAL;
    memset(&ensure, 0, sizeof(ensure));
    ensure.provider = provider;
    ensure.interactive = YVEX_ACCOUNT_INTERACTIVE_AUTO;
    for (i = 4; i < arg_count; ++i) {
        const char *value = NULL;
        if (strcmp(args[i], "--required") == 0) {
            ensure.required = 1;
        } else if (strcmp(args[i], "--output") == 0 ||
                   strcmp(args[i], "--token-env") == 0 ||
                   strcmp(args[i], "--interactive") == 0 ||
                   strcmp(args[i], "--cli") == 0) {
            const char *flag = args[i];
            rc = accounts_parse_value("ensure", flag, arg_count, args, &i, &value);
            if (rc != 0) return rc;
            if (strcmp(flag, "--output") == 0) {
                if (!accounts_parse_output_mode(value, &options.output_mode)) {
                    yvex_cli_out_writef(stderr, "accounts ensure: unsupported output mode: %s\n", value);
                    return 2;
                }
            } else if (strcmp(flag, "--token-env") == 0) {
                ensure.token_env_name = value;
            } else if (strcmp(flag, "--cli") == 0) {
                ensure.cli_override = value;
            } else if (strcmp(value, "auto") == 0) {
                ensure.interactive = YVEX_ACCOUNT_INTERACTIVE_AUTO;
            } else if (strcmp(value, "always") == 0) {
                ensure.interactive = YVEX_ACCOUNT_INTERACTIVE_ALWAYS;
            } else if (strcmp(value, "never") == 0) {
                ensure.interactive = YVEX_ACCOUNT_INTERACTIVE_NEVER;
            } else {
                yvex_cli_out_writef(stderr, "accounts ensure: --interactive requires auto|always|never\n");
                return 2;
            }
        } else if (strcmp(args[i], "--json") == 0) {
            options.output_mode = YVEX_ACCOUNTS_OUTPUT_JSON;
        } else {
            yvex_cli_out_writef(stderr, "accounts ensure: unknown option: %s\n", args[i]);
            return 2;
        }
    }
    yvex_error_clear(&err);
    rc = yvex_account_ensure(&ensure, &obs, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (strcmp(obs.auth_state, "logged-in") == 0 ||
        strcmp(obs.auth_state, "env-token-present") == 0) {
        snprintf(obs.status, sizeof(obs.status), "account-ensure-pass");
        accounts_print_single("ensure", &obs, options.output_mode);
        return 0;
    }
    if (!obs.status[0] || strcmp(obs.status, "account-provider-blocked") == 0) {
        snprintf(obs.status, sizeof(obs.status), "account-ensure-blocked");
    }
    accounts_print_single("ensure", &obs, options.output_mode);
    return exit_for_status(YVEX_ERR_UNSUPPORTED);
}

static int command_accounts_logout(int arg_count, char **args)
{
    yvex_accounts_cli_options options;
    yvex_account_provider provider;
    yvex_account_observe_options observe;
    yvex_account_observation obs;
    yvex_error err;
    const char *provider_args[5];
    int rc;
    int exit_code;

    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "accounts logout: PROVIDER is required\n");
        return 2;
    }
    if (!yvex_account_provider_from_name(args[3], &provider)) {
        yvex_cli_out_writef(stderr, "accounts logout: unsupported provider: %s\n", args[3]);
        return 2;
    }
    rc = accounts_parse_common("logout", arg_count, args, 4, &options);
    if (rc != 0) return rc;
    memset(&observe, 0, sizeof(observe));
    observe.provider = provider;
    observe.cli_override = options.cli_override;
    observe.token_env_name = options.token_env_name;
    yvex_error_clear(&err);
    rc = yvex_account_observe(&observe, &obs, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (!obs.cli_present) {
        snprintf(obs.status, sizeof(obs.status), "account-logout-blocked");
        accounts_print_single("logout", &obs, options.output_mode);
        return exit_for_status(YVEX_ERR_UNSUPPORTED);
    }
    provider_args[0] = obs.cli_path;
    provider_args[1] = "auth";
    provider_args[2] = "logout";
    provider_args[3] = NULL;
    exit_code = accounts_run_foreground(provider_args, &err);
    obs.command_exit_code = exit_code;
    snprintf(obs.status, sizeof(obs.status), exit_code == 0 ? "account-logout-pass" : "account-logout-fail");
    accounts_print_single("logout", &obs, options.output_mode);
    return exit_code == 0 ? 0 : 1;
}

int yvex_accounts_command(int arg_count, char **args)
{
    const char *sub;

    if (arg_count < 3) {
        yvex_cli_out_writef(stderr, "yvex: accounts requires providers, status, login, logout, whoami, or ensure\n");
        return 2;
    }
    sub = args[2];
    if (strcmp(sub, "providers") == 0) return command_accounts_providers(arg_count, args);
    if (strcmp(sub, "status") == 0) return command_accounts_status(arg_count, args);
    if (strcmp(sub, "login") == 0) return command_accounts_login(arg_count, args);
    if (strcmp(sub, "logout") == 0) return command_accounts_logout(arg_count, args);
    if (strcmp(sub, "whoami") == 0) return command_accounts_whoami(arg_count, args);
    if (strcmp(sub, "ensure") == 0) return command_accounts_ensure(arg_count, args);
    yvex_cli_out_writef(stderr, "yvex: unknown accounts subcommand: %s\n", sub);
    return 2;
}

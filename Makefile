# YVEX - Build and validation entrypoint
#
# Builds the YVEX C library, root binaries, CUDA kernel unit, and tests.
#
# Primary commands:
#   make info
#   make lib
#   make client
#   make package
#   make cuda-info
#   make cuda
#   make test-cuda
#   make check-cuda
#   make test
#   make test-core
#   make test-runtime
#   make test-runtime-deepseek-kv-live \
#       YVEX_RUNTIME_BINDING=/absolute/file.yvex-runtime-binding
#   make test-runtime-deepseek-prefill-live \
#       YVEX_RUNTIME_BINDING=/absolute/file.yvex-runtime-binding
#   make test-runtime-deepseek-moe-live \
#       YVEX_RUNTIME_BINDING=/absolute/file.yvex-runtime-binding
#   make test-runtime-deepseek-logits-live \
#       YVEX_RUNTIME_BINDING=/absolute/file.yvex-runtime-binding
#   make test-runtime-benchmark-chart-live YVEX_RUNTIME_BENCHMARK_DIR=/absolute/path \
#       YVEX_RUNTIME_BINDING=/absolute/file.yvex-runtime-binding
#   make update-runtime-benchmark-charts YVEX_RUNTIME_BENCHMARK_DIR=/absolute/empty/path \
#       YVEX_RUNTIME_BINDING=/absolute/file.yvex-runtime-binding
#   make test-runtime-sanitizers
#   make test-runtime-sanitizers-live
#   make test-cli
#   make smoke
#   make check
#   make clean
#
# Product topology:
#   - ./yvex is the sole public command and foreground model-server surface.

.DEFAULT_GOAL := all

.PHONY: all info lib client package generate-source-manifest \
	check-source-manifest generate-operator-registry \
	generate-command-migration \
	check-operator-registry test-operator-registry cuda-info cuda-kernels cuda test-cuda test-cuda-graph \
	test-cuda-native-sm121 \
	test-cuda-no-nvcc smoke-cuda check-cuda test test-core test-cli test-materialize \
	test-runtime-descriptor test-runtime-binding test-runtime-model-session \
	test-runtime-residency test-runtime-phases test-runtime-envelope \
	test-runtime-operator test-runtime-digests test-runtime-family-neutrality \
	test-runtime-state test-runtime-prefill test-runtime-profile test-runtime-benchmark \
	test-runtime-moe test-runtime-transformer test-runtime-decode test-runtime-logits \
	test-runtime-sampling test-runtime-speculation test-runtime-generation \
	test-runtime-tokenizer \
	test-runtime-benchmark-chart-live update-runtime-benchmark-charts \
	test-runtime-attention-live test-runtime-deepseek-kv-live \
	test-runtime-deepseek-prefill-live test-runtime-deepseek-moe-live \
	test-runtime-deepseek-transformer-live test-runtime-deepseek-decode-live \
	test-runtime-deepseek-logits-live test-runtime-deepseek-sampling-live \
	test-runtime-deepseek-tokenizer-live test-runtime-deepseek-generation-live \
	test-runtime test-runtime-asan test-runtime-asan-live \
	test-runtime-ubsan test-runtime-ubsan-live test-runtime-sanitizers \
	test-runtime-sanitizers-live test-materialize-live-plan \
	test-materialize-live test-minimax-audio-artifact-live \
	test-minimax-video-artifact-live test-minimax-text-conditioning-live \
	test-minimax-text-layer-live \
	test-attention test-attention-fixture-isolation \
	test-attention-live-plan test-attention-live test-attention-cli-live \
	test-attention-cuda test-quant test-quant-asan test-quant-ubsan \
	test-quant-sanitizers test-quant-live-plan test-quant-live \
	test-physical-variant-plan-deepseek-live test-quant-iq2-xxs-deepseek-live \
	test-artifact-emit-deepseek-variant-live test-materialize-deepseek-variant-live \
	test-runtime-deepseek-variant-generation-live \
	test-protocol test-runtime-host test-runtime-streaming test-repl \
	test-openai test-openai-sdk test-openai-bet-tennis test-openai-live \
	test-packaging test-product-topology test-runtime-client-refoundation-live \
	test-artifact-writer test-artifact-writer-fault test-artifact-live-plan \
	test-artifact-live-structure test-artifact-live test-transform-ir-live-plan \
	test-source-payload-live-plan test-source-payload-live test-gguf-artifact-abi \
	test-gguf-layout-integrity test-gguf-qtype-abi test-layout test-code-natural \
	test-project-control test-docs-surface \
	test-documentation-architecture test-surface test-source-ownership \
	test-repository-layout test-architecture-boundaries smoke check check-docs \
	check-guardrails clean

CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump
CUDA_HOME ?= /usr/local/cuda
NVCCFLAGS ?= -O3
CUDA_LDFLAGS ?=
YVEX_CUDA_ARCH ?= auto
NVCC_AVAILABLE := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)

CPPFLAGS ?= -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L -Iinclude -I.
YVEX_BUILD_COMMIT ?= $(shell git rev-parse --verify HEAD 2>/dev/null || printf unknown)
YVEX_BUILD_SOURCE_DELTA_IDENTITY ?= $(shell { \
	git diff --binary --no-ext-diff HEAD -- . 2>/dev/null; \
	git ls-files --others --exclude-standard 2>/dev/null | LC_ALL=C sort | \
		grep -v '__pycache__/' | grep -v '[.]pyc$$' | \
		while IFS= read -r path; do \
			printf 'untracked\t%s\t' "$$path"; stat -c 'mode=%a' "$$path"; \
			sha256sum "$$path"; \
			done; \
	} | sha256sum | cut -d' ' -f1)
YVEX_BUILD_SOURCE_STATE ?= $(if $(filter \
	e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,\
	$(YVEX_BUILD_SOURCE_DELTA_IDENTITY)),clean,dirty)
YVEX_BUILD_IDENTITY ?= $(shell printf '%s\n' \
	'cc=$(CC)' 'cc-version=$(shell $(CC) --version 2>/dev/null | head -1)' \
	'cc-target=$(shell $(CC) -dumpmachine 2>/dev/null)' \
	'cppflags=$(CPPFLAGS)' 'cflags=$(CFLAGS)' 'ldflags=$(LDFLAGS)' 'ldlibs=$(LDLIBS)' \
	'linker-version=$(shell $(CC) -Wl,--version 2>/dev/null | head -1)' \
	'nvcc=$(NVCC)' 'nvcc-version=$(shell $(NVCC) --version 2>/dev/null | tail -1)' \
	'nvccflags=$(NVCCFLAGS)' 'cuda-ldflags=$(CUDA_LDFLAGS)' 'cuda-arch=$(YVEX_CUDA_ARCH)' | \
	sha256sum | cut -d' ' -f1)
YVEX_BUILD_SOURCE_ROOT ?= $(shell pwd -P)
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -Wstrict-prototypes \
	-Wmissing-prototypes -Wmissing-declarations -Wshadow -Wformat=2 \
	-Wundef -Wvla -pthread
DEPFLAGS ?= -MMD -MP
LDFLAGS ?=
LDLIBS ?= -ldl -pthread -lm
TEST_CPPFLAGS := $(CPPFLAGS)

BUILD_DIR ?= build
OBJ_DIR ?= $(BUILD_DIR)/obj
LIB_DIR ?= $(BUILD_DIR)/lib
TEST_DIR ?= $(BUILD_DIR)/tests
BUILD_COMMIT_HEADER := $(BUILD_DIR)/generated/build_commit.h
SOURCE_OWNER_MANIFEST := config/source_owners.tsv
SOURCE_MANIFEST_GENERATOR := tools/generate_source_manifest.py
SOURCE_MANIFEST_MK := $(BUILD_DIR)/generated/sources.mk
OPERATOR_REGISTRY_SOURCE := config/operator/registry.json
OPERATOR_REGISTRY_GENERATOR := tools/generate_operator_registry.py
OPERATOR_REGISTRY_DIR := $(BUILD_DIR)/generated/operator
OPERATOR_REGISTRY_HEADER := $(OPERATOR_REGISTRY_DIR)/registry.h
OPERATOR_REGISTRY_C := $(OPERATOR_REGISTRY_DIR)/registry.c
OPERATOR_REGISTRY_IDENTITY := $(OPERATOR_REGISTRY_DIR)/registry.sha256
OPERATOR_REGISTRY_OBJ := $(OBJ_DIR)/generated/operator/registry.o
OPERATOR_AUDIT_ROOT := docs/audits/operator-surface-ec7dcc
OPERATOR_AUDIT_FILES := $(OPERATOR_AUDIT_ROOT)/commands.tsv \
	$(OPERATOR_AUDIT_ROOT)/flags.tsv $(OPERATOR_AUDIT_ROOT)/operations.tsv \
	$(OPERATOR_AUDIT_ROOT)/surfaces.tsv
OPERATOR_MIGRATION_DOC := docs/migrations/command-architecture-v1.md
DEEPSEEK_SOURCE ?= $(HOME)/lab/models/hf/deepseek/DeepSeek-V4-Flash-DSpark
DEEPSEEK_MODELS_ROOT ?= $(HOME)/lab/models/gguf
DEEPSEEK_SOURCE_MANIFEST ?= $(DEEPSEEK_MODELS_ROOT)/deepseek/deepseek-v4-flash-dspark-source-manifest.json
DEEPSEEK_OPERATOR_MODELS_ROOT ?= $(HOME)/lab/models
DEEPSEEK_SELECTED_ARTIFACT ?= $(DEEPSEEK_MODELS_ROOT)/deepseek/deepseek-v4-flash-dspark-bootstrap-q2-v1.gguf
YVEX_QUANT_DSPARK_PRESET ?= deepseek-v4-flash-dspark-bootstrap-q2-v1
YVEX_VARIANT_ARTIFACT ?=
YVEX_VARIANT_BINDING_DIR ?=
YVEX_RUNTIME_BENCHMARK_DIR ?=
YVEX_RUNTIME_BINDING ?=
YVEX_TOKENIZER_REFERENCE_PYTHON ?= /tmp/yvex-tokenizer-oracle/bin/python
PINNED_GGML_ROOT ?= /tmp/yvex-ggml-af97976
PINNED_GGML_BUILD ?= $(PINNED_GGML_ROOT)/build-yvex

LIBYVEX ?= $(LIB_DIR)/libyvex.a
YVEX_BIN ?= ./yvex

ifneq ($(strip $(MAKECMDGOALS)),clean)
include $(SOURCE_MANIFEST_MK)
endif

# Attention wrappers own a collision-free temporary root and delete only that
# root after validating its canonical parent and generated basename.
define ATTENTION_OWNED_TMP_BEGIN
tmp_parent=$${TMPDIR:-/tmp}; \
case "$$tmp_parent" in /*) ;; *) echo "attention temp parent must be absolute: $$tmp_parent" >&2; exit 1;; esac; \
test -d "$$tmp_parent" && test ! -L "$$tmp_parent"; \
tmp_parent=$$(cd "$$tmp_parent" && pwd -P); \
tmp_dir=$$(mktemp -d "$$tmp_parent/yvex-$$tmp_tag.XXXXXX"); \
case "$$tmp_dir" in "$$tmp_parent"/yvex-"$$tmp_tag".*) ;; *) echo "attention temp ownership mismatch: $$tmp_dir" >&2; exit 1;; esac; \
cleanup_attention_tmp() { \
	status=$$?; \
	trap - EXIT HUP INT TERM; \
	case "$$tmp_dir" in "$$tmp_parent"/yvex-"$$tmp_tag".*) ;; *) echo "refusing unowned attention cleanup: $$tmp_dir" >&2; exit 1;; esac; \
	if test -e "$$tmp_dir"; then \
		test -d "$$tmp_dir" && test ! -L "$$tmp_dir" || { echo "refusing unsafe attention cleanup: $$tmp_dir" >&2; exit 1; }; \
		find "$$tmp_dir" -xdev -mindepth 1 -delete || exit 1; \
		rmdir "$$tmp_dir" || exit 1; \
	fi; \
	exit $$status; \
}; \
trap cleanup_attention_tmp EXIT; \
trap 'exit 129' HUP; \
trap 'exit 130' INT; \
trap 'exit 143' TERM;
endef

YVEX_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(YVEX_SRCS)) $(OPERATOR_REGISTRY_OBJ)
CLIENT_LANE_OBJ := $(OBJ_DIR)/src/cli/io/client.o
CLIENT_PROTOCOL_OBJS := \
	$(OBJ_DIR)/src/core/status.o \
	$(OBJ_DIR)/src/core/sha256.o \
	$(OBJ_DIR)/src/core/json.o \
	$(OBJ_DIR)/src/provider/core.o \
	$(OBJ_DIR)/src/server/protocol.o \
	$(OBJ_DIR)/src/server/telemetry.o
OPENAI_ADAPTER_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(OPENAI_ADAPTER_SRCS))

CUDA_ARCH_FLAG := $(if $(filter auto,$(YVEX_CUDA_ARCH)),,-arch=$(YVEX_CUDA_ARCH))
CUDA_PTX := $(patsubst %.cu,$(OBJ_DIR)/%.ptx,$(CUDA_CU_SRCS))
CUDA_PTX_INC := $(OBJ_DIR)/generated/cuda_kernels_ptx.inc
CUDA_NATIVE_ARCH := $(filter sm_%,$(YVEX_CUDA_ARCH))
CUDA_CUBIN := $(if $(CUDA_NATIVE_ARCH),$(patsubst %.cu,$(OBJ_DIR)/%.cubin,$(CUDA_CU_SRCS)))
CUDA_CUBIN_INC := $(OBJ_DIR)/generated/cuda_kernels_cubin.inc

CORE_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(CORE_SRCS))
CUDA_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(CUDA_SRCS))
CORE_OBJS += $(CUDA_OBJS)

ifeq ($(NVCC_AVAILABLE),yes)
CPPFLAGS += -DYVEX_HAVE_CUDA_KERNEL_PTX=1
$(OBJ_DIR)/src/backend/cuda/capability.o: CPPFLAGS += -I$(OBJ_DIR)/generated
$(OBJ_DIR)/src/backend/cuda/capability.o: $(CUDA_PTX_INC)
ifneq ($(CUDA_NATIVE_ARCH),)
CPPFLAGS += -DYVEX_HAVE_CUDA_KERNEL_CUBIN=1
$(OBJ_DIR)/src/backend/cuda/capability.o: $(CUDA_CUBIN_INC)
endif
endif

$(OBJ_DIR)/src/cli/commands/graph.o: CPPFLAGS += -D_XOPEN_SOURCE=700 -I$(BUILD_DIR)/generated
$(OBJ_DIR)/src/cli/commands/graph.o: $(BUILD_COMMIT_HEADER)
$(OBJ_DIR)/src/runtime/benchmark.o: CPPFLAGS += -I$(BUILD_DIR)/generated
$(OBJ_DIR)/src/runtime/benchmark.o: $(BUILD_COMMIT_HEADER)
$(OBJ_DIR)/src/runtime/generation_context.o: CPPFLAGS += -I$(BUILD_DIR)/generated
$(OBJ_DIR)/src/runtime/generation_context.o: $(BUILD_COMMIT_HEADER)
OPERATOR_REGISTRY_CONSUMER_OBJS := $(OBJ_DIR)/src/cli/main.o \
	$(OBJ_DIR)/src/cli/io/client.o $(OBJ_DIR)/src/cli/io/out.o \
	$(OBJ_DIR)/src/cli/input/operator.o
$(OPERATOR_REGISTRY_CONSUMER_OBJS): CPPFLAGS += -I$(BUILD_DIR)/generated
$(OPERATOR_REGISTRY_CONSUMER_OBJS): $(OPERATOR_REGISTRY_HEADER)
$(OBJ_DIR)/src/cli/io/client.o: $(BUILD_COMMIT_HEADER)

TEST_RUNNER := $(TEST_DIR)/test
QUANT_TEST_RUNNER := $(TEST_DIR)/test_quant
ARTIFACT_TEST_RUNNER := $(TEST_DIR)/test_artifact_writer
SOURCE_PAYLOAD_LIVE_RUNNER := $(TEST_DIR)/source_payload_deepseek
QUANT_LIVE_RUNNER := $(TEST_DIR)/quant_deepseek
ARTIFACT_LIVE_RUNNER := $(TEST_DIR)/artifact_deepseek
MATERIALIZE_LIVE_RUNNER := $(TEST_DIR)/materialize_deepseek
MINIMAX_AUDIO_LIVE_RUNNER := $(TEST_DIR)/minimax_h3_audio
MINIMAX_VIDEO_LIVE_RUNNER := $(TEST_DIR)/minimax_h3_video
MINIMAX_TEXT_LIVE_RUNNER := $(TEST_DIR)/minimax_h3_text
ATTENTION_LIVE_RUNNER := $(TEST_DIR)/attention_deepseek
PREFILL_LIVE_RUNNER := $(TEST_DIR)/prefill_deepseek
MOE_LIVE_RUNNER := $(TEST_DIR)/moe_deepseek
TRANSFORMER_LIVE_RUNNER := $(TEST_DIR)/transformer_deepseek
DECODE_LIVE_RUNNER := $(TEST_DIR)/decode_deepseek
LOGITS_LIVE_RUNNER := $(TEST_DIR)/logits_deepseek
TOKENIZER_LIVE_RUNNER := $(TEST_DIR)/tokenizer_deepseek
GENERATION_LIVE_RUNNER := $(TEST_DIR)/generation_deepseek
OPENAI_FAKE_HOST := $(TEST_DIR)/openai_host
OPENAI_ADAPTER_HOST := $(TEST_DIR)/openai_adapter
OFFICIAL_GGUF_CHECKER := $(TEST_DIR)/ggml_gguf_check
CUDA_TEST_RUNNER := $(TEST_DIR)/test_cuda

TEST_UNIT_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(TEST_UNIT_SRCS))
TEST_REFERENCE_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(TEST_REFERENCE_SRCS))
TEST_MAIN_OBJ := $(OBJ_DIR)/tests/test.o

QUANT_TEST_UNIT_SRCS := \
	tests/unit/gguf_qtype_abi.c \
	tests/unit/source_payload.c \
	tests/unit/transform_ir.c \
	tests/unit/deepseek_tensor_coverage.c \
	tests/unit/quant_numeric.c \
	tests/unit/quant_execute.c \
	tests/unit/qtype_support.c \
	tests/unit/quant_policy.c \
	tests/unit/imatrix.c
QUANT_TEST_UNIT_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(QUANT_TEST_UNIT_SRCS))
QUANT_TEST_RUNNER_OBJ := $(OBJ_DIR)/tests/unit/quant_runner.o
ARTIFACT_TEST_RUNNER_OBJ := $(OBJ_DIR)/tests/unit/artifact_writer_runner.o

CUDA_TEST_UNIT_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(CUDA_TEST_UNIT_SRCS))
CUDA_TEST_MAIN_OBJ := $(OBJ_DIR)/tests/test_cuda.o

SOURCE_PAYLOAD_LIVE_OBJ := $(OBJ_DIR)/tests/live/source_payload_deepseek.o
QUANT_LIVE_OBJ := $(OBJ_DIR)/tests/live/quant_deepseek.o
ARTIFACT_LIVE_OBJ := $(OBJ_DIR)/tests/live/artifact_deepseek.o
MATERIALIZE_LIVE_OBJ := $(OBJ_DIR)/tests/live/materialize_deepseek.o
MINIMAX_AUDIO_LIVE_OBJ := $(OBJ_DIR)/tests/live/minimax_h3_audio.o
MINIMAX_VIDEO_LIVE_OBJ := $(OBJ_DIR)/tests/live/minimax_h3_video.o
MINIMAX_TEXT_LIVE_OBJ := $(OBJ_DIR)/tests/live/minimax_h3_text.o
ATTENTION_LIVE_OBJ := $(OBJ_DIR)/tests/live/attention_deepseek.o
PREFILL_LIVE_OBJ := $(OBJ_DIR)/tests/live/prefill_deepseek.o
MOE_LIVE_OBJ := $(OBJ_DIR)/tests/live/moe_deepseek.o
TRANSFORMER_LIVE_OBJ := $(OBJ_DIR)/tests/live/transformer_deepseek.o
DECODE_LIVE_OBJ := $(OBJ_DIR)/tests/live/decode_deepseek.o
LOGITS_LIVE_OBJ := $(OBJ_DIR)/tests/live/logits_deepseek.o
TOKENIZER_LIVE_OBJ := $(OBJ_DIR)/tests/live/tokenizer_deepseek.o
GENERATION_LIVE_OBJ := $(OBJ_DIR)/tests/live/generation_deepseek.o
OPENAI_FAKE_HOST_OBJ := $(OBJ_DIR)/tests/integration/openai_host.o
OPENAI_ADAPTER_HOST_OBJ := $(OBJ_DIR)/tests/integration/openai_adapter.o

RUNNER_OBJS := $(TEST_MAIN_OBJ) $(QUANT_TEST_RUNNER_OBJ) \
	$(ARTIFACT_TEST_RUNNER_OBJ) $(CUDA_TEST_MAIN_OBJ) \
	$(SOURCE_PAYLOAD_LIVE_OBJ) $(QUANT_LIVE_OBJ) $(ARTIFACT_LIVE_OBJ) \
	$(MATERIALIZE_LIVE_OBJ) $(MINIMAX_AUDIO_LIVE_OBJ) $(MINIMAX_VIDEO_LIVE_OBJ) \
	$(ATTENTION_LIVE_OBJ) \
	$(PREFILL_LIVE_OBJ) $(MOE_LIVE_OBJ) \
	$(TRANSFORMER_LIVE_OBJ) $(DECODE_LIVE_OBJ) $(LOGITS_LIVE_OBJ) $(TOKENIZER_LIVE_OBJ) \
	$(GENERATION_LIVE_OBJ) $(OPENAI_FAKE_HOST_OBJ) $(OPENAI_ADAPTER_HOST_OBJ)
DEPENDENCY_FILES := $(CORE_OBJS:.o=.d) $(YVEX_OBJS:.o=.d) \
	$(OPENAI_ADAPTER_OBJS:.o=.d) $(TEST_UNIT_OBJS:.o=.d) \
	$(TEST_REFERENCE_OBJS:.o=.d) $(QUANT_TEST_UNIT_OBJS:.o=.d) \
	$(CUDA_TEST_UNIT_OBJS:.o=.d) $(RUNNER_OBJS:.o=.d)

CLI_TEST := tests/cli.sh
CLIENT_CUTOVER_TEST := tests/client_cutover.sh
REPL_PTY_TEST := tests/repl_pty.sh
CLIENT_REFOUNDATION_LIVE_TEST := tests/live/client_refoundation.sh
OPENAI_INTEGRATION_TEST := tests/integration/openai.sh

info:
	@echo "yvex: native C/CUDA verified-artifact inference system"
	@echo "project_control: ROADMAP.md"
	@echo "interface: local client/protocol plus engine library ABI"
	@echo "library: libyvex.a"
	@echo "product: ./yvex server|run|chat|session|graph|artifact|quant|tokenizer"
	@echo "runtime_attention: CPU eager and admitted GB10 CUDA eager/piecewise/full implemented"
	@echo "benchmark_attention: identity-bound baseline, JSON/CSV, and deterministic SVG capability implemented"
	@echo "persistent_kv: session-owned DeepSeek CPU/CUDA state implemented"
	@echo "generation: implemented behind the admitted local runtime host"
	@echo "release: blocked"

all: generate-source-manifest generate-operator-registry lib client

generate-source-manifest: $(SOURCE_MANIFEST_MK)

check-source-manifest: $(SOURCE_MANIFEST_MK)
	python3 $(SOURCE_MANIFEST_GENERATOR) --manifest $(SOURCE_OWNER_MANIFEST) \
		--output $(SOURCE_MANIFEST_MK) --check
	@set -eu; \
	. tests/support/cleanup.sh; \
	first=$$(mktemp "$${TMPDIR:-/tmp}/yvex-sources.XXXXXX"); \
	second=$$(mktemp "$${TMPDIR:-/tmp}/yvex-sources.XXXXXX"); \
	trap 'yvex_test_cleanup "$$first" "$$second"' EXIT HUP INT TERM; \
	python3 $(SOURCE_MANIFEST_GENERATOR) --manifest $(SOURCE_OWNER_MANIFEST) --output "$$first"; \
	python3 $(SOURCE_MANIFEST_GENERATOR) --manifest $(SOURCE_OWNER_MANIFEST) --output "$$second"; \
	cmp "$$first" "$$second"

generate-operator-registry: $(OPERATOR_REGISTRY_HEADER) $(OPERATOR_REGISTRY_C) \
	$(OPERATOR_REGISTRY_IDENTITY)

generate-command-migration: $(OPERATOR_MIGRATION_DOC)

check-operator-registry: generate-operator-registry
	python3 $(OPERATOR_REGISTRY_GENERATOR) --registry $(OPERATOR_REGISTRY_SOURCE) \
		--output $(OPERATOR_REGISTRY_DIR) --audit-root $(OPERATOR_AUDIT_ROOT) \
		--migration-output $(OPERATOR_MIGRATION_DOC) --check
	@set -eu; \
	. tests/support/cleanup.sh; \
	first=$$(mktemp -d "$${TMPDIR:-/tmp}/yvex-operator-registry.XXXXXX"); \
	second=$$(mktemp -d "$${TMPDIR:-/tmp}/yvex-operator-registry.XXXXXX"); \
	trap 'yvex_test_cleanup "$$first" "$$second"' EXIT HUP INT TERM; \
	python3 $(OPERATOR_REGISTRY_GENERATOR) --registry $(OPERATOR_REGISTRY_SOURCE) --output "$$first"; \
	python3 $(OPERATOR_REGISTRY_GENERATOR) --registry $(OPERATOR_REGISTRY_SOURCE) --output "$$second"; \
	diff -ru "$$first" "$$second" >/dev/null

test-operator-registry: check-operator-registry client
	python3 tests/test_operator_registry.py

lib: $(LIBYVEX)

client: generate-operator-registry $(YVEX_BIN)

package: client config/package_manifest.tsv NOTICE.md
	@set -eu; \
	package_root='$(BUILD_DIR)/package'; \
	package_dir='$(BUILD_DIR)/package/product'; \
	if test -L "$$package_root"; then echo 'package root may not be a symlink' >&2; exit 1; fi; \
	if test -d "$$package_root"; then find "$$package_root" -depth -mindepth 1 -delete; fi; \
	mkdir -p "$$package_dir/bin" "$$package_dir/share/yvex"; \
	cp '$(YVEX_BIN)' "$$package_dir/bin/yvex"; \
	cp config/package_manifest.tsv NOTICE.md "$$package_dir/share/yvex/"; \
	printf '%s\n' 'yvex package: command and foreground model server' \
		> "$$package_dir/share/yvex/profile"; \
	commit=$$(git rev-parse HEAD); \
	client_sha=$$(sha256sum '$(YVEX_BIN)' | awk '{print $$1}'); \
	library_sha=$$(sha256sum '$(LIBYVEX)' | awk '{print $$1}'); \
	registry_identity=$$(cat '$(OPERATOR_REGISTRY_IDENTITY)'); \
	package_identity=$$(printf '%s\n' "$$commit" '8' 'cpu+cuda-dynamic' \
		"$$registry_identity" "$$client_sha" "$$library_sha" | \
		sha256sum | awk '{print $$1}'); \
	{ printf 'field\tvalue\n'; \
	  printf 'profile\tproduct\nsource_commit\t%s\n' "$$commit"; \
	  printf 'package_identity\t%s\n' "$$package_identity"; \
	  printf 'protocol_version\t%s\noperator_registry_identity\t%s\nbackend\t%s\n' \
		'8' "$$registry_identity" 'cpu+cuda-dynamic'; \
	  printf 'yvex_sha256\t%s\nlibyvex_sha256\t%s\n' \
		"$$client_sha" "$$library_sha"; \
	} > "$$package_dir/share/yvex/build.tsv"

cuda-info: $(YVEX_BIN)
	@echo "nvcc: $$(command -v $(NVCC) >/dev/null 2>&1 && command -v $(NVCC) || echo unavailable)"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "YVEX_CUDA_ARCH: $(YVEX_CUDA_ARCH)"
	$(YVEX_BIN) system cuda

cuda-kernels: $(CUDA_PTX_INC) $(if $(CUDA_NATIVE_ARCH),$(CUDA_CUBIN_INC))
	@echo "yvex cuda kernels: built from $(CUDA_CU_SRCS) arch=$(YVEX_CUDA_ARCH)"

cuda: cuda-kernels lib client $(CUDA_TEST_RUNNER)
	@echo "yvex cuda build: dynamic Driver API plus admitted PTX/native kernel image"

test-cuda: cuda
	$(YVEX_BIN) system cuda >/dev/null
	$(CUDA_TEST_RUNNER)

test-cuda-graph: cuda
	YVEX_CUDA_TEST_FILTER=graph $(CUDA_TEST_RUNNER)

test-cuda-native-sm121:
	$(MAKE) BUILD_DIR=build/sm121 YVEX_CUDA_ARCH=sm_121 \
		build/sm121/tests/test_cuda
	YVEX_REQUIRE_NATIVE_CUDA_TEST=sm_121 YVEX_CUDA_TEST_FILTER=info \
		build/sm121/tests/test_cuda

smoke-cuda: cuda $(YVEX_BIN)
	YVEX_BIN=$(YVEX_BIN) sh tests/cli/cuda.sh

check-cuda: cuda-info test-cuda smoke-cuda test-attention-cuda
	@echo "yvex check-cuda: ok"

test-cuda-no-nvcc: tests/test_cuda_failclosed.sh
	$(MAKE) BUILD_DIR=build/no-nvcc \
		YVEX_BIN=build/no-nvcc/yvex \
		NVCC=__yvex_nvcc_unavailable__ all
	YVEX_BIN=build/no-nvcc/yvex sh tests/test_cuda_failclosed.sh

test-core: $(TEST_RUNNER)
	$(TEST_RUNNER)

test-openai: $(TEST_RUNNER) $(OPENAI_ADAPTER_HOST) $(OPENAI_FAKE_HOST) \
	$(OPENAI_INTEGRATION_TEST)
	YVEX_OPENAI_ADAPTER=$(OPENAI_ADAPTER_HOST) sh tests/test_gateway_boundary.sh
	YVEX_TEST_FILTER=provider,protocol,openai,runtime_tokenizer $(TEST_RUNNER)
	YVEX_OPENAI_ADAPTER=$(OPENAI_ADAPTER_HOST) \
		YVEX_OPENAI_HOST=$(OPENAI_FAKE_HOST) sh $(OPENAI_INTEGRATION_TEST)

test-openai-sdk: test-openai
	YVEX_OPENAI_ADAPTER=$(OPENAI_ADAPTER_HOST) \
		YVEX_OPENAI_HOST=$(OPENAI_FAKE_HOST) \
		sh tests/integration/openai_sdk.sh

test-openai-bet-tennis: test-openai
	YVEX_OPENAI_ADAPTER=$(OPENAI_ADAPTER_HOST) \
		YVEX_OPENAI_HOST=$(OPENAI_FAKE_HOST) \
		sh tests/integration/bet_tennis.sh

test-openai-live: client
	YVEX_BIN=$(YVEX_BIN) \
		sh tests/live/openai.sh

test-cli: client $(CLI_TEST) $(CLIENT_CUTOVER_TEST)
	YVEX_BIN='$(YVEX_BIN)' sh $(CLI_TEST)
	YVEX_BIN='$(YVEX_BIN)' YVEX_CLIENT_LANE_OBJ='$(CLIENT_LANE_OBJ)' \
		sh $(CLIENT_CUTOVER_TEST)

test-materialize: $(TEST_RUNNER)
	$(TEST_RUNNER)

test-runtime-descriptor: $(TEST_RUNNER)
	YVEX_TEST_FILTER=materialization_runtime $(TEST_RUNNER)

test-runtime-binding: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_binding $(TEST_RUNNER)

# Runtime model/session lifecycle is exercised by the binding owner because the
# sealed model consumes one independently reopened binding.
test-runtime-model-session: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_binding $(TEST_RUNNER)

test-runtime-residency: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_binding $(TEST_RUNNER)

test-runtime-phases: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_state,deepseek_attention $(TEST_RUNNER)

test-runtime-envelope: $(TEST_RUNNER)
	YVEX_TEST_FILTER=deepseek_attention $(TEST_RUNNER)

test-runtime-operator: $(YVEX_BIN) tests/cli/attention_graph.sh
	YVEX_BIN=$(YVEX_BIN) sh tests/cli/attention_graph.sh

test-runtime-digests: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_state,runtime_benchmark,deepseek_attention $(TEST_RUNNER)

test-runtime-family-neutrality: $(TEST_RUNNER) test-architecture-boundaries
	YVEX_TEST_FILTER=runtime_binding $(TEST_RUNNER)

test-runtime-state: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_state $(TEST_RUNNER)

test-runtime-prefill: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_prefill $(TEST_RUNNER)

test-runtime-profile: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_profile $(TEST_RUNNER)

test-runtime-moe: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_moe $(TEST_RUNNER)

test-runtime-transformer: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_transformer $(TEST_RUNNER)

test-runtime-decode: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_decode $(TEST_RUNNER)

test-runtime-logits: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_logits $(TEST_RUNNER)

test-runtime-sampling: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_sampling $(TEST_RUNNER)

test-runtime-speculation: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_speculation $(TEST_RUNNER)

test-runtime-generation: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_generation $(TEST_RUNNER)

test-tokenizer: $(TEST_RUNNER)
	YVEX_TEST_FILTER=tokenizer,runtime_tokenizer,prompt $(TEST_RUNNER)

test-runtime-tokenizer: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_tokenizer $(TEST_RUNNER)

test-runtime-benchmark: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_benchmark $(TEST_RUNNER)

# This target retains identity-bound target-scale benchmark evidence in one
# caller-owned external directory. It never deletes, replaces, or tracks the
# baseline, reports, or SVG charts that it produces.
test-runtime-benchmark-chart-live: cuda
	@set -eu; \
	evidence_dir='$(YVEX_RUNTIME_BENCHMARK_DIR)'; \
	case "$$evidence_dir" in /*) ;; *) \
		echo "YVEX_RUNTIME_BENCHMARK_DIR must be an absolute directory" >&2; exit 2;; \
	esac; \
	test -d "$$evidence_dir" && test ! -L "$$evidence_dir" || { \
		echo "benchmark evidence directory must exist and must not be a symlink" >&2; exit 2; }; \
	evidence_dir=$$(cd "$$evidence_dir" && pwd -P); \
	repository_root=$$(pwd -P); \
	case "$$evidence_dir" in /|"$$repository_root"|"$$repository_root"/*) \
		echo "benchmark evidence directory must be outside the source repository" >&2; exit 2;; \
	esac; \
	test -z "$$(find "$$evidence_dir" -mindepth 1 -print -quit)" || { \
		echo "benchmark evidence directory must be empty" >&2; exit 2; }; \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	binding=$$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' \
		"$$binding"); \
	case "$$binding" in /|"$$repository_root"|"$$repository_root"/*) \
		echo "runtime binding must be outside the source repository" >&2; exit 2;; \
	esac; \
	for mode in eager piecewise full; do \
		$(YVEX_BIN) profile attention component --target deepseek4-v4-flash-dspark \
			--models-root "$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
			--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
			--backend cuda --phase decode --mode "$$mode" --scope full \
			--operation-scope release-attention-set --probe canonical \
			--warmup 3 --repeat 20 --progress off \
			--baseline "$$evidence_dir/$$mode.yvex-benchmark" --write-baseline \
			--chart "$$evidence_dir/$$mode.svg" --output json \
			>"$$evidence_dir/$$mode.json"; \
		$(YVEX_BIN) profile attention component --target deepseek4-v4-flash-dspark \
			--models-root "$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
			--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
			--backend cuda --phase decode --mode "$$mode" --scope full \
			--operation-scope release-attention-set --probe canonical \
			--warmup 3 --repeat 20 --progress off \
			--baseline "$$evidence_dir/$$mode.yvex-benchmark" \
			--chart "$$evidence_dir/$$mode-comparison.svg" --output csv \
			>"$$evidence_dir/$$mode-comparison.csv"; \
		test -s "$$evidence_dir/$$mode.yvex-benchmark"; \
		test -s "$$evidence_dir/$$mode.json"; \
		test -s "$$evidence_dir/$$mode-comparison.csv"; \
		test -s "$$evidence_dir/$$mode.svg"; \
		test -s "$$evidence_dir/$$mode-comparison.svg"; \
	done; \
	python3 tests/support/validate_runtime_benchmark.py "$$evidence_dir" "$$binding"; \
	printf 'runtime benchmark evidence retained: %s\n' "$$evidence_dir"

# Generate fresh identity-bound evidence outside the repository, validate the
# complete lane, then atomically publish only the six curated documentation
# charts. Raw baselines and JSON/CSV records remain operator-local.
update-runtime-benchmark-charts: test-runtime-benchmark-chart-live
	@set -eu; \
	evidence_dir='$(YVEX_RUNTIME_BENCHMARK_DIR)'; \
	evidence_dir=$$(cd "$$evidence_dir" && pwd -P); \
	repository_root=$$(pwd -P); \
	test -d docs && test ! -L docs; \
	mkdir -p docs/assets/benchmarks/attention; \
	for part in docs/assets docs/assets/benchmarks docs/assets/benchmarks/attention; do \
		test -d "$$part" && test ! -L "$$part" || { \
			echo "tracked chart destination must be a real repository directory" >&2; exit 2; }; \
	done; \
	chart_dir=$$(cd docs/assets/benchmarks/attention && pwd -P); \
	test "$$chart_dir" = "$$repository_root/docs/assets/benchmarks/attention" || { \
		echo "tracked chart destination escaped the repository contract" >&2; exit 2; }; \
	tmp_path=; \
	trap 'test -z "$$tmp_path" || test ! -e "$$tmp_path" || unlink -- "$$tmp_path"' \
		EXIT HUP INT TERM; \
	for name in eager eager-comparison piecewise piecewise-comparison full full-comparison; do \
		source_path="$$evidence_dir/$$name.svg"; \
		test -f "$$source_path" && test ! -L "$$source_path" || { \
			echo "validated chart is missing: $$source_path" >&2; exit 2; }; \
		tmp_path="$$chart_dir/.$$name.svg.tmp.$$$$"; \
		cp -- "$$source_path" "$$tmp_path"; \
		chmod 0644 "$$tmp_path"; \
		mv -f -- "$$tmp_path" "$$chart_dir/$$name.svg"; \
		tmp_path=; \
	done; \
	trap - EXIT HUP INT TERM; \
	printf 'tracked benchmark charts updated: %s\n' "$$chart_dir"

# Keep focused harness invocations serial even when the outer make uses -j.
test-runtime: $(TEST_RUNNER)
	YVEX_TEST_FILTER=protocol $(TEST_RUNNER)
	YVEX_TEST_FILTER=provider $(TEST_RUNNER)
	YVEX_TEST_FILTER=openai $(TEST_RUNNER)
	YVEX_TEST_FILTER=server $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_binding $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_decode $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_logits $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_sampling $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_speculation $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_generation $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_tokenizer $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_moe $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_transformer $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_prefill $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_profile $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_state $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_benchmark $(TEST_RUNNER)
	@! YVEX_TEST_FILTER=__unknown_runtime_test__ $(TEST_RUNNER) >/dev/null 2>&1
	@! YVEX_TEST_FILTER=runtime_benchmark,runtime_benchmark \
		$(TEST_RUNNER) >/dev/null 2>&1

test-protocol: $(TEST_RUNNER)
	YVEX_TEST_FILTER=protocol $(TEST_RUNNER)

test-runtime-host: $(TEST_RUNNER)
	YVEX_TEST_FILTER=server $(TEST_RUNNER)

test-runtime-streaming: $(TEST_RUNNER)
	YVEX_TEST_FILTER=protocol $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_generation $(TEST_RUNNER)

test-repl: client $(OPENAI_FAKE_HOST) $(REPL_PTY_TEST)
	YVEX_BIN='$(YVEX_BIN)' YVEX_TEST_HOST='$(OPENAI_FAKE_HOST)' \
		sh $(REPL_PTY_TEST)

test-packaging: package
	@test -x '$(BUILD_DIR)/package/product/bin/yvex'
	@test ! -e '$(BUILD_DIR)/package/product/bin/yvexd'
	@test ! -e '$(BUILD_DIR)/package/product/bin/yvex-openai'
	@test ! -e '$(BUILD_DIR)/package/product/bin/yvex-dev'
	@test -f '$(BUILD_DIR)/package/product/share/yvex/package_manifest.tsv'
	@test -f '$(BUILD_DIR)/package/product/share/yvex/build.tsv'
	@grep -F 'protocol_version	8' '$(BUILD_DIR)/package/product/share/yvex/build.tsv' >/dev/null
	@grep -F 'source_commit	' '$(BUILD_DIR)/package/product/share/yvex/build.tsv' >/dev/null
	@test ! -e '$(BUILD_DIR)/package/developer'

test-product-topology: all package tests/product_topology.sh
	YVEX_BIN='$(YVEX_BIN)' BUILD_DIR='$(BUILD_DIR)' \
		sh tests/product_topology.sh

test-runtime-client-refoundation-live: client $(CLIENT_REFOUNDATION_LIVE_TEST)
	YVEX_BIN='$(YVEX_BIN)' \
		YVEX_MODEL_ARTIFACT='$(YVEX_MODEL_ARTIFACT)' \
		YVEX_RUNTIME_BINDING='$(YVEX_RUNTIME_BINDING)' \
		sh $(CLIENT_REFOUNDATION_LIVE_TEST)

test-runtime-asan:
	@set -eu; \
	tmp_tag=runtime-asan; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	build_dir="$$tmp_dir/build"; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
		$(MAKE) BUILD_DIR="$$build_dir" \
			YVEX_BIN="$$build_dir/yvex" \
			NVCC=__yvex_nvcc_unavailable__ \
			CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=address,leak' \
			LDFLAGS='$(LDFLAGS) -fsanitize=address,leak' \
			test-runtime client test-openai; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
		YVEX_BIN="$$build_dir/yvex" \
		YVEX_TEST_HOST="$$build_dir/tests/openai_host" sh $(REPL_PTY_TEST); \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
		YVEX_TEST_FILTER=deepseek_attention \
		"$$build_dir/tests/test"

# The live sanitizer lane uses the admitted artifact but only one bounded CPU
# quick execution. It proves the main CLI reaches the instrumented production
# executor without pulling the target-scale CUDA/full validation into ASan.
test-runtime-asan-live: tests/cli/attention_graph.sh
	@set -eu; \
	tmp_tag=runtime-asan-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	build_dir="$$tmp_dir/build"; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
	$(MAKE) BUILD_DIR="$$build_dir" \
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=address,leak' \
		LDFLAGS='$(LDFLAGS) -fsanitize=address,leak' test-runtime; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
		YVEX_TEST_FILTER=deepseek_attention \
		"$$build_dir/tests/test"; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
	$(MAKE) BUILD_DIR="$$build_dir" YVEX_BIN="$$build_dir/yvex" \
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=address,leak' \
		LDFLAGS='$(LDFLAGS) -fsanitize=address,leak' cli; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
		YVEX_BIN="$$build_dir/yvex" YVEX_TEST_OUT_DIR="$$tmp_dir/output" \
		YVEX_ATTENTION_LIVE=1 YVEX_ATTENTION_CPU_QUICK_ONLY=1 \
		YVEX_ATTENTION_MODELS_ROOT="$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
		YVEX_ATTENTION_ARTIFACT="$(DEEPSEEK_SELECTED_ARTIFACT)" \
		sh tests/cli/attention_graph.sh

test-runtime-ubsan:
	@set -eu; \
	tmp_tag=runtime-ubsan; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	build_dir="$$tmp_dir/build"; \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		$(MAKE) BUILD_DIR="$$build_dir" \
			YVEX_BIN="$$build_dir/yvex" \
			NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=undefined \
			-fno-sanitize-recover=undefined' \
			LDFLAGS='$(LDFLAGS) -fsanitize=undefined' \
			test-runtime client test-openai; \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		YVEX_BIN="$$build_dir/yvex" \
		YVEX_TEST_HOST="$$build_dir/tests/openai_host" sh $(REPL_PTY_TEST); \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		YVEX_TEST_FILTER=deepseek_attention \
		"$$build_dir/tests/test"

test-runtime-ubsan-live: tests/cli/attention_graph.sh
	@set -eu; \
	tmp_tag=runtime-ubsan-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	build_dir="$$tmp_dir/build"; \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	$(MAKE) BUILD_DIR="$$build_dir" \
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=undefined \
			-fno-sanitize-recover=undefined' \
		LDFLAGS='$(LDFLAGS) -fsanitize=undefined' test-runtime; \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		YVEX_TEST_FILTER=deepseek_attention \
		"$$build_dir/tests/test"; \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	$(MAKE) BUILD_DIR="$$build_dir" YVEX_BIN="$$build_dir/yvex" \
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=undefined \
			-fno-sanitize-recover=undefined' \
		LDFLAGS='$(LDFLAGS) -fsanitize=undefined' cli; \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		YVEX_BIN="$$build_dir/yvex" YVEX_TEST_OUT_DIR="$$tmp_dir/output" \
		YVEX_ATTENTION_LIVE=1 YVEX_ATTENTION_CPU_QUICK_ONLY=1 \
		YVEX_ATTENTION_MODELS_ROOT="$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
		YVEX_ATTENTION_ARTIFACT="$(DEEPSEEK_SELECTED_ARTIFACT)" \
		sh tests/cli/attention_graph.sh

test-runtime-sanitizers:
	$(MAKE) test-runtime-asan
	$(MAKE) test-runtime-ubsan

test-runtime-sanitizers-live:
	$(MAKE) test-runtime-asan-live
	$(MAKE) test-runtime-ubsan-live

test-materialize-live-plan: $(MATERIALIZE_LIVE_RUNNER)
	$(MATERIALIZE_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-materialize-live: $(MATERIALIZE_LIVE_RUNNER)
	$(MATERIALIZE_LIVE_RUNNER) "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-minimax-audio-artifact-live: $(MINIMAX_AUDIO_LIVE_RUNNER)
	@test -n "$(MINIMAX_H3_AUDIO_ARTIFACT)" || { \
		echo "MINIMAX_H3_AUDIO_ARTIFACT is required" >&2; exit 2; }
	$(MINIMAX_AUDIO_LIVE_RUNNER) "$(MINIMAX_H3_AUDIO_ARTIFACT)"

test-minimax-video-artifact-live: $(MINIMAX_VIDEO_LIVE_RUNNER)
	@test -n "$(MINIMAX_H3_VIDEO_ARTIFACT)" || { \
		echo "MINIMAX_H3_VIDEO_ARTIFACT is required" >&2; exit 2; }
	$(MINIMAX_VIDEO_LIVE_RUNNER) "$(MINIMAX_H3_VIDEO_ARTIFACT)"

test-minimax-text-conditioning-live: $(MINIMAX_TEXT_LIVE_RUNNER)
	@test -n "$(MINIMAX_H3_TEXT_ARTIFACT)" || { \
		echo "MINIMAX_H3_TEXT_ARTIFACT is required" >&2; exit 2; }
	@test -n "$(MINIMAX_H3_TEXT_REFERENCE)" || { \
		echo "MINIMAX_H3_TEXT_REFERENCE is required" >&2; exit 2; }
	$(MINIMAX_TEXT_LIVE_RUNNER) "$(MINIMAX_H3_TEXT_ARTIFACT)" 1 \
		"$(BUILD_DIR)/tests/minimax_h3_text.f32" "$(MINIMAX_H3_TEXT_REFERENCE)"

test-minimax-text-layer-live: $(MINIMAX_TEXT_LIVE_RUNNER)
	@test -n "$(MINIMAX_H3_TEXT_ARTIFACT)" || { \
		echo "MINIMAX_H3_TEXT_ARTIFACT is required" >&2; exit 2; }
	@test -n "$(MINIMAX_H3_TEXT_LAYER_REFERENCE)" || { \
		echo "MINIMAX_H3_TEXT_LAYER_REFERENCE is required" >&2; exit 2; }
	$(MINIMAX_TEXT_LIVE_RUNNER) "$(MINIMAX_H3_TEXT_ARTIFACT)" 1 \
		"$(BUILD_DIR)/tests/minimax_h3_text_layer.f32" \
		"$(MINIMAX_H3_TEXT_LAYER_REFERENCE)" layer0

test-minimax-text-encoder-live: $(MINIMAX_TEXT_LIVE_RUNNER)
	@test -n "$(MINIMAX_H3_TEXT_ARTIFACT)" || { \
		echo "MINIMAX_H3_TEXT_ARTIFACT is required" >&2; exit 2; }
	@test -n "$(MINIMAX_H3_TEXT_ENCODER_REFERENCE)" || { \
		echo "MINIMAX_H3_TEXT_ENCODER_REFERENCE is required" >&2; exit 2; }
	$(MINIMAX_TEXT_LIVE_RUNNER) "$(MINIMAX_H3_TEXT_ARTIFACT)" 1 \
		"$(BUILD_DIR)/tests/minimax_h3_text_encoder.f32" \
		"$(MINIMAX_H3_TEXT_ENCODER_REFERENCE)" encoder50

test-attention: $(TEST_RUNNER) test-attention-fixture-isolation
	$(TEST_RUNNER)

test-attention-fixture-isolation: $(YVEX_BIN) tests/cli/attention_graph.sh
	@set -eu; \
	tmp_tag=attention-fixture-isolation; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	YVEX_BIN="$(YVEX_BIN)" YVEX_TEST_OUT_DIR="$$tmp_dir/first" \
		sh tests/cli/attention_graph.sh >"$$tmp_dir/first.log" 2>&1 & \
	first_pid=$$!; \
	YVEX_BIN="$(YVEX_BIN)" YVEX_TEST_OUT_DIR="$$tmp_dir/second" \
		sh tests/cli/attention_graph.sh >"$$tmp_dir/second.log" 2>&1 & \
	second_pid=$$!; \
	set +e; \
	wait $$first_pid; first_status=$$?; \
	wait $$second_pid; second_status=$$?; \
	set -e; \
	test $$first_status -eq 0 || { cat "$$tmp_dir/first.log" >&2; exit $$first_status; }; \
	test $$second_status -eq 0 || { cat "$$tmp_dir/second.log" >&2; exit $$second_status; }; \
	test -d "$$tmp_dir/first" && test -d "$$tmp_dir/second"; \
	test "$$tmp_dir/first" != "$$tmp_dir/second"; \
	cmp "$$tmp_dir/first.log" "$$tmp_dir/second.log"; \
	echo "attention fixture isolation: concurrent runs byte-identical"

test-attention-live-plan: $(ATTENTION_LIVE_RUNNER)
	$(ATTENTION_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-attention-live: $(ATTENTION_LIVE_RUNNER)
	$(ATTENTION_LIVE_RUNNER) "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

# This focused session/oracle lane refuses an absent binding rather than
# silently skipping the compilation-free runtime consumer.
test-runtime-attention-live: $(ATTENTION_LIVE_RUNNER)
	@set -eu; \
	tmp_tag=runtime-attention-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	YVEX_ATTENTION_RUNTIME_BINDING="$$binding" $(ATTENTION_LIVE_RUNNER) \
		"$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" \
		"$(DEEPSEEK_SOURCE_MANIFEST)" >"$$tmp_dir/first.out"; \
	YVEX_ATTENTION_RUNTIME_BINDING="$$binding" $(ATTENTION_LIVE_RUNNER) \
		"$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" \
		"$(DEEPSEEK_SOURCE_MANIFEST)" >"$$tmp_dir/second.out"; \
	cmp "$$tmp_dir/first.out" "$$tmp_dir/second.out"; \
	cat "$$tmp_dir/first.out"; \
	echo "runtime attention live repeat: byte-identical"

# This operator lane proves all 43 real layers consume session-persistent state
# on CPU and CUDA without retaining the external artifact or result files.
test-runtime-deepseek-kv-live: cuda
	@set -eu; \
	tmp_tag=runtime-deepseek-kv-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	for backend in cpu cuda; do \
		$(YVEX_BIN) execute attention state exercise --target deepseek4-v4-flash-dspark \
			--models-root "$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
			--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
			--backend "$$backend" --phase prefill --mode eager --scope full \
			--operation-scope core --tokens 2 --probe canonical --progress off \
			--output json >"$$tmp_dir/$$backend.json"; \
	done; \
	python3 -c 'import json,sys; rows=[json.load(open(p,encoding="utf-8")) for p in sys.argv[1:]]; \
		assert all(r["status"]=="complete" and r["layers_executed"]==43 and \
		r["bindings_executed"]==634 and r["swa_layers_executed"]==2 and \
		r["csa_layers_executed"]==21 and r["hca_layers_executed"]==20 and \
		r["state_layer_count"]==43 and r["state_prepared_layer_count"]==43 and \
		r["state_persistent"] and r["state_position_consistent"] and \
		r["state_read_after_write_verified"] and r["state_clear_reuse_verified"] and \
		r["persistent_kv_ready"] and not r["runtime_generation_ready"] for r in rows); \
		assert not rows[0]["state_cuda_ready"] and rows[1]["state_cuda_ready"]; \
		assert rows[0]["tensor_output_digest"]==rows[1]["tensor_output_digest"]; \
		assert rows[0]["state_delta_digest"]==rows[1]["state_delta_digest"]' \
		"$$tmp_dir/cpu.json" "$$tmp_dir/cuda.json"; \
	echo "persistent DeepSeek KV live: CPU/CUDA 43 layers and 634 bindings"

# This serial target proves tensor-file prefill, causality, rollback, and real CPU/CUDA state.
test-runtime-deepseek-prefill-live: cuda $(PREFILL_LIVE_RUNNER) $(YVEX_BIN)
	@set -eu; \
	tmp_tag=runtime-deepseek-prefill-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	activations="$$tmp_dir/deepseek-prefill.yvex-activations"; \
	$(PREFILL_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		"$$activations" >"$$tmp_dir/api.out"; \
	for backend in cpu cuda; do \
		$(YVEX_BIN) execute attention run --target deepseek4-v4-flash-dspark \
			--models-root "$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
			--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
			--backend "$$backend" --phase prefill --mode eager --scope full \
			--operation-scope core --input tensor-file --input-file "$$activations" \
			--chunk-tokens 1 --context-capacity 2 --progress off --output json \
			>"$$tmp_dir/$$backend.json"; \
	done; \
	python3 -c 'import json,sys; rows=[json.load(open(p,encoding="utf-8")) for p in sys.argv[1:]]; \
		assert all(r["status"]=="complete" and r["input_class"]=="typed_activation_tensor_file" \
		and r["layers_executed"]==43 and r["bindings_executed"]==634 \
		and r["swa_layers_executed"]==2 and r["csa_layers_executed"]==21 \
		and r["hca_layers_executed"]==20 and r["prefill_chunk_count"]==2 \
		and r["committed_prefix"]==2 and r["activation_prefill_ready"] \
		and r["prefill_persistent_state_ready"] and not r["full_model_prefill_ready"] \
		and r["persistent_kv_ready"] and not r["transformer_ready"] \
		and not r["runtime_generation_ready"] for r in rows); \
		assert rows[0]["tensor_output_digest"]==rows[1]["tensor_output_digest"]; \
		assert rows[0]["state_delta_digest"]==rows[1]["state_delta_digest"]' \
		"$$tmp_dir/cpu.json" "$$tmp_dir/cuda.json"; \
	cat "$$tmp_dir/api.out"; \
	echo "production DeepSeek activation prefill live: CPU/CUDA 43 layers and 634 bindings"

# This serial target proves real selected-expert CPU/CUDA and the full CUDA operator path.
test-runtime-deepseek-moe-live: cuda $(MOE_LIVE_RUNNER) $(YVEX_BIN)
	@set -eu; \
	tmp_tag=runtime-deepseek-moe-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	input="$$tmp_dir/deepseek-moe.yvex-moe-input"; \
	$(MOE_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" "$$input" \
		>"$$tmp_dir/api.out"; \
	$(YVEX_BIN) execute moe --target deepseek4-v4-flash-dspark \
		--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --input tensor-file --input-file "$$input" \
		--scope full --progress off --output json >"$$tmp_dir/cuda.json"; \
	python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
		assert r["status"]=="complete" and r["backend"]=="cuda" \
		and r["layers"]==43 and r["layers_executed"]==43 \
		and r["hash_router_executions"]==3 and r["learned_router_executions"]==40 \
		and r["routed_expert_executions"]==258 and r["shared_expert_executions"]==43 \
		and r["expert_subviews_accessed"]==774 and r["moe_block_ready"] \
		and not r["moe_prefill_composed"] and not r["transformer_ready"] \
		and not r["generation_ready"]' "$$tmp_dir/cuda.json"; \
	cat "$$tmp_dir/api.out"; \
	echo "production DeepSeek MoE live: CPU hash/learned and CUDA 43-layer operator"

# This serial target proves numeric-token CPU/CUDA backbone execution and operator reachability.
test-runtime-deepseek-transformer-live: cuda $(TRANSFORMER_LIVE_RUNNER) $(YVEX_BIN)
	@set -eu; \
	tmp_tag=runtime-deepseek-transformer-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	input="$$tmp_dir/deepseek-transformer.yvex-transformer-input"; \
	$(TRANSFORMER_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" "$$input" \
		>"$$tmp_dir/api.out"; \
	$(YVEX_BIN) execute transformer run --target deepseek4-v4-flash-dspark \
		--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --phase prefill --input token-ids --input-file "$$input" \
		--chunk-tokens 1 --context-capacity 1 --progress off --output json \
		>"$$tmp_dir/cuda.json"; \
	python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
		assert r["status"]=="complete" and r["layers"]==43 \
		and r["layers_executed"]==43 and r["swa_layers"]==2 \
		and r["csa_layers"]==21 and r["hca_layers"]==20 \
		and r["hash_router_executions"]==3 and r["learned_router_executions"]==40 \
		and r["routed_expert_executions"]==258 and r["shared_expert_executions"]==43 \
		and r["embedding_ready"] and r["transformer_ready"] \
		and r["full_model_prefill_ready"] and not r["model_decode_ready"] \
		and not r["logits_ready"] and not r["generation_ready"]' "$$tmp_dir/cuda.json"; \
	cat "$$tmp_dir/api.out"; \
	echo "production DeepSeek transformer live: CPU/CUDA token-to-normalized-hidden backbone"

# This serial target proves shared-context prefill and two real CPU/CUDA decode steps.
test-runtime-deepseek-decode-live: cuda $(DECODE_LIVE_RUNNER) $(YVEX_BIN)
	@set -eu; \
	tmp_tag=runtime-deepseek-decode-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	input="$$tmp_dir/deepseek-decode.yvex-transformer-input"; \
	$(DECODE_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" "$$input" \
		>"$$tmp_dir/api.out"; \
	$(YVEX_BIN) execute transformer decode --target deepseek4-v4-flash-dspark \
		--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --input token-ids --input-file "$$input" \
		--prefill-tokens 1 --prefill-chunk-tokens 1 --context-capacity 3 \
		--progress off --output json >"$$tmp_dir/cuda.json"; \
	python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
		assert r["status"]=="complete" and r["model_decode_ready"] \
		and r["decode_steps_requested"]==2 and r["decode_steps_completed"]==2 \
		and r["initial_committed_prefix"]==1 and r["final_committed_prefix"]==3 \
		and r["layers_executed"]==86 and r["swa_layers"]==4 \
		and r["csa_layers"]==42 and r["hca_layers"]==40 \
		and r["hash_router_executions"]==6 and r["learned_router_executions"]==80 \
		and r["routed_expert_executions"]==516 and r["shared_expert_executions"]==86 \
		and len(r["steps"])==2 and not r["logits_ready"] \
		and not r["sampling_ready"] and not r["generation_ready"]' "$$tmp_dir/cuda.json"; \
	cat "$$tmp_dir/api.out"; \
	echo "production DeepSeek decode live: shared-context CPU/CUDA repeated teacher-forced steps"

# This serial target proves exact resident-head projection for one prefill and two decode rows.
test-runtime-deepseek-logits-live: cuda $(LOGITS_LIVE_RUNNER) $(YVEX_BIN)
	@set -eu; \
	tmp_tag=runtime-deepseek-logits-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	input="$$tmp_dir/deepseek-logits.yvex-transformer-input"; \
	$(LOGITS_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" "$$input" \
		>"$$tmp_dir/api.out"; \
	$(YVEX_BIN) execute transformer logits --target deepseek4-v4-flash-dspark \
		--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --input token-ids --input-file "$$input" \
		--prefill-tokens 1 --prefill-chunk-tokens 1 --context-capacity 3 \
		--progress off --output json >"$$tmp_dir/cuda.json"; \
	python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
		assert r["status"]=="complete" and r["output_head_binding_ready"] \
		and r["output_head_residency_ready"] and r["logits_cpu_ready"] \
		and r["logits_cuda_ready"] and r["logits_prefill_ready"] \
		and r["logits_decode_ready"] and r["logits_full_vocabulary_ready"] \
		and r["logits_ready"] and r["vocabulary_size"]==129280 \
		and r["logits_rows_completed"]==3 and r["prefill_logits_rows"]==1 \
		and r["decode_logits_rows"]==2 and len(r["rows"])==3 \
		and all(x["logits_count"]==129280 for x in r["rows"]) \
		and not r["sampling_ready"] and not r["generation_ready"]' "$$tmp_dir/cuda.json"; \
	$(YVEX_BIN) execute transformer sample --target deepseek4-v4-flash-dspark \
		--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --input token-ids --input-file "$$input" \
		--prefill-tokens 1 --prefill-chunk-tokens 1 --context-capacity 3 \
		--strategy stochastic --temperature 0.8 --top-k 50 --top-p 0.95 \
		--min-p 0.05 --typical-p 0.9 --seed 42 --progress off --output json \
		>"$$tmp_dir/sample.json"; \
	python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
		assert r["status"]=="complete" and r["sampling_real_logits_ready"] \
		and r["sampling_ready"] and r["samples"]==3 \
		and r["strategy"]=="stochastic" and r["rng_algorithm"]==1 \
		and r["rng_version"]==1 and r["filter_order_version"]==2 \
		and r["sampling_completed_samples"]==3 and not r["sampling_partial"] \
		and r["prefill_samples"]==1 and r["decode_samples"]==2 \
		and len(r["selected_tokens"])==3 \
		and all(x["candidates"]>0 and x["rng_before"] and x["rng_after"] \
		        and x["source_identity"] and x["candidate_identity"] \
		        for x in r["selected_tokens"]) \
		and not r["token_append_ready"] and not r["tokenizer_runtime_ready"] \
		and not r["generation_ready"] and not r["cuda_sampling_ready"]' \
		"$$tmp_dir/sample.json"; \
	cat "$$tmp_dir/api.out"; \
	echo "production DeepSeek logits live: CPU/CUDA complete-vocabulary prefill/decode projection"

test-runtime-deepseek-sampling-live: test-runtime-deepseek-logits-live

# This serial target reuses the real-logits live workflow to hand actual sampled IDs
# into the metadata-only artifact tokenizer proof after all model resources close.
test-runtime-deepseek-tokenizer-live: cuda $(TOKENIZER_LIVE_RUNNER) $(LOGITS_LIVE_RUNNER) $(YVEX_BIN)
	@set -eu; \
	tmp_tag=runtime-deepseek-tokenizer-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	reference_python='$(YVEX_TOKENIZER_REFERENCE_PYTHON)'; \
	case "$$reference_python" in /*) ;; *) \
		echo "YVEX_TOKENIZER_REFERENCE_PYTHON must be absolute" >&2; exit 2;; \
	esac; \
	test -x "$$reference_python" || { echo "tokenizer reference Python is unavailable" >&2; exit 2; }; \
	$(LOGITS_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		"$$tmp_dir/tokenizer-sampling-input" >"$$tmp_dir/sampling.out"; \
	sampled=$$(sed -n 's/.*sampling_greedy_tokens=\([^ ]*\).*/\1/p' "$$tmp_dir/sampling.out"); \
	test -n "$$sampled" || { echo "real sampled token IDs are absent" >&2; exit 1; }; \
	$(TOKENIZER_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" "$$sampled" \
		>"$$tmp_dir/tokenizer.out"; \
	PYTHONDONTWRITEBYTECODE=1 "$$reference_python" tests/reference/tokenizer.py "$(DEEPSEEK_SOURCE)" \
		"$(abspath $(YVEX_BIN))" "$(DEEPSEEK_SELECTED_ARTIFACT)" >"$$tmp_dir/reference.out"; \
	$(YVEX_BIN) inspect tokenizer "$(DEEPSEEK_SELECTED_ARTIFACT)" >"$$tmp_dir/inspect.out"; \
	$(YVEX_BIN) execute tokenizer encode "$(DEEPSEEK_SELECTED_ARTIFACT)" --text 'hello world' --pieces \
		>"$$tmp_dir/tokenize.out"; \
	$(YVEX_BIN) execute tokenizer decode "$(DEEPSEEK_SELECTED_ARTIFACT)" --ids 33310,2058 \
		>"$$tmp_dir/detokenize.out"; \
	$(YVEX_BIN) execute tokenizer prompt "$(DEEPSEEK_SELECTED_ARTIFACT)" --system policy --user hi \
		--assistant ok --user next --tokens >"$$tmp_dir/prompt.out"; \
	grep -q '^tokenizer_runtime_ready: true$$' "$$tmp_dir/inspect.out"; \
	grep -q '^ids: 33310 2058$$' "$$tmp_dir/tokenize.out"; \
	grep -q '^text: "hello world"$$' "$$tmp_dir/detokenize.out"; \
	grep -q '^template: deepseek-v4-family-policy$$' "$$tmp_dir/prompt.out"; \
	cat "$$tmp_dir/tokenizer.out"; \
	cat "$$tmp_dir/reference.out"; \
	echo "production DeepSeek tokenizer live: artifact BPE, exact prompt, and incremental decode"

# This serial lane proves sampled-token feedback with independent lower-owner composition.
test-runtime-deepseek-generation-live: cuda $(GENERATION_LIVE_RUNNER) $(YVEX_BIN)
	@set -eu; \
	tmp_tag=runtime-deepseek-generation-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	binding='$(YVEX_RUNTIME_BINDING)'; \
	case "$$binding" in /*) ;; *) \
		echo "YVEX_RUNTIME_BINDING must be an absolute file" >&2; exit 2;; \
	esac; \
	test -f "$$binding" && test ! -L "$$binding" || { \
		echo "runtime binding must be a regular non-symlink file" >&2; exit 2; }; \
	$(GENERATION_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		cpu target-only greedy 0 1 >"$$tmp_dir/cpu.out"; \
	$(GENERATION_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		cuda target-only greedy 0 3 >"$$tmp_dir/cuda-greedy.out"; \
	$(GENERATION_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		cuda target-only stochastic 42 2 >"$$tmp_dir/cuda-stochastic-first.out"; \
	$(GENERATION_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		cuda target-only stochastic 42 2 >"$$tmp_dir/cuda-stochastic-second.out"; \
	cmp "$$tmp_dir/cuda-stochastic-first.out" "$$tmp_dir/cuda-stochastic-second.out"; \
	$(GENERATION_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		cuda dspark greedy 0 8 >"$$tmp_dir/cuda-dspark-greedy.out"; \
	$(GENERATION_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		cuda dspark stochastic 42 8 >"$$tmp_dir/cuda-dspark-stochastic-first.out"; \
	$(GENERATION_LIVE_RUNNER) "$(DEEPSEEK_SELECTED_ARTIFACT)" "$$binding" \
		cuda dspark stochastic 42 8 >"$$tmp_dir/cuda-dspark-stochastic-second.out"; \
	cmp "$$tmp_dir/cuda-dspark-stochastic-first.out" \
		"$$tmp_dir/cuda-dspark-stochastic-second.out"; \
	python3 -c 'import sys; f=dict(x.split("=",1) for x in open(sys.argv[1]).read().split() if "=" in x); \
		assert int(f["draft_cycles"])>0 and int(f["proposed"])>0 \
		and int(f["verified"])>0 and f["acceptance_corpus"]=="pass" \
		and int(f["corpus_prompts"])==3 and int(f["corpus_proposed"])>0 \
		and int(f["corpus_verified"])>0 and int(f["corpus_accepted"])>0 \
		and int(f["corpus_max_accepted_prefix"])>=2 \
		and f["speculation_cancellation"]=="pass"' \
		"$$tmp_dir/cuda-dspark-greedy.out"; \
	$(YVEX_BIN) execute transformer generate --target deepseek4-v4-flash-dspark \
		--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --text Hi --max-new-tokens 1 --max-output-bytes 64 \
		--context-capacity 8 --prefill-chunk-tokens 8 --strategy greedy \
		--progress off --output json >"$$tmp_dir/operator.json"; \
	python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
		assert r["status"]=="complete" and r["generation_ready"] \
		and not r["cli_generate_ready"] and r["sampled_tokens"]==1 \
		and r["model_committed_tokens"]==1 and r["decode_steps"]==1 \
		and r["logits_projections"]==1 and len(r["generated_tokens"])==1 \
		and r["generated_tokens"][0]["decode_submitted"] \
		and r["generated_tokens"][0]["token_id"]==r["generated_tokens"][0]["decode_input_id"]' \
		"$$tmp_dir/operator.json"; \
	$(YVEX_BIN) execute transformer generate --target deepseek4-v4-flash-dspark \
		--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --generation-mode dspark --text 'The capital of France is' \
		--max-new-tokens 8 \
		--max-output-bytes 1024 --context-capacity 32 --prefill-chunk-tokens 8 \
		--strategy greedy --progress off --output json >"$$tmp_dir/operator-dspark.json"; \
	python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
		assert r["status"]=="complete" and r["execution_mode"]=="dspark" \
		and r["draft_cycles"]>0 and r["proposed_tokens"]>0 \
		and r["target_verifications"]>0 and r["accepted_draft_tokens"]>0 \
		and r["sampled_tokens"]==r["model_committed_tokens"]' \
		"$$tmp_dir/operator-dspark.json"; \
	cat "$$tmp_dir/cpu.out" "$$tmp_dir/cuda-greedy.out" \
		"$$tmp_dir/cuda-stochastic-first.out" \
		"$$tmp_dir/cuda-dspark-greedy.out" \
		"$$tmp_dir/cuda-dspark-stochastic-first.out"; \
	echo "production DeepSeek generation live: target-only parity and verified DSpark speculation"

test-attention-cli-live: $(YVEX_BIN) tests/cli/attention_graph.sh
	@set -eu; \
	tmp_tag=attention-cli-live; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	YVEX_BIN="$(YVEX_BIN)" YVEX_TEST_OUT_DIR="$$tmp_dir/output" \
		YVEX_ATTENTION_LIVE=1 \
		YVEX_ATTENTION_MODELS_ROOT="$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
		YVEX_ATTENTION_ARTIFACT="$(DEEPSEEK_SELECTED_ARTIFACT)" \
		sh tests/cli/attention_graph.sh

test-attention-cuda: $(ATTENTION_LIVE_RUNNER)
	@set -eu; \
	tmp_tag=attention-cuda; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	YVEX_ATTENTION_CUDA_ONLY=1 $(ATTENTION_LIVE_RUNNER) \
		"$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" \
		"$(DEEPSEEK_SOURCE_MANIFEST)" >"$$tmp_dir/first.out"; \
	YVEX_ATTENTION_CUDA_ONLY=1 $(ATTENTION_LIVE_RUNNER) \
		"$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" \
		"$(DEEPSEEK_SOURCE_MANIFEST)" >"$$tmp_dir/second.out"; \
	cmp "$$tmp_dir/first.out" "$$tmp_dir/second.out"; \
	cat "$$tmp_dir/first.out"; \
	echo "attention CUDA live repeat: byte-identical"

test-source-payload-live-plan: $(SOURCE_PAYLOAD_LIVE_RUNNER)
	$(SOURCE_PAYLOAD_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-transform-ir-live-plan: $(SOURCE_PAYLOAD_LIVE_RUNNER)
	$(SOURCE_PAYLOAD_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-quant: $(QUANT_TEST_RUNNER)
	$(QUANT_TEST_RUNNER)

test-quant-asan:
	@set -eu; \
	tmp_tag=quant-asan; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	build_dir="$$tmp_dir/build"; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
	$(MAKE) BUILD_DIR="$$build_dir" \
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=address,leak' \
		LDFLAGS='$(LDFLAGS) -fsanitize=address,leak' test-quant

test-quant-ubsan:
	@set -eu; \
	tmp_tag=quant-ubsan; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	build_dir="$$tmp_dir/build"; \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	$(MAKE) BUILD_DIR="$$build_dir" \
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=undefined \
			-fno-sanitize-recover=undefined' \
		LDFLAGS='$(LDFLAGS) -fsanitize=undefined' test-quant

test-quant-sanitizers:
	$(MAKE) test-quant-asan
	$(MAKE) test-quant-ubsan

test-artifact-writer: $(ARTIFACT_TEST_RUNNER)
	$(ARTIFACT_TEST_RUNNER)

# Replays the writer suite as the explicit preallocation/IO/protocol fault lane.
test-artifact-writer-fault: $(ARTIFACT_TEST_RUNNER)
	$(ARTIFACT_TEST_RUNNER)

test-quant-live-plan: $(QUANT_LIVE_RUNNER)
	$(QUANT_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-quant-live: $(QUANT_LIVE_RUNNER)
	$(QUANT_LIVE_RUNNER) "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)" "$(DEEPSEEK_SELECTED_ARTIFACT)"

test-physical-variant-plan-deepseek-live: $(QUANT_LIVE_RUNNER) $(YVEX_BIN)
	@test -n "$(YVEX_IMATRIX)" || { echo "YVEX_IMATRIX is required" >&2; exit 2; }
	YVEX_QUANT_PRESET="$(YVEX_QUANT_DSPARK_PRESET)" YVEX_IMATRIX_PATH="$(YVEX_IMATRIX)" \
		$(QUANT_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"
	YVEX_BIN="$(YVEX_BIN)" YVEX_TEST_OUT_DIR="$(BUILD_DIR)/tests/physical-variant-refusal" \
		YVEX_DEEPSEEK_SOURCE="$(DEEPSEEK_SOURCE)" \
		YVEX_DEEPSEEK_MODELS_ROOT="$(DEEPSEEK_MODELS_ROOT)" \
		YVEX_DEEPSEEK_SOURCE_MANIFEST="$(DEEPSEEK_SOURCE_MANIFEST)" \
		YVEX_IMATRIX_PATH="$(YVEX_IMATRIX)" \
		sh tests/live/physical_variant.sh

test-quant-iq2-xxs-deepseek-live: test-physical-variant-plan-deepseek-live test-cuda

test-artifact-emit-deepseek-variant-live: $(ARTIFACT_LIVE_RUNNER) $(OFFICIAL_GGUF_CHECKER)
	@test -n "$(YVEX_IMATRIX)" || { echo "YVEX_IMATRIX is required" >&2; exit 2; }
	@test -n "$(YVEX_VARIANT_ARTIFACT)" || { echo "YVEX_VARIANT_ARTIFACT is required" >&2; exit 2; }
	@test -n "$(YVEX_VARIANT_BINDING_DIR)" || { echo "YVEX_VARIANT_BINDING_DIR is required" >&2; exit 2; }
	@mkdir -p "$(YVEX_VARIANT_BINDING_DIR)"
	YVEX_GGML_CHECKER="$(OFFICIAL_GGUF_CHECKER)" \
		YVEX_QUANT_PRESET="$(YVEX_QUANT_DSPARK_PRESET)" YVEX_IMATRIX_PATH="$(YVEX_IMATRIX)" \
		YVEX_VARIANT_BINDING_DIR="$(YVEX_VARIANT_BINDING_DIR)" \
		$(ARTIFACT_LIVE_RUNNER) --variant "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" \
			"$(DEEPSEEK_SOURCE_MANIFEST)" "$(YVEX_VARIANT_ARTIFACT)"

test-materialize-deepseek-variant-live: $(YVEX_BIN)
	@test -n "$(YVEX_IMATRIX)" || { echo "YVEX_IMATRIX is required" >&2; exit 2; }
	@test -f "$(YVEX_VARIANT_ARTIFACT)" || { echo "emitted YVEX_VARIANT_ARTIFACT is required" >&2; exit 2; }
	@set -eu; \
	root=$$(mktemp -d "$(BUILD_DIR)/tests/variant-materialize.XXXXXX"); \
	cleanup() { status=$$?; trap - 0 HUP INT TERM; \
		find "$$root" -xdev -mindepth 1 -delete; rmdir "$$root"; exit $$status; }; \
	trap cleanup 0 HUP INT TERM; \
	mkdir -p "$$root/bindings"; \
	$(YVEX_BIN) compile quant plan --target deepseek4-v4-flash-dspark \
		--source "$(DEEPSEEK_SOURCE)" --models-root "$(DEEPSEEK_MODELS_ROOT)" \
		--source-manifest "$(DEEPSEEK_SOURCE_MANIFEST)" \
		--preset "$(YVEX_QUANT_DSPARK_PRESET)" --imatrix-manifest "$(YVEX_IMATRIX)" \
		--out-plan "$$root/variant.plan" >/dev/null; \
	$(YVEX_BIN) execute attention prepare --target deepseek4-v4-flash-dspark \
		--source "$(DEEPSEEK_SOURCE)" --source-manifest "$(DEEPSEEK_SOURCE_MANIFEST)" \
		--models-root "$(DEEPSEEK_MODELS_ROOT)" --artifact "$(YVEX_VARIANT_ARTIFACT)" \
		--runtime-binding-dir "$$root/bindings" \
		--physical-variant-plan "$$root/variant.plan" \
		--quant-preset "$(YVEX_QUANT_DSPARK_PRESET)" \
		--imatrix-manifest "$(YVEX_IMATRIX)" --output json; \
	echo "variant materialization live: canonical operator binding accepted"

test-runtime-deepseek-variant-generation-live: cuda $(YVEX_BIN)
	@test -f "$(YVEX_VARIANT_ARTIFACT)" || { echo "emitted YVEX_VARIANT_ARTIFACT is required" >&2; exit 2; }
	@binding=$$(find "$(YVEX_VARIANT_BINDING_DIR)" -maxdepth 1 -type f \
		-name '*.yvex-runtime-binding' -print | sort | tail -1); \
	test -n "$$binding" || { echo "variant runtime binding is required" >&2; exit 2; }; \
	$(YVEX_BIN) execute transformer generate --target deepseek4-v4-flash-dspark \
		--artifact "$(YVEX_VARIANT_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cpu --text Hi --max-new-tokens 1 --max-output-bytes 64 \
		--context-capacity 8 --prefill-chunk-tokens 8 --strategy greedy \
		--progress off --output json; \
	$(YVEX_BIN) execute transformer generate --target deepseek4-v4-flash-dspark \
		--artifact "$(YVEX_VARIANT_ARTIFACT)" --runtime-binding "$$binding" \
		--backend cuda --text Hi --max-new-tokens 1 --max-output-bytes 64 \
		--context-capacity 8 --prefill-chunk-tokens 8 --strategy greedy \
		--progress off --output json

test-artifact-live-plan: $(ARTIFACT_LIVE_RUNNER)
	$(ARTIFACT_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-artifact-live-structure: $(ARTIFACT_LIVE_RUNNER) $(OFFICIAL_GGUF_CHECKER)
	YVEX_GGML_CHECKER="$(OFFICIAL_GGUF_CHECKER)" $(ARTIFACT_LIVE_RUNNER) --structure-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-artifact-live: $(ARTIFACT_LIVE_RUNNER) $(OFFICIAL_GGUF_CHECKER)
	YVEX_GGML_CHECKER="$(OFFICIAL_GGUF_CHECKER)" $(ARTIFACT_LIVE_RUNNER) "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-source-payload-live: $(SOURCE_PAYLOAD_LIVE_RUNNER)
	$(SOURCE_PAYLOAD_LIVE_RUNNER) "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test: test-core test-cli

test-gguf-artifact-abi: $(TEST_RUNNER) tests/test_gguf_artifact_abi.sh
	YVEX_TEST_FILTER=gguf_artifact_abi $(TEST_RUNNER)
	sh tests/test_gguf_artifact_abi.sh

test-gguf-layout-integrity: $(TEST_RUNNER) tests/test_gguf_layout_integrity.sh
	YVEX_TEST_FILTER=gguf_layout_integrity $(TEST_RUNNER)
	sh tests/test_gguf_layout_integrity.sh

test-gguf-qtype-abi: $(TEST_RUNNER) tests/test_gguf_qtype_abi.sh
	YVEX_TEST_FILTER=gguf_qtype_abi $(TEST_RUNNER)
	sh tests/test_gguf_qtype_abi.sh

test-layout: $(LIBYVEX) $(YVEX_BIN) $(TEST_REFERENCE_OBJS) tests/test_source_layout.sh
	sh tests/test_source_layout.sh

test-code-natural: tests/test_code_natural.sh
	sh tests/test_code_natural.sh

test-project-control: tests/test_project_control.sh ROADMAP.md CONTRIBUTING.md
	sh tests/test_project_control.sh

test-docs-surface: tests/test_docs_surface.sh
	sh tests/test_docs_surface.sh

test-documentation-architecture: tests/documentation_architecture.py \
		config/documentation_owners.tsv config/frozen_documents.tsv
	python3 tests/documentation_architecture.py

test-surface: tests/test_surface.sh
	sh tests/test_surface.sh

test-source-ownership: tests/test_source_ownership.sh config/source_owners.tsv
	sh tests/test_source_ownership.sh

test-repository-layout: $(LIBYVEX) tests/test_repository_layout.sh Makefile
	sh tests/test_repository_layout.sh

test-architecture-boundaries: $(LIBYVEX) $(YVEX_BIN) $(TEST_REFERENCE_OBJS) tests/test_architecture_boundaries.sh
	YVEX_LIB="$(LIBYVEX)" YVEX_BIN="$(YVEX_BIN)" \
		YVEX_CLIENT_LANE_OBJ="$(CLIENT_LANE_OBJ)" \
		YVEX_REFERENCE_OBJS="$(TEST_REFERENCE_OBJS)" \
		sh tests/test_architecture_boundaries.sh

smoke: test-cli

check: check-docs check-guardrails lib client test test-cuda-no-nvcc test-gguf-artifact-abi test-gguf-layout-integrity test-gguf-qtype-abi test-layout test-code-natural test-project-control test-docs-surface test-documentation-architecture test-surface test-source-ownership test-repository-layout test-architecture-boundaries smoke
	@echo "yvex check: ok"

$(LIBYVEX): $(CORE_OBJS)
	@mkdir -p $(@D)
	rm -f $@
	$(AR) rcsP $@ $^

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(SOURCE_MANIFEST_MK): $(SOURCE_OWNER_MANIFEST) $(SOURCE_MANIFEST_GENERATOR)
	@mkdir -p $(@D)
	python3 $(SOURCE_MANIFEST_GENERATOR) --manifest $(SOURCE_OWNER_MANIFEST) --output $@

$(OPERATOR_REGISTRY_HEADER) $(OPERATOR_REGISTRY_C) $(OPERATOR_REGISTRY_IDENTITY) &: \
		$(OPERATOR_REGISTRY_SOURCE) $(OPERATOR_REGISTRY_GENERATOR)
	python3 $(OPERATOR_REGISTRY_GENERATOR) --registry $(OPERATOR_REGISTRY_SOURCE) \
		--output $(OPERATOR_REGISTRY_DIR)

$(OPERATOR_MIGRATION_DOC): $(OPERATOR_REGISTRY_SOURCE) $(OPERATOR_REGISTRY_GENERATOR) \
		$(OPERATOR_AUDIT_FILES)
	python3 $(OPERATOR_REGISTRY_GENERATOR) --registry $(OPERATOR_REGISTRY_SOURCE) \
		--output $(OPERATOR_REGISTRY_DIR) --audit-root $(OPERATOR_AUDIT_ROOT) \
		--migration-output $@

$(OPERATOR_REGISTRY_OBJ): $(OPERATOR_REGISTRY_C) $(OPERATOR_REGISTRY_HEADER)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) -I$(BUILD_DIR)/generated $(CFLAGS) $(DEPFLAGS) -c \
		$(OPERATOR_REGISTRY_C) -o $@

.PHONY: FORCE
FORCE:

# Revalidate commit and source cleanliness on every invocation; replace the
# generated header only when exact provenance changes.
$(BUILD_COMMIT_HEADER): FORCE
	@mkdir -p $(@D)
	@tmp="$@.tmp"; \
	printf '#ifndef YVEX_BUILD_PROVENANCE_INCLUDED\n#define YVEX_BUILD_PROVENANCE_INCLUDED\n#define YVEX_BUILD_COMMIT "%s"\n#define YVEX_BUILD_SOURCE_STATE "%s"\n#define YVEX_BUILD_SOURCE_DELTA_IDENTITY "%s"\n#define YVEX_BUILD_IDENTITY "%s"\n#define YVEX_BUILD_SOURCE_ROOT "%s"\n#endif\n' \
		'$(YVEX_BUILD_COMMIT)' '$(YVEX_BUILD_SOURCE_STATE)' \
		'$(YVEX_BUILD_SOURCE_DELTA_IDENTITY)' '$(YVEX_BUILD_IDENTITY)' \
		'$(YVEX_BUILD_SOURCE_ROOT)' >"$$tmp"; \
	if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; \
	else mv "$$tmp" "$@"; fi

$(OBJ_DIR)/tests/unit/%.o: tests/unit/%.c tests/test.h
	@mkdir -p $(@D)
	$(CC) $(TEST_CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/tests/unit/cuda/%.o: tests/unit/cuda/%.c tests/test.h
	@mkdir -p $(@D)
	$(CC) $(TEST_CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/%.ptx: %.cu include/yvex/qtype.h src/backend/cuda/kernel_primitives.h
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_ARCH_FLAG) -ptx $< -o $@

$(OBJ_DIR)/%.cubin: %.cu include/yvex/qtype.h src/backend/cuda/kernel_primitives.h
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_ARCH_FLAG) -cubin $< -o $@
	@$(CUOBJDUMP) --list-elf $@ | grep -F '$(CUDA_NATIVE_ARCH)' >/dev/null || { \
		echo "native CUDA image does not contain $(CUDA_NATIVE_ARCH): $@" >&2; exit 1; }
	@$(CUOBJDUMP) --dump-sass $@ | grep -F 'Function :' >/dev/null || { \
		echo "native CUDA image contains no SASS functions: $@" >&2; exit 1; }

$(CUDA_PTX_INC): $(CUDA_PTX)
	@mkdir -p $(@D)
	@tmp="$@.tmp.$$$$"; trap 'rm -f "$$tmp"' EXIT HUP INT TERM; { \
		index=0; names=''; \
		for image in $(CUDA_PTX); do \
			name="cuda_kernel_ptx_$${index}"; names="$$names $$name"; \
			printf 'static const unsigned char %s[] = {\n' "$$name"; \
			{ cat "$$image"; printf '\0'; } | xxd -i; \
			printf '};\n'; index=$$((index + 1)); \
		done; \
		printf 'static const unsigned char *const cuda_kernel_ptx_images[] = {\n'; \
		for name in $$names; do printf '    %s,\n' "$$name"; done; \
		printf '};\nstatic const unsigned long long cuda_kernel_ptx_image_bytes[] = {\n'; \
		for name in $$names; do printf '    sizeof(%s) - 1u,\n' "$$name"; done; \
		printf '};\n#define CUDA_KERNEL_PTX_IMAGE_COUNT %s\n' "$$index"; \
	} >"$$tmp"; mv "$$tmp" "$@"; trap - EXIT HUP INT TERM

$(CUDA_CUBIN_INC): $(CUDA_CUBIN)
	@mkdir -p $(@D)
	@tmp="$@.tmp.$$$$"; trap 'rm -f "$$tmp"' EXIT HUP INT TERM; { \
		index=0; names=''; \
		for image in $(CUDA_CUBIN); do \
			name="cuda_kernel_cubin_$${index}"; names="$$names $$name"; \
			printf 'static const unsigned char %s[] = {\n' "$$name"; \
			cat "$$image" | xxd -i; \
			printf '};\n'; index=$$((index + 1)); \
		done; \
		printf 'static const unsigned char *const cuda_kernel_cubin_images[] = {\n'; \
		for name in $$names; do printf '    %s,\n' "$$name"; done; \
		printf '};\nstatic const unsigned long long cuda_kernel_cubin_image_bytes[] = {\n'; \
		for name in $$names; do printf '    sizeof(%s),\n' "$$name"; done; \
		printf '};\n#define CUDA_KERNEL_CUBIN_IMAGE_COUNT %s\n' "$$index"; \
		printf 'static const char cuda_kernels_cubin_arch[] = "%s";\n' \
			'$(CUDA_NATIVE_ARCH)'; \
	} >"$$tmp"; mv "$$tmp" "$@"; trap - EXIT HUP INT TERM

$(YVEX_BIN): $(YVEX_OBJS) $(OPENAI_ADAPTER_OBJS) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(YVEX_OBJS) $(OPENAI_ADAPTER_OBJS) $(LIBYVEX) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): $(TEST_MAIN_OBJ) $(TEST_UNIT_OBJS) $(TEST_REFERENCE_OBJS) \
	$(OPENAI_ADAPTER_OBJS) $(LIBYVEX) tests/test.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(TEST_MAIN_OBJ) $(TEST_UNIT_OBJS) $(TEST_REFERENCE_OBJS) \
		$(OPENAI_ADAPTER_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(QUANT_TEST_RUNNER): $(QUANT_TEST_RUNNER_OBJ) $(QUANT_TEST_UNIT_OBJS) $(LIBYVEX) tests/test.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(QUANT_TEST_RUNNER_OBJ) $(QUANT_TEST_UNIT_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(ARTIFACT_TEST_RUNNER): $(ARTIFACT_TEST_RUNNER_OBJ) $(OBJ_DIR)/tests/unit/quant_execute.o $(LIBYVEX) tests/test.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(ARTIFACT_TEST_RUNNER_OBJ) $(OBJ_DIR)/tests/unit/quant_execute.o $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(OPENAI_FAKE_HOST): $(OPENAI_FAKE_HOST_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(OPENAI_FAKE_HOST_OBJ) $(LIBYVEX) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(OPENAI_ADAPTER_HOST): $(OPENAI_ADAPTER_HOST_OBJ) $(OPENAI_ADAPTER_OBJS) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(OPENAI_ADAPTER_HOST_OBJ) $(OPENAI_ADAPTER_OBJS) $(LIBYVEX) \
		$(LDFLAGS) $(LDLIBS) -o $@

$(SOURCE_PAYLOAD_LIVE_RUNNER): $(SOURCE_PAYLOAD_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(SOURCE_PAYLOAD_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(QUANT_LIVE_RUNNER): $(QUANT_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(QUANT_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(ARTIFACT_LIVE_RUNNER): $(ARTIFACT_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(ARTIFACT_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(MATERIALIZE_LIVE_RUNNER): $(MATERIALIZE_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(MATERIALIZE_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(MINIMAX_AUDIO_LIVE_RUNNER): $(MINIMAX_AUDIO_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(MINIMAX_AUDIO_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(MINIMAX_VIDEO_LIVE_RUNNER): $(MINIMAX_VIDEO_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(MINIMAX_VIDEO_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(MINIMAX_TEXT_LIVE_RUNNER): $(MINIMAX_TEXT_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(MINIMAX_TEXT_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(ATTENTION_LIVE_RUNNER): $(ATTENTION_LIVE_OBJ) $(TEST_REFERENCE_OBJS) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(ATTENTION_LIVE_OBJ) $(TEST_REFERENCE_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(PREFILL_LIVE_RUNNER): $(PREFILL_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(PREFILL_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(MOE_LIVE_RUNNER): $(MOE_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(MOE_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(TRANSFORMER_LIVE_RUNNER): $(TRANSFORMER_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(TRANSFORMER_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(DECODE_LIVE_RUNNER): $(DECODE_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DECODE_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(LOGITS_LIVE_RUNNER): $(LOGITS_LIVE_OBJ) $(TEST_REFERENCE_OBJS) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LOGITS_LIVE_OBJ) $(TEST_REFERENCE_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(TOKENIZER_LIVE_RUNNER): $(TOKENIZER_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(TOKENIZER_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(GENERATION_LIVE_RUNNER): $(GENERATION_LIVE_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(GENERATION_LIVE_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(OFFICIAL_GGUF_CHECKER): tests/external/ggml_gguf_check.cpp
	@test "$$(git -C "$(PINNED_GGML_ROOT)" rev-parse HEAD)" = af97976c7810cdabb1863172f31c432dab767de7
	@test -z "$$(git -C "$(PINNED_GGML_ROOT)" status --porcelain --untracked-files=no)"
	cmake -S "$(PINNED_GGML_ROOT)" -B "$(PINNED_GGML_BUILD)" \
		-DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
		-DGGML_BUILD_TOOLS=OFF -DGGML_BUILD_SERVER=OFF \
		-DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
	cmake --build "$(PINNED_GGML_BUILD)" -j4
	c++ -std=c++17 -Wall -Wextra -pedantic \
		-I"$(PINNED_GGML_ROOT)/include" $< \
		"$(PINNED_GGML_BUILD)/src/libggml-base.a" \
		-fopenmp -ldl -pthread -lm -o $@

$(CUDA_TEST_RUNNER): $(CUDA_TEST_MAIN_OBJ) $(CUDA_TEST_UNIT_OBJS) $(LIBYVEX) tests/test.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CUDA_TEST_MAIN_OBJ) $(CUDA_TEST_UNIT_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

check-docs: test-documentation-architecture test-project-control test-docs-surface
	@echo "yvex documentation: ok"

check-guardrails: check-source-manifest $(LIBYVEX) $(YVEX_BIN) \
		$(TEST_REFERENCE_OBJS)
	@sh tests/test_source_ownership.sh
	@sh tests/test_repository_layout.sh
	@YVEX_LIB="$(LIBYVEX)" YVEX_BIN="$(YVEX_BIN)" \
		YVEX_REFERENCE_OBJS="$(TEST_REFERENCE_OBJS)" \
		sh tests/test_architecture_boundaries.sh
	@test ! -e docs/spine.md
	@test ! -d docs/spines
	@test ! -d docs/integration
	@test ! -d docs/benchmark
	@test -f docs/README.md
	@test ! -e docs/backend-contract.md
	@test ! -e docs/cli-commands.md
	@test ! -e docs/cli-interface-spine.md
	@test ! -e docs/cli-runtime.md
	@test ! -e docs/runtime-filesystem.md
	@test ! -d benches
	@test ! -d examples
	@test ! -d protocols
	@test ! -d cli
	@test ! -d server
	@test ! -e tests/README.md
	@test ! -e include/yvex/sampler.h
	@test ! -d backends
	@test -d src
	@test -d src/app
	@test -d src/cli
	@test -d src/cli/commands
	@test -d src/cli/render
	@test -d src/cli/io
	@test -d config/operator
	@test -f config/operator/registry.json
	@test -f tools/generate_operator_registry.py
	@test ! -d src/cli/catalog
	@test -d src/cli/schema
	@test -d src/core
	@test -d src/artifact
	@test -d src/backend
	@test -d src/backend/cuda
	@test -d src/server
	@test -d src/gguf
	@test -d src/model
	@test -d src/tokenizer
	@test -d src/runtime
	@test ! -d src/generation
	@test ! -d src/eval
	@test ! -d src/bench
	@test ! -d cuda
	@test ! -d gguf
	@test ! -d models
	@test -d tests/vectors
	@test -f tests/vectors/manifest.json
	@test -f tests/test.c
	@test -f tests/test_cuda.c
	@test -f tests/cli.sh
	@test "$$(find tests -maxdepth 1 -type f \( -name 'test.c' -o -name 'test_*.c' \) | wc -l | tr -d ' ')" -le "2"
	@test "$$(find tests -maxdepth 1 -type f -name 'test_cli*.sh' | wc -l | tr -d ' ')" = "0"
	@test -f include/yvex/server.h
	@test ! -d fixtures
	@test -f src/cli/main.c
	@test -f src/cli/io/server.c
	@test -f src/server/core.c
	@test -z "$$(git ls-files 'yvex_*.c')"
	@test -z "$$(git ls-files 'yvex_*_private.h')"
	@test ! -d ui
	@test ! -d app
	@test ! -d desktop
	@! grep -RIn -E "N[E]T\\.SPINE|N[E]T moves streams|C[L]ORI|c[l]ori-codename|docs/arc[h]ive|c[l]ori_|libc[l]ori|c[l]orid|include/c[l]ori|~/\\.config/c[l]ori|github\\.com/yailabs/c[l]ori|yailabs/c[l]ori" --exclude-dir=.git --exclude-dir=build . >/dev/null
	@! grep -Ei "production-read[y]|implemented infer[e]nce|implemented ser[v]er|supports C[U]DA|supports M[e]tal|supports M[L]X|supports llama\\.cpp|O[p]enAI-compatible ser[v]er" README.md >/dev/null
	@! grep -Ei "benchmark results" README.md | grep -vi "benchmark results are not measured" >/dev/null
-include $(DEPENDENCY_FILES)

clean:
	@set -eu; \
	build_dir='$(BUILD_DIR)'; \
	case "$$build_dir" in \
		build|build/*|/tmp/yvex-*/build|/tmp/yvex.*/build) ;; \
		*) printf 'clean: refusing unowned BUILD_DIR: %s\n' "$$build_dir" >&2; exit 1 ;; \
	esac; \
	if [ -L "$$build_dir" ]; then \
		printf 'clean: refusing symlink BUILD_DIR: %s\n' "$$build_dir" >&2; exit 1; \
	fi; \
	if [ -d "$$build_dir" ]; then \
		find "$$build_dir" -depth -mindepth 1 -delete; \
		rmdir "$$build_dir"; \
	elif [ -e "$$build_dir" ]; then \
		printf 'clean: refusing non-directory BUILD_DIR: %s\n' "$$build_dir" >&2; exit 1; \
	fi; \
	if test "$$build_dir" = build; then \
		rm -f -- ./yvex ./yvexd ./yvex-openai ./yvex-dev ./*.o; \
	fi

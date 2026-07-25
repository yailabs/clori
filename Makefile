# YVEX - Build and validation entrypoint
#
# Builds the YVEX C library, root binaries, CUDA kernel unit, and tests.
#
# Primary commands:
#   make info
#   make lib
#   make cli
#   make server
#   make cuda-info
#   make cuda
#   make test-cuda
#   make check-cuda
#   make test
#   make test-core
#   make test-runtime
#   make test-runtime-benchmark-chart
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
# Interface policy:
#   - YVEX is CLI-only.
#   - ./yvex and ./yvexd are repository-local compiled products.

.DEFAULT_GOAL := all

.PHONY: all info lib cli server cuda-info cuda-kernels cuda test-cuda test-cuda-graph \
	test-cuda-no-nvcc smoke-cuda check-cuda test test-core test-cli test-materialize \
	test-runtime-descriptor test-runtime-binding test-runtime-model-session \
	test-runtime-residency test-runtime-phases test-runtime-envelope \
	test-runtime-operator test-runtime-digests test-runtime-family-neutrality \
	test-runtime-state test-runtime-benchmark test-runtime-benchmark-chart \
	test-runtime-benchmark-chart-live update-runtime-benchmark-charts \
	test-runtime-attention-live \
	test-runtime test-runtime-asan test-runtime-asan-live \
	test-runtime-ubsan test-runtime-ubsan-live test-runtime-sanitizers \
	test-runtime-sanitizers-live test-materialize-live-plan \
	test-materialize-live test-attention test-attention-fixture-isolation \
	test-attention-live-plan test-attention-live test-attention-cli-live \
	test-attention-cuda test-quant test-quant-live-plan test-quant-live \
	test-artifact-writer test-artifact-writer-fault test-artifact-live-plan \
	test-artifact-live-structure test-artifact-live test-transform-ir-live-plan \
	test-source-payload-live-plan test-source-payload-live test-gguf-artifact-abi \
	test-gguf-layout-integrity test-gguf-qtype-abi test-layout test-code-natural \
	test-project-ledger test-docs-surface test-surface test-source-ownership \
	test-repository-layout test-architecture-boundaries smoke check check-docs \
	check-guardrails clean

CC ?= cc
AR ?= ar
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
NVCCFLAGS ?=
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
CFLAGS ?= -std=c11 -Wall -Wextra -pedantic -Wstrict-prototypes \
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
DEEPSEEK_SOURCE ?= $(HOME)/lab/models/hf/deepseek/DeepSeek-V4-Flash
DEEPSEEK_MODELS_ROOT ?= $(HOME)/lab/models/gguf
DEEPSEEK_SOURCE_MANIFEST ?= $(DEEPSEEK_MODELS_ROOT)/deepseek/deepseek-source-manifest.json
DEEPSEEK_OPERATOR_MODELS_ROOT ?= $(HOME)/lab/models
DEEPSEEK_SELECTED_ARTIFACT ?= $(DEEPSEEK_MODELS_ROOT)/deepseek/deepseek-v4-flash-q8_0-q2_k-v1.gguf
YVEX_RUNTIME_BENCHMARK_DIR ?=
YVEX_RUNTIME_BINDING ?=
PINNED_GGML_ROOT ?= /tmp/yvex-ggml-af97976
PINNED_GGML_BUILD ?= $(PINNED_GGML_ROOT)/build-yvex

LIBYVEX ?= $(LIB_DIR)/libyvex.a
YVEX_BIN ?= ./yvex
YVEXD_BIN ?= ./yvexd

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

CLI_COMMAND_SRCS := src/cli/commands/graph.c \
	src/cli/commands/model_artifacts.c \
	src/cli/commands/model_target.c \
	$(sort $(filter-out src/cli/commands/graph.c src/cli/commands/model_artifacts.c src/cli/commands/model_target.c,$(wildcard src/cli/commands/*.c)))
CLI_INPUT_SRCS := src/cli/input/graph.c \
	src/cli/input/model_artifacts.c \
	src/cli/input/model_target.c \
	$(sort $(filter-out src/cli/input/graph.c src/cli/input/model_artifacts.c src/cli/input/model_target.c,$(wildcard src/cli/input/*.c)))
CLI_RENDER_SRCS := src/cli/render/graph.c \
	src/cli/render/model_artifacts.c \
	src/cli/render/model_target.c \
	$(sort $(filter-out src/cli/render/graph.c src/cli/render/model_artifacts.c src/cli/render/model_target.c,$(wildcard src/cli/render/*.c)))
CLI_MODEL_ARTIFACT_SRCS := $(sort $(wildcard src/cli/model_artifacts/*.c))
CLI_IO_SRCS := $(sort $(wildcard src/cli/io/*.c))

CORE_SRCS := \
	src/core/status.c \
	src/core/fs.c \
	src/core/sha256.c \
	src/core/shard_index.c \
	src/accounts/provider.c \
	src/artifact/core.c \
	src/artifact/descriptor.c \
	src/artifact/identity.c \
	src/artifact/integrity.c \
	src/artifact/materialize.c \
	src/artifact/roundtrip_gate.c \
	src/backend/core.c \
	src/backend/cpu.c \
	src/backend/report.c \
	src/runtime/graph.c \
	src/runtime/benchmark.c \
	src/runtime/binding.c \
	src/runtime/residency.c \
	src/graph/state.c \
	src/gguf/core.c \
	src/gguf/conversion.c \
	src/gguf/imatrix.c \
	src/gguf/quant_job.c \
	src/gguf/quant_policy.c \
	src/gguf/tools.c \
	src/gguf/descriptor.c \
	src/gguf/file_sink.c \
	src/gguf/layout_integrity.c \
	src/gguf/qtype.c \
	src/gguf/reader.c \
	src/gguf/tokenizer_metadata.c \
	src/gguf/writer.c \
	src/gguf/quant_registry.c \
	src/gguf/quant_scalar.c \
	src/gguf/quant_block.c \
	src/gguf/quant_compute.c \
	src/gguf/quant_plan.c \
	src/gguf/quant_sink.c \
	src/gguf/quant_execute.c \
	src/graph/attention.c \
	src/graph/numeric.c \
	src/graph/families/deepseek_v4.c \
	src/graph/core.c \
	src/graph/plan.c \
	src/graph/memory_plan.c \
	src/io/writer.c \
	src/model/core.c \
	src/model/families/deepseek_v4.c \
	src/model/compilation/ir.c \
	src/model/compilation/ir_identity.c \
	src/model/compilation/ir_validate.c \
	src/model/compilation/binding.c \
	src/runtime/descriptor.c \
	src/model/artifacts/gate.c \
	src/model/artifacts/ref.c \
	src/model/artifacts/registry.c \
	src/model/artifacts/write.c \
	src/model/target/mapping_gate.c \
	src/model/target/missing_role.c \
	src/model/target/model_class_profile.c \
	src/model/target/candidates.c \
	src/model/target/catalog.c \
	src/model/target/decision.c \
	src/model/target/report.c \
	src/model/target/sidecar_write.c \
	src/model/target/output_head_map.c \
	src/model/target/qtype_policy.c \
	src/model/target/qtype_role_support.c \
	src/model/target/tensor_collection.c \
	src/model/target/tensor_naming.c \
	src/model/target/tokenizer_map.c \
	src/runtime/core.c \
	src/source/native_weights.c \
	src/source/safetensors_header.c \
	src/source/inventory.c \
	src/core/json.c \
	src/source/manifest.c \
	src/source/payload.c \
	src/source/payload_identity.c \
	src/source/payload_plan.c \
	src/source/payload_stream.c \
	src/source/provenance.c \
	src/source/report.c \
	src/source/scan.c \
	src/source/verify.c \
	src/source/write.c \
	src/tokenizer/token_input.c \
	src/tokenizer/core.c \
	src/server/core.c

CLI_SRCS := \
	src/cli/main.c \
	$(CLI_COMMAND_SRCS) \
	$(CLI_INPUT_SRCS) \
	$(CLI_MODEL_ARTIFACT_SRCS) \
	$(CLI_RENDER_SRCS) \
	$(CLI_IO_SRCS)

CLI_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(CLI_SRCS))
DAEMON_OBJ := $(OBJ_DIR)/src/daemon/yvexd.o

CUDA_SRCS := \
	src/backend/cuda/backend.c \
	src/backend/cuda/capability.c \
	src/backend/cuda/graph.c \
	src/backend/cuda/tensor.c \
	src/backend/cuda/ops.c \
	src/backend/cuda/info.c \
	src/backend/cuda/qtype.c \
	src/backend/cuda/families/deepseek_v4.c \
	src/backend/cuda/errors.c

CUDA_CU_SRCS := \
	src/backend/cuda/kernels.cu

CUDA_ARCH_FLAG := $(if $(filter auto,$(YVEX_CUDA_ARCH)),,-arch=$(YVEX_CUDA_ARCH))
CUDA_PTX := $(patsubst %.cu,$(OBJ_DIR)/%.ptx,$(CUDA_CU_SRCS))
CUDA_PTX_INC := $(OBJ_DIR)/generated/cuda_kernels_ptx.inc

CORE_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(CORE_SRCS))
CUDA_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(CUDA_SRCS))
CORE_OBJS += $(CUDA_OBJS)

ifeq ($(NVCC_AVAILABLE),yes)
CPPFLAGS += -DYVEX_HAVE_CUDA_KERNEL_PTX=1
$(OBJ_DIR)/src/backend/cuda/capability.o: CPPFLAGS += -I$(OBJ_DIR)/generated
$(OBJ_DIR)/src/backend/cuda/capability.o: $(CUDA_PTX_INC)
endif

$(OBJ_DIR)/src/cli/commands/graph.o: CPPFLAGS += -D_XOPEN_SOURCE=700 -I$(BUILD_DIR)/generated
$(OBJ_DIR)/src/cli/commands/graph.o: $(BUILD_COMMIT_HEADER)
$(OBJ_DIR)/src/runtime/benchmark.o: CPPFLAGS += -I$(BUILD_DIR)/generated
$(OBJ_DIR)/src/runtime/benchmark.o: $(BUILD_COMMIT_HEADER)

TEST_RUNNER := $(TEST_DIR)/test
QUANT_TEST_RUNNER := $(TEST_DIR)/test_quant
ARTIFACT_TEST_RUNNER := $(TEST_DIR)/test_artifact_writer
SOURCE_PAYLOAD_LIVE_RUNNER := $(TEST_DIR)/source_payload_deepseek
QUANT_LIVE_RUNNER := $(TEST_DIR)/quant_deepseek
ARTIFACT_LIVE_RUNNER := $(TEST_DIR)/artifact_deepseek
MATERIALIZE_LIVE_RUNNER := $(TEST_DIR)/materialize_deepseek
ATTENTION_LIVE_RUNNER := $(TEST_DIR)/attention_deepseek
OFFICIAL_GGUF_CHECKER := $(TEST_DIR)/ggml_gguf_check
CUDA_TEST_RUNNER := $(TEST_DIR)/test_cuda

TEST_UNIT_SRCS := $(sort $(filter-out tests/unit/quant_runner.c tests/unit/artifact_writer_runner.c,$(wildcard tests/unit/*.c)))
TEST_UNIT_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(TEST_UNIT_SRCS))
TEST_REFERENCE_SRCS := $(sort $(wildcard tests/reference/*.c))
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
	tests/unit/quant_policy.c
QUANT_TEST_UNIT_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(QUANT_TEST_UNIT_SRCS))
QUANT_TEST_RUNNER_OBJ := $(OBJ_DIR)/tests/unit/quant_runner.o
ARTIFACT_TEST_RUNNER_OBJ := $(OBJ_DIR)/tests/unit/artifact_writer_runner.o

CUDA_TEST_UNIT_SRCS := $(sort $(wildcard tests/unit/cuda/*.c))
CUDA_TEST_UNIT_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(CUDA_TEST_UNIT_SRCS))
CUDA_TEST_MAIN_OBJ := $(OBJ_DIR)/tests/test_cuda.o

SOURCE_PAYLOAD_LIVE_OBJ := $(OBJ_DIR)/tests/live/source_payload_deepseek.o
QUANT_LIVE_OBJ := $(OBJ_DIR)/tests/live/quant_deepseek.o
ARTIFACT_LIVE_OBJ := $(OBJ_DIR)/tests/live/artifact_deepseek.o
MATERIALIZE_LIVE_OBJ := $(OBJ_DIR)/tests/live/materialize_deepseek.o
ATTENTION_LIVE_OBJ := $(OBJ_DIR)/tests/live/attention_deepseek.o

RUNNER_OBJS := $(TEST_MAIN_OBJ) $(QUANT_TEST_RUNNER_OBJ) \
	$(ARTIFACT_TEST_RUNNER_OBJ) $(CUDA_TEST_MAIN_OBJ) \
	$(SOURCE_PAYLOAD_LIVE_OBJ) $(QUANT_LIVE_OBJ) $(ARTIFACT_LIVE_OBJ) \
	$(MATERIALIZE_LIVE_OBJ) $(ATTENTION_LIVE_OBJ)
DEPENDENCY_FILES := $(CORE_OBJS:.o=.d) $(CLI_OBJS:.o=.d) \
	$(DAEMON_OBJ:.o=.d) $(TEST_UNIT_OBJS:.o=.d) \
	$(TEST_REFERENCE_OBJS:.o=.d) $(QUANT_TEST_UNIT_OBJS:.o=.d) \
	$(CUDA_TEST_UNIT_OBJS:.o=.d) $(RUNNER_OBJS:.o=.d)

CLI_TEST := tests/cli.sh

CURRENT_DOCS := README.md AGENTS.md PROJECT.md MODEL_ARTIFACTS.md NOTICE.md \
	docs/api.md docs/contract.md docs/model-families.md \
	docs/operator-runbook.md docs/cli-output-architecture.md \
	docs/reference-architecture.md docs/v010-release-doctrine.md \
	docs/topology-closure-audit.md docs/system-target.md

info:
	@echo "yvex: native C/CUDA verified-artifact inference system"
	@echo "project_control: PROJECT.md"
	@echo "interface: operator CLI plus C library ABI"
	@echo "library: libyvex.a"
	@echo "operator: ./yvex graph attention"
	@echo "daemon: ./yvexd bounded status shell"
	@echo "runtime_attention: CPU eager and admitted GB10 CUDA eager/piecewise/full implemented"
	@echo "benchmark_attention: identity-bound baseline, JSON/CSV, and deterministic SVG capability implemented"
	@echo "persistent_kv: not implemented"
	@echo "full_model_inference: not implemented"
	@echo "generation: not implemented"
	@echo "release: blocked"

all: lib cli server

lib: $(LIBYVEX)

cli: $(YVEX_BIN)

server: $(YVEXD_BIN)

cuda-info: $(YVEX_BIN)
	@echo "nvcc: $$(command -v $(NVCC) >/dev/null 2>&1 && command -v $(NVCC) || echo unavailable)"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "YVEX_CUDA_ARCH: $(YVEX_CUDA_ARCH)"
	$(YVEX_BIN) cuda-info

cuda-kernels: $(CUDA_PTX_INC)
	@echo "yvex cuda kernels: built from $(CUDA_CU_SRCS)"

cuda: cuda-kernels lib cli server $(CUDA_TEST_RUNNER)
	@echo "yvex cuda build: dynamic CUDA Driver API path plus CUDA kernel PTX"

test-cuda: cuda
	$(YVEX_BIN) cuda-info >/dev/null
	$(CUDA_TEST_RUNNER)

test-cuda-graph: cuda
	YVEX_CUDA_TEST_FILTER=graph $(CUDA_TEST_RUNNER)

smoke-cuda: cuda
	YVEX_BIN=$(YVEX_BIN) YVEXD_BIN=$(YVEXD_BIN) sh $(CLI_TEST) --cuda

check-cuda: cuda-info test-cuda smoke-cuda test-attention-cuda
	@echo "yvex check-cuda: ok"

test-cuda-no-nvcc: tests/test_cuda_failclosed.sh
	$(MAKE) BUILD_DIR=build/no-nvcc \
		YVEX_BIN=build/no-nvcc/yvex \
		YVEXD_BIN=build/no-nvcc/yvexd \
		NVCC=__yvex_nvcc_unavailable__ all
	YVEX_BIN=build/no-nvcc/yvex sh tests/test_cuda_failclosed.sh

test-core: $(TEST_RUNNER)
	$(TEST_RUNNER)

test-cli: $(YVEX_BIN) $(YVEXD_BIN) $(CLI_TEST)
	YVEX_BIN=$(YVEX_BIN) YVEXD_BIN=$(YVEXD_BIN) sh $(CLI_TEST)

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

test-runtime-benchmark: $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_benchmark $(TEST_RUNNER)

# The benchmark owner generates and authenticates the bounded SVG chart as part
# of its transactional evidence lifecycle.
test-runtime-benchmark-chart: $(TEST_RUNNER)
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
		$(YVEX_BIN) graph attention benchmark --target deepseek4-v4-flash \
			--models-root "$(DEEPSEEK_OPERATOR_MODELS_ROOT)" \
			--artifact "$(DEEPSEEK_SELECTED_ARTIFACT)" --runtime-binding "$$binding" \
			--backend cuda --phase decode --mode "$$mode" --scope full \
			--operation-scope release-attention-set --probe canonical \
			--warmup 3 --repeat 20 --progress off \
			--baseline "$$evidence_dir/$$mode.yvex-benchmark" --write-baseline \
			--chart "$$evidence_dir/$$mode.svg" --output json \
			>"$$evidence_dir/$$mode.json"; \
		$(YVEX_BIN) graph attention benchmark --target deepseek4-v4-flash \
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
	YVEX_TEST_FILTER=runtime_binding $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_state $(TEST_RUNNER)
	YVEX_TEST_FILTER=runtime_benchmark $(TEST_RUNNER)
	@! YVEX_TEST_FILTER=__unknown_runtime_test__ $(TEST_RUNNER) >/dev/null 2>&1
	@! YVEX_TEST_FILTER=runtime_benchmark,runtime_benchmark \
		$(TEST_RUNNER) >/dev/null 2>&1

test-runtime-asan:
	@set -eu; \
	tmp_tag=runtime-asan; \
	$(ATTENTION_OWNED_TMP_BEGIN) \
	build_dir="$$tmp_dir/build"; \
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
	$(MAKE) BUILD_DIR="$$build_dir" \
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=address,leak' \
		LDFLAGS='$(LDFLAGS) -fsanitize=address,leak' test-runtime; \
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
		NVCC=__yvex_nvcc_unavailable__ \
		CFLAGS='$(CFLAGS) -O1 -g -fno-omit-frame-pointer -fsanitize=undefined \
			-fno-sanitize-recover=undefined' \
		LDFLAGS='$(LDFLAGS) -fsanitize=undefined' test-runtime; \
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

test-artifact-writer: $(ARTIFACT_TEST_RUNNER)
	$(ARTIFACT_TEST_RUNNER)

# Replays the writer suite as the explicit preallocation/IO/protocol fault lane.
test-artifact-writer-fault: $(ARTIFACT_TEST_RUNNER)
	$(ARTIFACT_TEST_RUNNER)

test-quant-live-plan: $(QUANT_LIVE_RUNNER)
	$(QUANT_LIVE_RUNNER) --plan-only "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)"

test-quant-live: $(QUANT_LIVE_RUNNER)
	$(QUANT_LIVE_RUNNER) "$(DEEPSEEK_SOURCE)" "$(DEEPSEEK_MODELS_ROOT)" "$(DEEPSEEK_SOURCE_MANIFEST)" "$(DEEPSEEK_SELECTED_ARTIFACT)"

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
	sh tests/test_gguf_artifact_abi.sh

test-gguf-layout-integrity: $(TEST_RUNNER) tests/test_gguf_layout_integrity.sh
	sh tests/test_gguf_layout_integrity.sh

test-gguf-qtype-abi: $(TEST_RUNNER) tests/test_gguf_qtype_abi.sh
	sh tests/test_gguf_qtype_abi.sh

test-layout: tests/test_source_layout.sh
	sh tests/test_source_layout.sh

test-code-natural: tests/test_code_natural.sh
	sh tests/test_code_natural.sh

test-project-ledger: tests/test_project_ledger.sh PROJECT.md
	sh tests/test_project_ledger.sh

test-docs-surface: tests/test_docs_surface.sh
	sh tests/test_docs_surface.sh

test-surface: tests/test_surface.sh
	sh tests/test_surface.sh

test-source-ownership: tests/test_source_ownership.sh config/source_owners.tsv
	sh tests/test_source_ownership.sh

test-repository-layout: tests/test_repository_layout.sh Makefile
	sh tests/test_repository_layout.sh

test-architecture-boundaries: $(LIBYVEX) $(YVEX_BIN) $(TEST_REFERENCE_OBJS) tests/test_architecture_boundaries.sh
	YVEX_LIB="$(LIBYVEX)" YVEX_BIN="$(YVEX_BIN)" \
		YVEX_REFERENCE_OBJS="$(TEST_REFERENCE_OBJS)" \
		sh tests/test_architecture_boundaries.sh

smoke: test-cli

check: check-docs check-guardrails lib cli server test test-cuda-no-nvcc test-gguf-artifact-abi test-gguf-layout-integrity test-gguf-qtype-abi test-layout test-code-natural test-project-ledger test-docs-surface test-surface test-source-ownership test-repository-layout test-architecture-boundaries smoke
	@echo "yvex check: ok"

$(LIBYVEX): $(CORE_OBJS)
	@mkdir -p $(@D)
	rm -f $@
	$(AR) rcsP $@ $^

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

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

$(OBJ_DIR)/%.ptx: %.cu include/yvex/qtype.h
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_ARCH_FLAG) -ptx $< -o $@

$(CUDA_PTX_INC): $(CUDA_PTX)
	@mkdir -p $(@D)
	@{ \
		printf 'static const unsigned char cuda_kernels_ptx[] = {\n'; \
		{ cat $(CUDA_PTX); printf '\0'; } | xxd -i; \
		printf '};\n'; \
	} > $@

$(YVEX_BIN): $(CLI_OBJS) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CLI_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(YVEXD_BIN): $(DAEMON_OBJ) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DAEMON_OBJ) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_RUNNER): $(TEST_MAIN_OBJ) $(TEST_UNIT_OBJS) $(TEST_REFERENCE_OBJS) $(LIBYVEX) tests/test.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(TEST_MAIN_OBJ) $(TEST_UNIT_OBJS) $(TEST_REFERENCE_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(QUANT_TEST_RUNNER): $(QUANT_TEST_RUNNER_OBJ) $(QUANT_TEST_UNIT_OBJS) $(LIBYVEX) tests/test.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(QUANT_TEST_RUNNER_OBJ) $(QUANT_TEST_UNIT_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

$(ARTIFACT_TEST_RUNNER): $(ARTIFACT_TEST_RUNNER_OBJ) $(OBJ_DIR)/tests/unit/quant_execute.o $(LIBYVEX) tests/test.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(ARTIFACT_TEST_RUNNER_OBJ) $(OBJ_DIR)/tests/unit/quant_execute.o $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

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

$(ATTENTION_LIVE_RUNNER): $(ATTENTION_LIVE_OBJ) $(TEST_REFERENCE_OBJS) $(LIBYVEX)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(ATTENTION_LIVE_OBJ) $(TEST_REFERENCE_OBJS) $(LIBYVEX) $(LDFLAGS) $(LDLIBS) -o $@

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

check-docs:
	@test -f README.md
	@test -f NOTICE.md
	@test -f AGENTS.md
	@test -f PROJECT.md
	@test -f MODEL_ARTIFACTS.md
	@test ! -e docs/spine.md
	@test -f docs/api.md
	@test -f docs/contract.md
	@test -f docs/model-families.md
	@test -f docs/operator-runbook.md
	@test -f docs/v010-release-doctrine.md
	@test -f docs/topology-closure-audit.md
	@test -f docs/system-target.md
	@test -f docs/reference-architecture.md
	@test -z "$$(find docs -maxdepth 1 -type d -name repair -print -quit)"
	@! find docs -maxdepth 1 -type f -name '*.md' \
		! -name api.md \
		! -name contract.md \
		! -name model-families.md \
		! -name operator-runbook.md \
		! -name cli-output-architecture.md \
		! -name v010-release-doctrine.md \
		! -name topology-closure-audit.md \
		! -name system-target.md \
		! -name reference-architecture.md \
		-print | grep .
	@grep -F "YVEX Project Control" PROJECT.md >/dev/null
	@grep -F "## 7. Track Registry And Dashboard" PROJECT.md >/dev/null
	@grep -F "## 8. First-Class Milestone Roadmap" PROJECT.md >/dev/null
	@grep -F "## 9. Complete Track/Wave Ledger" PROJECT.md >/dev/null
	@grep -E '^# YVEX$$' README.md >/dev/null
	@grep -F "Transformation IR" README.md >/dev/null
	@grep -F "PROJECT.md" README.md >/dev/null
	@sh tests/test_project_ledger.sh >/dev/null
	@grep -F "YVEX System Target" docs/system-target.md >/dev/null
	@grep -F "Reference Architecture for Verified Transformer Inference" \
		docs/reference-architecture.md >/dev/null
	@grep -F "YVEX API" docs/api.md >/dev/null
	@grep -F "YVEX Runtime Contract" docs/contract.md >/dev/null
	@grep -F "YVEX Operator Runbook" docs/operator-runbook.md >/dev/null

check-guardrails: $(LIBYVEX) $(YVEX_BIN) $(TEST_REFERENCE_OBJS)
	@sh tests/test_source_ownership.sh
	@sh tests/test_repository_layout.sh
	@YVEX_LIB="$(LIBYVEX)" YVEX_BIN="$(YVEX_BIN)" \
		YVEX_REFERENCE_OBJS="$(TEST_REFERENCE_OBJS)" \
		sh tests/test_architecture_boundaries.sh
	@test ! -e docs/spine.md
	@test ! -d docs/spines
	@test ! -d docs/integration
	@test ! -d docs/benchmark
	@test ! -e docs/README.md
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
	@test -d src/cli/catalog
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
	@test -f src/daemon/yvexd.c
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
	rm -f -- ./yvex ./yvexd ./*.o

#define _GNU_SOURCE

/*
 * Reserve stable graph-state addresses without committing the model context envelope.
 *
 * Attention kernels still consume contiguous spans.  Anonymous virtual mappings preserve that
 * ABI while PROT_NONE makes page admission explicit: a caller must commit the model-derived
 * semantic page before the first read or write.  The shared pool owns the physical budget and
 * therefore refuses growth before state mutation rather than relying on an eventual OOM fault.
 */
#include "src/graph/private.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

struct yvex_graph_state_page_pool {
    pthread_mutex_t mutex;
    unsigned long long maximum_bytes, metadata_bytes, resident_bytes;
    unsigned long long virtual_bytes, page_count, resident_page_count;
    unsigned long long page_commit_count, page_release_count, store_count;
    int mutex_ready;
};

struct yvex_graph_state_page_store {
    yvex_graph_state_page_pool *pool;
    void *address;
    unsigned char *committed, *system_committed;
    unsigned long long element_count, element_width, page_elements;
    unsigned long long requested_bytes, mapped_bytes, page_count;
    unsigned long long committed_bytes, system_page_count;
    unsigned long long system_committed_bytes, resident_system_pages;
    unsigned long long metadata_charge, resident_bytes, resident_pages;
    unsigned long long system_page_bytes;
    int paged;
};

static int pages_reject(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "graph.state.pages", reason);
    return status;
}

static int pages_lock(yvex_graph_state_page_pool *pool, yvex_error *err)
{
    if (pool && pool->mutex_ready && pthread_mutex_lock(&pool->mutex) == 0)
        return YVEX_OK;
    return pages_reject(err, YVEX_ERR_STATE,
                        "state page-pool synchronization is unavailable");
}

static int pages_add(unsigned long long left, unsigned long long right,
                     unsigned long long *result)
{
    return yvex_core_u64_add(left, right, result);
}

static int pages_pool_admit_locked(const yvex_graph_state_page_pool *pool,
                                   unsigned long long metadata,
                                   unsigned long long resident)
{
    unsigned long long current, added, required;

    if (!pages_add(pool->metadata_bytes, pool->resident_bytes, &current) ||
        !pages_add(metadata, resident, &added) ||
        !pages_add(current, added, &required))
        return 0;
    return !pool->maximum_bytes || required <= pool->maximum_bytes;
}

int yvex_graph_state_page_pool_open(yvex_graph_state_page_pool **out,
                                    unsigned long long maximum_bytes,
                                    yvex_error *err)
{
    yvex_graph_state_page_pool *pool;

    if (out) *out = NULL;
    if (!out)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state page-pool output is required");
    pool = (yvex_graph_state_page_pool *)calloc(1u, sizeof(*pool));
    if (!pool)
        return pages_reject(err, YVEX_ERR_NOMEM,
                            "state page-pool allocation failed");
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool);
        return pages_reject(err, YVEX_ERR_STATE,
                            "state page-pool mutex initialization failed");
    }
    pool->mutex_ready = 1;
    pool->maximum_bytes = maximum_bytes;
    *out = pool;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int pages_pool_reserve(yvex_graph_state_page_pool *pool,
                              unsigned long long bytes, yvex_error *err)
{
    int rc = pages_lock(pool, err);

    if (rc != YVEX_OK) return rc;
    if (!pages_pool_admit_locked(pool, bytes, 0ull) ||
        !pages_add(pool->metadata_bytes, bytes, &pool->metadata_bytes))
        rc = pages_reject(err, YVEX_ERR_BOUNDS,
                          "state page-pool metadata exceeds its host budget");
    (void)pthread_mutex_unlock(&pool->mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_graph_state_page_pool_bind_capacity(
    yvex_graph_state_page_pool *pool,
    const yvex_execution_capacity_plan *capacity, yvex_error *err)
{
    unsigned long long budget, required;
    int rc = pages_lock(pool, err);

    if (rc != YVEX_OK) return rc;
    if (!capacity ||
        !pages_add(capacity->state_pool_bytes,
                   capacity->candidate_reserve_bytes, &budget) ||
        !pages_add(pool->metadata_bytes, pool->resident_bytes, &required)) {
        rc = pages_reject(err, YVEX_ERR_INVALID_ARG,
                          "state capacity budget is invalid");
    } else if (budget && required > budget) {
        rc = pages_reject(err, YVEX_ERR_BOUNDS,
                          "state capacity is below committed page bytes");
    } else if (budget &&
               (!pool->maximum_bytes || budget < pool->maximum_bytes)) {
        pool->maximum_bytes = budget;
    }
    (void)pthread_mutex_unlock(&pool->mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

void yvex_graph_state_page_pool_release(yvex_graph_state_page_pool *pool,
                                        unsigned long long bytes)
{
    if (!pool || !pool->mutex_ready || pthread_mutex_lock(&pool->mutex) != 0)
        return;
    if (bytes <= pool->metadata_bytes) pool->metadata_bytes -= bytes;
    (void)pthread_mutex_unlock(&pool->mutex);
}

static int pages_geometry(unsigned long long elements, size_t width,
                          unsigned long long page_elements,
                          unsigned long long system_page,
                          unsigned long long *requested,
                          unsigned long long *mapped,
                          unsigned long long *pages)
{
    unsigned long long adjusted;

    if (!elements || !width || !page_elements || !system_page ||
        !yvex_core_u64_mul(elements, (unsigned long long)width, requested) ||
        !pages_add(elements, page_elements - 1ull, &adjusted))
        return 0;
    *pages = adjusted / page_elements;
    if (!pages_add(*requested, system_page - 1ull, &adjusted)) return 0;
    *mapped = adjusted - adjusted % system_page;
    return *requested <= SIZE_MAX && *mapped <= SIZE_MAX && *pages <= SIZE_MAX;
}

static int pages_store_register(yvex_graph_state_page_store *store,
                                unsigned long long metadata,
                                unsigned long long resident,
                                yvex_error *err)
{
    yvex_graph_state_page_pool *pool = store->pool;
    unsigned long long metadata_next, resident_next, virtual_next;
    unsigned long long page_next, resident_page_next, stores_next;
    int rc = pages_lock(pool, err);

    if (rc != YVEX_OK) return rc;
    if (!pages_pool_admit_locked(pool, metadata, resident) ||
        !pages_add(pool->metadata_bytes, metadata, &metadata_next) ||
        !pages_add(pool->resident_bytes, resident, &resident_next) ||
        !pages_add(pool->virtual_bytes, store->requested_bytes, &virtual_next) ||
        !pages_add(pool->page_count, store->page_count, &page_next) ||
        !pages_add(pool->resident_page_count,
                   store->paged ? 0ull : store->page_count,
                   &resident_page_next) ||
        !pages_add(pool->store_count, 1ull, &stores_next)) {
        rc = pages_reject(err, YVEX_ERR_BOUNDS,
                          "state page store exceeds its pool budget or counters");
    } else {
        pool->metadata_bytes = metadata_next;
        pool->resident_bytes = resident_next;
        pool->virtual_bytes = virtual_next;
        pool->page_count = page_next;
        pool->resident_page_count = resident_page_next;
        pool->store_count = stores_next;
    }
    (void)pthread_mutex_unlock(&pool->mutex);
    return rc;
}

static int pages_store_open(
    yvex_graph_state_page_pool *pool, yvex_graph_state_page_store **out,
    unsigned long long element_count, size_t element_width,
    unsigned long long page_elements, int paged, void **address,
    yvex_error *err)
{
    yvex_graph_state_page_store *store = NULL;
    long system_page;
    unsigned long long bitmap_bytes = 0ull, system_bitmap_bytes = 0ull;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    int rc;

    if (out) *out = NULL;
    if (address) *address = NULL;
    if (!pool || !out || !address || !element_count || !element_width ||
        !page_elements || (paged != 0 && paged != 1))
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "complete state page-store geometry is required");
    system_page = sysconf(_SC_PAGESIZE);
    if (system_page <= 0)
        return pages_reject(err, YVEX_ERR_STATE,
                            "host page geometry is unavailable");
    store = (yvex_graph_state_page_store *)calloc(1u, sizeof(*store));
    if (!store)
        return pages_reject(err, YVEX_ERR_NOMEM,
                            "state page-store owner allocation failed");
    store->pool = pool;
    store->element_count = element_count;
    store->element_width = (unsigned long long)element_width;
    store->page_elements = page_elements;
    store->system_page_bytes = (unsigned long long)system_page;
    store->paged = paged;
    if (!pages_geometry(element_count, element_width, page_elements,
                        store->system_page_bytes, &store->requested_bytes,
                        &store->mapped_bytes, &store->page_count)) {
        free(store);
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state page-store geometry overflowed");
    }
    if (paged) {
        bitmap_bytes = (store->page_count + 7ull) / 8ull;
        store->system_page_count =
            store->mapped_bytes / store->system_page_bytes;
        system_bitmap_bytes = (store->system_page_count + 7ull) / 8ull;
        store->committed = (unsigned char *)calloc((size_t)bitmap_bytes, 1u);
        store->system_committed =
            (unsigned char *)calloc((size_t)system_bitmap_bytes, 1u);
#ifdef MAP_NORESERVE
        flags |= MAP_NORESERVE;
#endif
        store->address = mmap(NULL, (size_t)store->mapped_bytes, PROT_NONE,
                              flags, -1, 0);
        if (!store->committed || !store->system_committed ||
            store->address == MAP_FAILED) {
            if (store->address != MAP_FAILED && store->address)
                (void)munmap(store->address, (size_t)store->mapped_bytes);
            free(store->committed);
            free(store->system_committed);
            free(store);
            return pages_reject(err, YVEX_ERR_NOMEM,
                                "state virtual page reservation failed");
        }
        store->committed_bytes = bitmap_bytes;
        store->system_committed_bytes = system_bitmap_bytes;
        if (!pages_add(bitmap_bytes, system_bitmap_bytes,
                       &store->metadata_charge)) {
            (void)munmap(store->address, (size_t)store->mapped_bytes);
            free(store->committed);
            free(store->system_committed);
            free(store);
            return pages_reject(err, YVEX_ERR_BOUNDS,
                                "state page metadata extent overflowed");
        }
    } else {
        store->address = calloc(1u, (size_t)store->requested_bytes);
        if (!store->address) {
            free(store);
            return pages_reject(err, YVEX_ERR_NOMEM,
                                "reference state storage allocation failed");
        }
        store->resident_bytes = store->requested_bytes;
        store->resident_pages = store->page_count;
    }
    rc = pages_store_register(store, store->metadata_charge,
                              store->resident_bytes, err);
    if (rc != YVEX_OK) {
        if (store->paged)
            (void)munmap(store->address, (size_t)store->mapped_bytes);
        else
            free(store->address);
        free(store->committed);
        free(store->system_committed);
        free(store);
        return rc;
    }
    *address = store->address;
    *out = store;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int pages_bit_get(const unsigned char *bitmap,
                         unsigned long long page)
{
    return (bitmap[page / 8ull] >> (page % 8ull)) & 1u;
}

static void pages_bit_set(unsigned char *bitmap,
                          unsigned long long page)
{
    bitmap[page / 8ull] |= (unsigned char)(1u << (page % 8ull));
}

static int pages_store_touch(
    yvex_graph_state_page_store *store, unsigned long long element_start,
    unsigned long long element_count, yvex_error *err)
{
    yvex_graph_state_page_pool *pool;
    unsigned long long end, first, last, page, new_pages = 0ull;
    unsigned long long system_first, system_last, new_system_pages = 0ull;
    unsigned long long additional;
    unsigned long long byte_start, byte_end, protect_start, protect_end;
    unsigned long long pool_resident_next, pool_pages_next, commits_next;
    unsigned long long store_resident_next, store_pages_next, system_pages_next;
    int rc;

    if (!store || !element_count || element_start >= store->element_count ||
        !pages_add(element_start, element_count, &end) ||
        end > store->element_count)
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state page touch exceeds its virtual extent");
    if (!store->paged) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    first = element_start / store->page_elements;
    last = (end - 1ull) / store->page_elements;
    if (!yvex_core_u64_mul(first, store->page_elements, &byte_start) ||
        !yvex_core_u64_mul(byte_start, store->element_width, &byte_start) ||
        !pages_add(last, 1ull, &byte_end) ||
        !yvex_core_u64_mul(byte_end, store->page_elements, &byte_end) ||
        !yvex_core_u64_mul(byte_end, store->element_width, &byte_end))
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state page protection range overflowed");
    if (byte_end > store->requested_bytes) byte_end = store->requested_bytes;
    protect_start = byte_start - byte_start % store->system_page_bytes;
    if (!pages_add(byte_end, store->system_page_bytes - 1ull, &protect_end))
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state page protection rounding overflowed");
    protect_end -= protect_end % store->system_page_bytes;
    system_first = protect_start / store->system_page_bytes;
    system_last = protect_end / store->system_page_bytes - 1ull;
    pool = store->pool;
    rc = pages_lock(pool, err);
    if (rc != YVEX_OK) return rc;
    for (page = first;; ++page) {
        if (!pages_bit_get(store->committed, page) &&
            !pages_add(new_pages, 1ull, &new_pages)) {
                rc = pages_reject(err, YVEX_ERR_BOUNDS,
                                  "state page commitment accounting overflowed");
                goto done;
        }
        if (page == last) break;
    }
    for (page = system_first;; ++page) {
        if (!pages_bit_get(store->system_committed, page) &&
            !pages_add(new_system_pages, 1ull, &new_system_pages)) {
            rc = pages_reject(err, YVEX_ERR_BOUNDS,
                              "host page commitment accounting overflowed");
            goto done;
        }
        if (page == system_last) break;
    }
    if (!yvex_core_u64_mul(new_system_pages, store->system_page_bytes,
                           &additional)) {
        rc = pages_reject(err, YVEX_ERR_BOUNDS,
                          "host page commitment byte extent overflowed");
        goto done;
    }
    if (!pages_pool_admit_locked(pool, 0ull, additional) ||
        !pages_add(pool->resident_bytes, additional, &pool_resident_next) ||
        !pages_add(pool->resident_page_count, new_pages, &pool_pages_next) ||
        !pages_add(pool->page_commit_count, new_pages, &commits_next) ||
        !pages_add(store->resident_bytes, additional, &store_resident_next) ||
        !pages_add(store->resident_pages, new_pages, &store_pages_next) ||
        !pages_add(store->resident_system_pages, new_system_pages,
                   &system_pages_next)) {
        rc = pages_reject(err, YVEX_ERR_BOUNDS,
                          "state page commitment exceeds its budget or counters");
        goto done;
    }
    if (mprotect((unsigned char *)store->address + (size_t)protect_start,
                 (size_t)(protect_end - protect_start),
                 PROT_READ | PROT_WRITE) != 0) {
        rc = pages_reject(err, YVEX_ERR_NOMEM,
                          "state page commitment failed");
        goto done;
    }
    /* Fault each newly admitted OS page before publication can depend on its backing. */
    for (page = system_first;; ++page) {
        if (!pages_bit_get(store->system_committed, page)) {
            volatile unsigned char *byte =
                (volatile unsigned char *)store->address +
                (size_t)(page * store->system_page_bytes);
            *byte = 0u;
        }
        if (page == system_last) break;
    }
    for (page = first;; ++page) {
        if (!pages_bit_get(store->committed, page))
            pages_bit_set(store->committed, page);
        if (page == last) break;
    }
    for (page = system_first;; ++page) {
        if (!pages_bit_get(store->system_committed, page))
            pages_bit_set(store->system_committed, page);
        if (page == system_last) break;
    }
    pool->resident_bytes = pool_resident_next;
    pool->resident_page_count = pool_pages_next;
    pool->page_commit_count = commits_next;
    store->resident_bytes = store_resident_next;
    store->resident_pages = store_pages_next;
    store->resident_system_pages = system_pages_next;
    rc = YVEX_OK;
done:
    (void)pthread_mutex_unlock(&pool->mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

static int pages_store_reset(yvex_graph_state_page_store *store,
                             yvex_error *err)
{
    yvex_graph_state_page_pool *pool;
    unsigned long long releases_next;
    int rc;

    if (!store)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state page store is required");
    if (!store->paged) {
        memset(store->address, 0, (size_t)store->requested_bytes);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    pool = store->pool;
    rc = pages_lock(pool, err);
    if (rc != YVEX_OK) return rc;
    if (!pages_add(pool->page_release_count, store->resident_pages,
                   &releases_next)) {
        rc = pages_reject(err, YVEX_ERR_BOUNDS,
                          "state page release counter overflowed");
    } else if (madvise(store->address, (size_t)store->mapped_bytes,
                MADV_DONTNEED) != 0 ||
        mprotect(store->address, (size_t)store->mapped_bytes, PROT_NONE) != 0) {
        rc = pages_reject(err, YVEX_ERR_STATE,
                          "state page release failed");
    } else {
        memset(store->committed, 0, (size_t)store->committed_bytes);
        memset(store->system_committed, 0,
               (size_t)store->system_committed_bytes);
        pool->resident_bytes -= store->resident_bytes;
        pool->resident_page_count -= store->resident_pages;
        pool->page_release_count = releases_next;
        store->resident_bytes = 0ull;
        store->resident_pages = 0ull;
        store->resident_system_pages = 0ull;
        rc = YVEX_OK;
    }
    (void)pthread_mutex_unlock(&pool->mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

static void pages_store_close(yvex_graph_state_page_store **owner)
{
    yvex_graph_state_page_store *store = owner ? *owner : NULL;
    yvex_graph_state_page_pool *pool;
    unsigned long long releases_next;

    if (!store) return;
    pool = store->pool;
    if (pool && pool->mutex_ready && pthread_mutex_lock(&pool->mutex) == 0) {
        releases_next = pool->page_release_count;
        (void)pages_add(releases_next, store->resident_pages, &releases_next);
        pool->metadata_bytes -= store->metadata_charge;
        pool->resident_bytes -= store->resident_bytes;
        pool->virtual_bytes -= store->requested_bytes;
        pool->page_count -= store->page_count;
        pool->resident_page_count -= store->resident_pages;
        pool->page_release_count = releases_next;
        pool->store_count--;
        (void)pthread_mutex_unlock(&pool->mutex);
    }
    if (store->paged)
        (void)munmap(store->address, (size_t)store->mapped_bytes);
    else
        free(store->address);
    free(store->committed);
    free(store->system_committed);
    memset(store, 0, sizeof(*store));
    free(store);
    *owner = NULL;
}

int yvex_graph_state_capacity_plan_valid(
    const yvex_execution_capacity_plan *capacity)
{
    unsigned long long index, blocks, expected_page_bytes;

    if (!capacity ||
        capacity->schema_version != YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1 ||
        !capacity->state_class_count ||
        capacity->state_class_count > YVEX_MODEL_STATE_CLASS_COUNT ||
        !capacity->per_session_maximum ||
        !yvex_sha256_hex_valid(capacity->model_execution_identity) ||
        !yvex_sha256_hex_valid(capacity->hardware_profile_identity) ||
        !yvex_sha256_hex_valid(capacity->workload_profile_identity) ||
        !yvex_sha256_hex_valid(capacity->identity))
        return 0;
    for (index = 0ull; index < capacity->state_class_count; ++index) {
        const yvex_execution_state_class_plan *state =
            &capacity->state_classes[index];

        if (state->state_class >= YVEX_MODEL_STATE_CLASS_COUNT ||
            (index && state->state_class <=
                          capacity->state_classes[index - 1ull].state_class) ||
            !state->logical_block_tokens || !state->bytes_per_block ||
            !state->page_tokens || !state->page_bytes ||
            state->page_tokens % state->logical_block_tokens ||
            !(blocks = state->page_tokens / state->logical_block_tokens) ||
            !yvex_core_u64_mul(blocks, state->bytes_per_block,
                               &expected_page_bytes) ||
            expected_page_bytes != state->page_bytes)
            return 0;
    }
    return 1;
}

static yvex_model_state_class pages_component_class(
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_component_recipe *recipe)
{
    if (summary->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT)
        return YVEX_MODEL_STATE_DRAFT_PERSISTENT;
    switch (recipe->binding) {
    case YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY:
        return YVEX_MODEL_STATE_SWA_RING;
    case YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY:
        return layer->attention_class == YVEX_ATTENTION_CLASS_HCA
                   ? YVEX_MODEL_STATE_HCA_HISTORY
                   : YVEX_MODEL_STATE_COMPRESSED_HISTORY;
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY:
        return YVEX_MODEL_STATE_INDEXER_HISTORY;
    case YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING:
        return YVEX_MODEL_STATE_MAIN_ROLLING;
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING:
        return YVEX_MODEL_STATE_INDEXER_ROLLING;
    default: return YVEX_MODEL_STATE_CLASS_COUNT;
    }
}

static const yvex_execution_state_class_plan *pages_component_plan(
    const yvex_execution_capacity_plan *capacity,
    yvex_model_state_class state_class)
{
    unsigned long long index;

    if (!capacity) return NULL;
    for (index = 0ull; index < capacity->state_class_count; ++index)
        if (capacity->state_classes[index].state_class == state_class)
            return &capacity->state_classes[index];
    return NULL;
}

static int pages_component_rows(
    const yvex_execution_capacity_plan *capacity,
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_component_recipe *recipe,
    unsigned long long *rows)
{
    const yvex_execution_state_class_plan *plan;
    unsigned long long period;

    if (!capacity) {
        *rows = recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY
                    ? recipe->capacity
                    : recipe->rolling.state_slots;
        return *rows != 0ull;
    }
    plan = pages_component_plan(
        capacity, pages_component_class(summary, layer, recipe));
    if (!plan || !plan->logical_block_tokens || !plan->page_tokens ||
        plan->page_tokens % plan->logical_block_tokens)
        return 0;
    /*
     * A class page spans model-token time. Its aggregate logical block is the LCM across
     * participating layers, while one compressed layer contributes a row at its own recurrence.
     * Dividing by the aggregate block would under-size stores when layers in the class use
     * different compression periods.
     */
    period = plan->logical_block_tokens;
    if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY &&
        (recipe->binding == YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY ||
         recipe->binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY))
        period = layer->compression_ratio;
    if (!period || plan->page_tokens % period) return 0;
    *rows = plan->page_tokens / period;
    return *rows != 0ull;
}

static void pages_component_close(
    yvex_graph_state_component_storage *storage)
{
    if (!storage) return;
    pages_store_close(&storage->value_pages);
    pages_store_close(&storage->position_pages);
    pages_store_close(&storage->auxiliary_pages);
    memset(storage, 0, sizeof(*storage));
}

static int pages_component_open(
    yvex_graph_state_page_pool *pool,
    const yvex_execution_capacity_plan *capacity,
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_component_recipe *recipe,
    yvex_graph_state_component_storage *storage, yvex_error *err)
{
    unsigned long long rows, value_count, value_page_elements;
    int paged = capacity != NULL;

    if (!pool || !summary || !layer || !recipe || !storage ||
        (capacity && !yvex_graph_state_capacity_plan_valid(capacity)) ||
        !pages_component_rows(capacity, summary, layer, recipe, &rows) ||
        !yvex_core_u64_mul(rows, recipe->value_width,
                           &value_page_elements))
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state component page geometry is invalid");
    memset(storage, 0, sizeof(*storage));
    storage->recipe = *recipe;
    if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
        storage->allocated_rows = recipe->capacity;
        if (recipe->binding == YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY &&
            !yvex_core_u64_mul(storage->allocated_rows, 2ull,
                               &storage->allocated_rows))
            goto overflow;
        if (!yvex_core_u64_mul(storage->allocated_rows,
                               recipe->value_width, &value_count))
            goto overflow;
        if (pages_store_open(
                pool, &storage->value_pages, value_count, sizeof(float),
                value_page_elements, paged, (void **)&storage->values,
                err) != YVEX_OK ||
            pages_store_open(
                pool, &storage->position_pages, storage->allocated_rows,
                sizeof(unsigned long long), rows, paged,
                (void **)&storage->positions, err) != YVEX_OK)
            goto failed;
        return YVEX_OK;
    }
    if (pages_store_open(
            pool, &storage->value_pages, recipe->rolling.kv_state_extent,
            sizeof(float), value_page_elements, paged,
            (void **)&storage->values, err) != YVEX_OK ||
        pages_store_open(
            pool, &storage->auxiliary_pages,
            recipe->rolling.score_state_extent, sizeof(float),
            value_page_elements, paged, (void **)&storage->auxiliary,
            err) != YVEX_OK ||
        pages_store_touch(
            storage->value_pages, 0ull, recipe->rolling.kv_state_extent,
            err) != YVEX_OK ||
        pages_store_touch(
            storage->auxiliary_pages, 0ull,
            recipe->rolling.score_state_extent, err) != YVEX_OK)
        goto failed;
    return YVEX_OK;

overflow:
    pages_reject(err, YVEX_ERR_BOUNDS,
                 "state component page geometry overflowed");
failed:
    pages_component_close(storage);
    return (int)yvex_error_code(err);
}

static int pages_component_reset(
    yvex_graph_state_component_storage *storage, yvex_error *err)
{
    if (!storage || !storage->value_pages)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "prepared state component pages are required");
    if (pages_store_reset(storage->value_pages, err) != YVEX_OK ||
        (storage->position_pages &&
         pages_store_reset(
             storage->position_pages, err) != YVEX_OK) ||
        (storage->auxiliary_pages &&
         pages_store_reset(
             storage->auxiliary_pages, err) != YVEX_OK))
        return (int)yvex_error_code(err);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int pages_component_touch_history(
    yvex_graph_state_component_storage *storage,
    unsigned long long row_start, unsigned long long row_count,
    yvex_error *err)
{
    unsigned long long value_start, value_count;

    if (!row_count) return YVEX_OK;
    if (!storage || !storage->position_pages ||
        !yvex_core_u64_mul(row_start, storage->recipe.value_width,
                           &value_start) ||
        !yvex_core_u64_mul(row_count, storage->recipe.value_width,
                           &value_count))
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state history page range overflowed");
    if (pages_store_touch(
            storage->value_pages, value_start, value_count, err) != YVEX_OK ||
        pages_store_touch(
            storage->position_pages, row_start, row_count, err) != YVEX_OK)
        return (int)yvex_error_code(err);
    return YVEX_OK;
}

static yvex_attention_rolling_state_view *pages_rolling_view(
    yvex_attention_history_view *view, yvex_attention_state_binding binding)
{
    if (binding == YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING)
        return &view->main_rolling_state;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING)
        return &view->indexer_rolling_state;
    return NULL;
}

static const yvex_attention_rolling_state_view *pages_rolling_view_const(
    const yvex_attention_history_view *view,
    yvex_attention_state_binding binding)
{
    if (binding == YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING)
        return &view->main_rolling_state;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING)
        return &view->indexer_rolling_state;
    return NULL;
}

typedef struct {
    const float *values;
    const unsigned long long *positions;
    unsigned long long count, width;
} pages_history_span;

static pages_history_span pages_history_project(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component)
{
    if (component->binding == YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY)
        return (pages_history_span){view->local_kv, view->local_positions,
                                    view->local_tail_count,
                                    component->value_width};
    if (component->binding ==
        YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY)
        return (pages_history_span){view->compressed_kv,
                                    view->compressed_positions,
                                    view->compressed_entry_count,
                                    component->value_width};
    if (component->binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY)
        return (pages_history_span){view->indexer_kv, view->indexer_positions,
                                    view->indexer_entry_count,
                                    component->value_width};
    return (pages_history_span){0};
}

void yvex_graph_state_bank_pages_bind(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe)
{
    unsigned int index;

    if (!components || !view || !recipe) return;
    for (index = 0u; index < recipe->component_count; ++index) {
        const yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        yvex_graph_state_component_storage *storage =
            &components[component->binding];
        yvex_attention_rolling_state_view *rolling;

        if (component->kind == YVEX_ATTENTION_STATE_COMPONENT_ROLLING) {
            rolling = pages_rolling_view(view, component->binding);
            if (rolling) {
                rolling->kv_state = storage->values;
                rolling->score_state = storage->auxiliary;
            }
        } else if (component->binding ==
                   YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) {
            view->local_kv = storage->values +
                             storage->start * component->value_width;
            view->local_positions = storage->positions + storage->start;
            view->local_kv_stride = component->value_width;
        } else if (component->binding ==
                   YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) {
            view->compressed_kv = storage->values;
            view->compressed_positions = storage->positions;
            view->compressed_kv_stride = component->value_width;
        } else if (component->binding ==
                   YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY) {
            view->indexer_kv = storage->values;
            view->indexer_positions = storage->positions;
            view->indexer_kv_stride = component->value_width;
        }
    }
    view->immutable = 1;
}

int yvex_graph_state_bank_pages_transfer(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_history_view *source,
    int validate_storage, yvex_error *err)
{
    yvex_attention_history_view allocated;
    unsigned int index;

    if (!components || !view || !recipe || !source)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state history transfer arguments are invalid");
    allocated = *view;
    *view = *source;
    for (index = 0u; index < recipe->component_count; ++index) {
        const yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        yvex_graph_state_component_storage *storage =
            &components[component->binding];

        if (component->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            pages_history_span span = pages_history_project(source, component);
            unsigned long long count;

            if (!yvex_core_u64_mul(span.count, span.width, &count) ||
                (validate_storage &&
                 (span.count > component->capacity ||
                  (count && !span.values) ||
                  (span.count && !span.positions))))
                return pages_reject(err, YVEX_ERR_FORMAT,
                                    "state history transfer is malformed");
            storage->start = 0ull;
            if (span.count &&
                (pages_component_touch_history(
                     storage, 0ull, span.count, err) != YVEX_OK ||
                 (component->binding ==
                      YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY &&
                  pages_component_touch_history(
                      storage, component->capacity, span.count,
                      err) != YVEX_OK)))
                return (int)yvex_error_code(err);
            if (count)
                memcpy(storage->values, span.values,
                       (size_t)count * sizeof(float));
            if (span.count)
                memcpy(storage->positions, span.positions,
                       (size_t)span.count * sizeof(unsigned long long));
            if (component->binding ==
                YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) {
                if (count)
                    memcpy(storage->values +
                               component->capacity * component->value_width,
                           span.values, (size_t)count * sizeof(float));
                if (span.count)
                    memcpy(storage->positions + component->capacity,
                           span.positions,
                           (size_t)span.count * sizeof(unsigned long long));
            }
            continue;
        }
        {
            const yvex_attention_rolling_state_view *input =
                pages_rolling_view_const(source, component->binding);
            const yvex_attention_rolling_state_view *target =
                pages_rolling_view(&allocated, component->binding);
            yvex_attention_rolling_state_view *output =
                pages_rolling_view(view, component->binding);

            if (!input || !target || !output)
                return pages_reject(err, YVEX_ERR_STATE,
                                    "state rolling transfer is unbound");
            if (!input->present) {
                if (!validate_storage) *output = *target;
                continue;
            }
            if (validate_storage &&
                (!storage->values || !storage->auxiliary || !input->kv_state ||
                 !input->score_state ||
                 input->kv_state_extent != target->kv_state_extent ||
                 input->score_state_extent != target->score_state_extent))
                return pages_reject(err, YVEX_ERR_FORMAT,
                                    "state rolling transfer is malformed");
            if (pages_store_touch(
                    storage->value_pages, 0ull, input->kv_state_extent,
                    err) != YVEX_OK ||
                pages_store_touch(
                    storage->auxiliary_pages, 0ull,
                    input->score_state_extent, err) != YVEX_OK)
                return (int)yvex_error_code(err);
            *output = *input;
            memcpy(storage->values, input->kv_state,
                   (size_t)input->kv_state_extent * sizeof(float));
            memcpy(storage->auxiliary, input->score_state,
                   (size_t)input->score_state_extent * sizeof(float));
        }
    }
    yvex_graph_state_bank_pages_bind(components, view, recipe);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_graph_state_bank_pages_close(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT])
{
    unsigned int index;

    if (!components) return;
    for (index = 0u; index < YVEX_ATTENTION_STATE_BINDING_COUNT; ++index)
        pages_component_close(&components[index]);
}

int yvex_graph_state_bank_pages_open(
    yvex_graph_state_page_pool *pool,
    const yvex_execution_capacity_plan *capacity,
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_recipe *recipe,
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view, yvex_error *err)
{
    unsigned int index;

    if (!pool || !summary || !layer || !recipe || !components || !view)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state bank page owners are required");
    memset(components, 0, sizeof(*components) *
                              YVEX_ATTENTION_STATE_BINDING_COUNT);
    memset(view, 0, sizeof(*view));
    for (index = 0u; index < recipe->component_count; ++index) {
        const yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        yvex_graph_state_component_storage *storage =
            &components[component->binding];
        yvex_attention_rolling_state_view *rolling;
        unsigned long long element;

        if (pages_component_open(
                pool, capacity, summary, layer, component, storage,
                err) != YVEX_OK)
            goto failed;
        if (component->kind != YVEX_ATTENTION_STATE_COMPONENT_ROLLING)
            continue;
        rolling = pages_rolling_view(view, component->binding);
        if (!rolling) {
            pages_reject(err, YVEX_ERR_STATE,
                         "state rolling page binding is invalid");
            goto failed;
        }
        *rolling = component->rolling;
        for (element = 0ull;
             element < component->rolling.score_state_extent; ++element)
            storage->auxiliary[element] = -INFINITY;
    }
    yvex_error_clear(err);
    return YVEX_OK;

failed:
    yvex_graph_state_bank_pages_close(components);
    memset(view, 0, sizeof(*view));
    return (int)yvex_error_code(err);
}

int yvex_graph_state_bank_pages_reset(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe, yvex_error *err)
{
    unsigned int index;

    if (!components || !view || !recipe)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "prepared state bank pages are required");
    view->token_count = view->local_tail_count =
        view->compressed_entry_count = view->indexer_entry_count = 0ull;
    for (index = 0u; index < recipe->component_count; ++index) {
        const yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        yvex_graph_state_component_storage *storage =
            &components[component->binding];
        yvex_attention_rolling_state_view *rolling;
        unsigned long long element;

        if (pages_component_reset(storage, err) != YVEX_OK)
            return (int)yvex_error_code(err);
        if (component->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            storage->start = 0ull;
            continue;
        }
        rolling = pages_rolling_view(view, component->binding);
        if (!rolling ||
            pages_store_touch(
                storage->value_pages, 0ull, rolling->kv_state_extent,
                err) != YVEX_OK ||
            pages_store_touch(
                storage->auxiliary_pages, 0ull,
                rolling->score_state_extent, err) != YVEX_OK)
            return pages_reject(err, YVEX_ERR_STATE,
                                "state rolling pages could not be reset");
        for (element = 0ull; element < rolling->score_state_extent; ++element)
            storage->auxiliary[element] = -INFINITY;
        rolling->next_token_position = 0ull;
        rolling->previous_fill = rolling->current_fill = rolling->cursor = 0ull;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static unsigned long long *pages_history_count(
    yvex_attention_history_view *view, yvex_attention_state_binding binding)
{
    if (binding == YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY)
        return &view->local_tail_count;
    if (binding == YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY)
        return &view->compressed_entry_count;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY)
        return &view->indexer_entry_count;
    return NULL;
}

static void pages_publication_span(
    const yvex_attention_publication *publication,
    yvex_attention_state_binding binding, unsigned long long *count,
    int *position_backed)
{
    if (binding == YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) {
        *count = publication->token_count;
        *position_backed = 0;
    } else if (binding == YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) {
        *count = publication->compressed_count;
        *position_backed = publication->compressed_positions != NULL;
    } else {
        *count = publication->indexer_count;
        *position_backed = publication->indexer_positions != NULL;
    }
}

int yvex_graph_state_pages_prepare_publications(
    yvex_attention_history_view *view,
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_publication *const *publications,
    unsigned long long publication_count, yvex_error *err)
{
    unsigned int component;

    if (!view || !components || !recipe || !publications ||
        !publication_count)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state page publication inventory is required");
    for (component = 0u; component < recipe->component_count; ++component) {
        const yvex_attention_state_component_recipe *component_recipe =
            &recipe->components[component];
        yvex_graph_state_component_storage *storage =
            &components[component_recipe->binding];
        unsigned long long *visible_count;
        unsigned long long count, start, publication_index;

        if (component_recipe->kind != YVEX_ATTENTION_STATE_COMPONENT_HISTORY ||
            !component_recipe->capacity)
            continue;
        visible_count = pages_history_count(view, component_recipe->binding);
        if (!visible_count)
            return pages_reject(err, YVEX_ERR_STATE,
                                "state history count owner is unavailable");
        count = *visible_count;
        start = storage->start;
        for (publication_index = 0ull;
             publication_index < publication_count; ++publication_index) {
            unsigned long long remaining;
            int position_backed;

            pages_publication_span(
                publications[publication_index], component_recipe->binding,
                &remaining, &position_backed);
            while (remaining) {
                unsigned long long slot, contiguous;

                if (!position_backed && count == component_recipe->capacity) {
                    start = (start + 1ull) % component_recipe->capacity;
                    slot = (start + component_recipe->capacity - 1ull) %
                           component_recipe->capacity;
                } else {
                    if (count >= component_recipe->capacity)
                        return pages_reject(
                            err, YVEX_ERR_BOUNDS,
                            "position-backed state history exceeds capacity");
                    slot = (start + count) % component_recipe->capacity;
                    ++count;
                }
                contiguous = component_recipe->capacity - slot;
                if (contiguous > remaining) contiguous = remaining;
                if (!position_backed && count == component_recipe->capacity &&
                    contiguous > 1ull) {
                    start = (start + contiguous - 1ull) %
                            component_recipe->capacity;
                } else if (position_backed) {
                    count += contiguous - 1ull;
                } else if (count < component_recipe->capacity) {
                    unsigned long long available =
                        component_recipe->capacity - count;
                    unsigned long long advance = contiguous - 1ull;
                    if (advance > available) advance = available;
                    count += advance;
                    if (advance < contiguous - 1ull)
                        start = (start + contiguous - 1ull - advance) %
                                component_recipe->capacity;
                }
                if (pages_component_touch_history(
                        storage, slot, contiguous, err) != YVEX_OK ||
                    (!position_backed &&
                     pages_component_touch_history(
                         storage, slot + component_recipe->capacity,
                         contiguous, err) != YVEX_OK))
                    return (int)yvex_error_code(err);
                remaining -= contiguous;
            }
        }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_graph_state_pointer_table_reserve(
    yvex_graph_state_page_pool *pool, void ***table,
    unsigned long long *capacity, unsigned long long *bytes,
    unsigned long long limit, yvex_error *err)
{
    void **resized;
    unsigned long long next_capacity, next_bytes, delta;

    if (!pool || !table || !capacity || !bytes || !limit)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state pointer-table geometry is invalid");
    if (*capacity && !yvex_core_u64_mul(*capacity, 2ull, &next_capacity))
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state pointer-table capacity overflowed");
    if (!*capacity) next_capacity = 8ull;
    if (next_capacity > limit) next_capacity = limit;
    if (next_capacity <= *capacity ||
        !yvex_core_u64_mul(next_capacity, sizeof(void *), &next_bytes) ||
        next_bytes > SIZE_MAX || next_bytes < *bytes)
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state pointer-table byte extent overflowed");
    delta = next_bytes - *bytes;
    if (pages_pool_reserve(pool, delta, err) != YVEX_OK)
        return (int)yvex_error_code(err);
    resized = (void **)realloc(*table, (size_t)next_bytes);
    if (!resized) {
        yvex_graph_state_page_pool_release(pool, delta);
        return pages_reject(err, YVEX_ERR_NOMEM,
                            "state pointer-table allocation failed");
    }
    memset((unsigned char *)resized + (size_t)*bytes, 0, (size_t)delta);
    *table = resized;
    *capacity = next_capacity;
    *bytes = next_bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_graph_state_page_pool_summary(
    const yvex_graph_state_page_pool *pool,
    yvex_graph_state_page_summary *out, yvex_error *err)
{
    yvex_graph_state_page_pool *mutable_pool =
        (yvex_graph_state_page_pool *)pool;

    if (!pool || !out || !pool->mutex_ready ||
        pthread_mutex_lock(&mutable_pool->mutex) != 0)
        return pages_reject(err, YVEX_ERR_INVALID_ARG,
                            "state page pool and summary output are required");
    memset(out, 0, sizeof(*out));
    out->metadata_bytes = pool->metadata_bytes;
    out->resident_bytes = pool->resident_bytes;
    out->virtual_bytes = pool->virtual_bytes;
    out->page_count = pool->page_count;
    out->resident_page_count = pool->resident_page_count;
    out->page_commit_count = pool->page_commit_count;
    out->page_release_count = pool->page_release_count;
    if (!pages_add(out->metadata_bytes, out->resident_bytes,
                   &out->allocated_bytes)) {
        (void)pthread_mutex_unlock(&mutable_pool->mutex);
        return pages_reject(err, YVEX_ERR_BOUNDS,
                            "state page summary accounting overflowed");
    }
    (void)pthread_mutex_unlock(&mutable_pool->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_graph_state_page_pool_close(yvex_graph_state_page_pool **owner)
{
    yvex_graph_state_page_pool *pool = owner ? *owner : NULL;

    if (!pool) return;
    if (pool->mutex_ready) (void)pthread_mutex_destroy(&pool->mutex);
    memset(pool, 0, sizeof(*pool));
    free(pool);
    *owner = NULL;
}

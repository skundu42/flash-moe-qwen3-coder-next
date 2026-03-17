/*
 * fast_moe_load.c — FRESH mx.array expert weight loading for MoE inference
 *
 * Architecture: ALL I/O and data assembly in C, returns FRESH mx.arrays each call.
 *
 * The stale-buffer problem: pre-allocated mx.arrays that get pread'd into behind
 * MLX's back cause stale data issues (MLX caches Metal buffer contents).
 * Solution: create NEW mx.arrays every call. The I/O is 100% C (parallel pread
 * into a C staging buffer, GIL released). The mx.array creation uses Python C API
 * but is just memory allocation + memcpy — fast and unavoidable.
 *
 * Data flow:
 *   1. init(): allocate page-aligned staging buffer, open FDs, start worker pool
 *   2. load_and_assemble(routing_list):
 *      a. pread all expert data into C staging buffer (parallel, GIL released)
 *      b. For each (layer, component): create numpy array from staging slice,
 *         convert to mx.array via mx.array(np_arr), apply .view(bfloat16) if needed
 *      c. Build Python dicts, return list of 60 dicts
 *      All mx.arrays are FRESH allocations — MLX sees them as new data.
 *
 * API:
 *   init(num_workers, num_layers, K, components, packed_dir, expert_size)
 *   load_and_assemble(routing_list) -> list of dicts
 *   stats() -> dict
 *   shutdown()
 *
 * BF16 handling:
 *   Create as uint16 numpy array, convert to mx.array, then .view(mx.bfloat16).
 *   The .view() is zero-cost (reinterpret, no copy).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* numpy C API */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdatomic.h>

/* macOS QoS */
#include <sys/qos.h>
#include <pthread/qos.h>

/* ---- Constants ---- */

#define FML_MAX_LAYERS       64
#define FML_MAX_K            16      /* max active experts per token */
#define FML_MAX_COMPONENTS   9       /* gate/up/down x weight/scales/biases */
#define FML_MAX_WORKERS      32
#define FML_MAX_WORK_ITEMS   (FML_MAX_LAYERS * FML_MAX_K * FML_MAX_COMPONENTS)
#define FML_COMP_NAME_LEN    48
#define FML_PAGE_SIZE        16384   /* macOS ARM64 page size */

/* ---- Component spec ---- */

typedef struct {
    char name[FML_COMP_NAME_LEN];   /* e.g. "gate_proj.weight" */
    size_t offset;                   /* byte offset within expert block */
    size_t size;                     /* byte size per expert for this component */
    int ndim;
    int shape[4];                    /* per-expert shape (without K dimension) */
    int npy_dtype;                   /* NPY_UINT32, NPY_UINT16, etc. */
    int needs_bf16_view;             /* 1 if stored as uint16 but needs bfloat16 view */
} ComponentSpec;

/* ---- Per-layer file descriptor ---- */

typedef struct {
    int fd;
} LayerFd;

/* ---- Single pread work item ---- */

typedef struct {
    int    fd;
    void  *dest;        /* points into staging buffer */
    size_t nbytes;
    off_t  file_offset;
    int    error;
    ssize_t bytes_read;
} WorkItem;

/* ---- Worker thread context ---- */

typedef struct {
    pthread_t       thread;
    int             worker_id;
    int             running;

    WorkItem       *items;
    int             item_count;

    pthread_mutex_t work_mutex;
    pthread_cond_t  work_cond;
    int             has_work;

    pthread_mutex_t *done_mutex;
    pthread_cond_t  *done_cond;
    atomic_int      *completed_count;
} WorkerCtx;

/*
 * ---- Staging buffer layout ----
 *
 * One contiguous page-aligned buffer holds ALL expert data for one call.
 * Layout: for each layer entry, for each expert slot (0..K-1), for each component:
 *   staging[layer_entry_offset + slot * per_expert_size + comp_offset] = comp data
 *
 * But it's simpler to organize per (layer_entry, component, slot):
 *   Each (layer, comp) gets a contiguous [K, *shape] region.
 *   Region size = K * comp_size.
 *   Slot i within region: offset = i * comp_size.
 *
 * Total per layer entry: K * sum(comp_sizes) = K * expert_size
 * Total staging: num_layer_entries * K * expert_size
 *
 * Actually, we organize per (layer_entry, comp) so each region maps
 * directly to one [K, *shape] numpy array.
 */

/* ---- Module state ---- */

typedef struct {
    /* Worker pool */
    WorkerCtx      *workers;
    int             num_workers;

    /* Staging buffer */
    void           *staging;         /* page-aligned C buffer for pread */
    size_t          staging_size;    /* current allocation size */

    /* Component specs */
    ComponentSpec   comp_specs[FML_MAX_COMPONENTS];
    int             num_comps;
    size_t          expert_size;     /* total bytes per expert block in packed file */
    size_t          per_expert_comp_total; /* sum of all comp sizes (== expert_size usually) */

    /* Layer file descriptors */
    LayerFd         layer_fds[FML_MAX_LAYERS];
    int             num_layers;
    int             K;               /* experts per token (stacking dimension) */

    /* Cached Python objects for mx.array creation */
    PyObject       *mx_array_fn;     /* mx.array function */
    PyObject       *mx_eval_fn;      /* mx.eval function */
    PyObject       *mx_bfloat16;     /* mx.bfloat16 dtype object */
    PyObject       *comp_name_strs[FML_MAX_COMPONENTS]; /* interned Python strings */

    /* Completion sync */
    pthread_mutex_t done_mutex;
    pthread_cond_t  done_cond;
    atomic_int      completed_count;

    int             initialized;

    /* Stats */
    long long       total_loads;
    long long       total_bytes_read;
    long long       total_calls;
    double          total_io_ms;
    double          total_create_ms;
} ModuleState;

static ModuleState g_state = {0};

/* ---- Worker thread function ---- */

static void *worker_func(void *arg) {
    WorkerCtx *ctx = (WorkerCtx *)arg;

    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);

    while (ctx->running) {
        pthread_mutex_lock(&ctx->work_mutex);
        while (!ctx->has_work && ctx->running) {
            pthread_cond_wait(&ctx->work_cond, &ctx->work_mutex);
        }
        if (!ctx->running) {
            pthread_mutex_unlock(&ctx->work_mutex);
            break;
        }

        WorkItem *items = ctx->items;
        int count = ctx->item_count;
        ctx->has_work = 0;
        pthread_mutex_unlock(&ctx->work_mutex);

        /* Execute pread calls -- no GIL, no Python involvement */
        for (int i = 0; i < count; i++) {
            WorkItem *wi = &items[i];
            ssize_t nread = pread(wi->fd, wi->dest, wi->nbytes, wi->file_offset);
            if (nread < 0) {
                wi->error = errno;
                wi->bytes_read = -1;
            } else if ((size_t)nread < wi->nbytes) {
                /* Short read -- retry to complete */
                size_t total = (size_t)nread;
                while (total < wi->nbytes) {
                    ssize_t n = pread(wi->fd,
                                      (char *)wi->dest + total,
                                      wi->nbytes - total,
                                      wi->file_offset + (off_t)total);
                    if (n <= 0) {
                        wi->error = (n < 0) ? errno : EIO;
                        wi->bytes_read = (ssize_t)total;
                        break;
                    }
                    total += (size_t)n;
                }
                if (total == wi->nbytes) {
                    wi->error = 0;
                    wi->bytes_read = (ssize_t)total;
                }
            } else {
                wi->error = 0;
                wi->bytes_read = nread;
            }
        }

        /* Signal completion */
        atomic_fetch_add(ctx->completed_count, count);
        pthread_mutex_lock(ctx->done_mutex);
        pthread_cond_signal(ctx->done_cond);
        pthread_mutex_unlock(ctx->done_mutex);
    }

    return NULL;
}

/* ---- Timing helper ---- */

#include <mach/mach_time.h>

static double get_time_ms(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t t = mach_absolute_time();
    return (double)(t * info.numer / info.denom) / 1e6;
}

/*
 * ---- init(num_workers, num_layers, K, components, packed_dir, expert_size) ----
 *
 * One-shot initialization:
 *   - Parse component specs
 *   - Open layer FDs
 *   - Allocate staging buffer
 *   - Start worker thread pool
 *   - Cache mx.array function and dtype objects
 */

static PyObject *fml_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"num_workers", "num_layers", "K", "components",
                             "packed_dir", "expert_size", NULL};
    int num_workers = 8;
    int num_layers = 0;
    int K = 0;
    PyObject *py_components = NULL;
    const char *packed_dir = NULL;
    Py_ssize_t expert_size_ss = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiOsn", kwlist,
                                     &num_workers, &num_layers, &K,
                                     &py_components, &packed_dir, &expert_size_ss))
    {
        /* Try simpler parse for backwards compat: init(num_workers=8) */
        PyErr_Clear();
        static char *kwlist2[] = {"num_workers", NULL};
        num_workers = 8;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist2, &num_workers))
            return NULL;
    }

    size_t expert_size = (size_t)expert_size_ss;

    if (g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "fast_moe_load already initialized");
        return NULL;
    }

    if (num_workers < 1 || num_workers > FML_MAX_WORKERS) {
        PyErr_Format(PyExc_ValueError, "num_workers must be 1-%d", FML_MAX_WORKERS);
        return NULL;
    }

    /* Parse component specs if provided */
    if (py_components && PyList_Check(py_components)) {
        Py_ssize_t num_comps = PyList_Size(py_components);
        if (num_comps <= 0 || num_comps > FML_MAX_COMPONENTS) {
            PyErr_Format(PyExc_ValueError, "components count must be 1-%d", FML_MAX_COMPONENTS);
            return NULL;
        }

        g_state.num_comps = (int)num_comps;
        g_state.per_expert_comp_total = 0;

        for (Py_ssize_t ci = 0; ci < num_comps; ci++) {
            PyObject *comp = PyList_GetItem(py_components, ci);
            if (!comp || !PyDict_Check(comp)) {
                PyErr_Format(PyExc_TypeError, "components[%zd] must be a dict", ci);
                return NULL;
            }

            ComponentSpec *cs = &g_state.comp_specs[ci];

            /* name */
            PyObject *py_name = PyDict_GetItemString(comp, "name");
            if (!py_name) {
                PyErr_Format(PyExc_ValueError, "components[%zd] missing 'name'", ci);
                return NULL;
            }
            const char *name = PyUnicode_AsUTF8(py_name);
            if (!name) return NULL;
            strncpy(cs->name, name, FML_COMP_NAME_LEN - 1);
            cs->name[FML_COMP_NAME_LEN - 1] = '\0';

            /* Intern the component name string for fast dict insertion */
            g_state.comp_name_strs[ci] = PyUnicode_InternFromString(cs->name);
            if (!g_state.comp_name_strs[ci]) return NULL;

            /* offset */
            PyObject *py_off = PyDict_GetItemString(comp, "offset");
            if (!py_off) {
                PyErr_Format(PyExc_ValueError, "components[%zd] missing 'offset'", ci);
                return NULL;
            }
            cs->offset = (size_t)PyLong_AsUnsignedLongLong(py_off);

            /* size */
            PyObject *py_sz = PyDict_GetItemString(comp, "size");
            if (!py_sz) {
                PyErr_Format(PyExc_ValueError, "components[%zd] missing 'size'", ci);
                return NULL;
            }
            cs->size = (size_t)PyLong_AsUnsignedLongLong(py_sz);
            g_state.per_expert_comp_total += cs->size;

            /* shape */
            PyObject *py_shape = PyDict_GetItemString(comp, "shape");
            if (!py_shape || !PyList_Check(py_shape)) {
                PyErr_Format(PyExc_ValueError, "components[%zd] missing or invalid 'shape'", ci);
                return NULL;
            }
            cs->ndim = (int)PyList_Size(py_shape);
            if (cs->ndim <= 0 || cs->ndim > 4) {
                PyErr_Format(PyExc_ValueError, "components[%zd] shape dims must be 1-4", ci);
                return NULL;
            }
            for (int d = 0; d < cs->ndim; d++) {
                cs->shape[d] = (int)PyLong_AsLong(PyList_GetItem(py_shape, d));
            }

            /* dtype -> numpy dtype */
            PyObject *py_dtype = PyDict_GetItemString(comp, "dtype");
            if (!py_dtype) {
                PyErr_Format(PyExc_ValueError, "components[%zd] missing 'dtype'", ci);
                return NULL;
            }
            const char *dtype_str = PyUnicode_AsUTF8(py_dtype);
            if (!dtype_str) return NULL;

            if (strcmp(dtype_str, "uint32") == 0)       cs->npy_dtype = NPY_UINT32;
            else if (strcmp(dtype_str, "uint16") == 0)   cs->npy_dtype = NPY_UINT16;
            else if (strcmp(dtype_str, "float16") == 0)  cs->npy_dtype = NPY_FLOAT16;
            else if (strcmp(dtype_str, "float32") == 0)  cs->npy_dtype = NPY_FLOAT32;
            else if (strcmp(dtype_str, "int32") == 0)    cs->npy_dtype = NPY_INT32;
            else {
                PyErr_Format(PyExc_ValueError, "Unknown dtype '%s'", dtype_str);
                return NULL;
            }

            /* needs_bf16_view */
            PyObject *py_bf16 = PyDict_GetItemString(comp, "needs_bf16_view");
            cs->needs_bf16_view = (py_bf16 && PyObject_IsTrue(py_bf16)) ? 1 : 0;
        }
    }

    /* Store parameters */
    g_state.expert_size = expert_size;
    g_state.num_layers = num_layers;
    g_state.K = K;

    /* Open packed layer files */
    if (packed_dir && num_layers > 0) {
        if (num_layers > FML_MAX_LAYERS) {
            PyErr_Format(PyExc_ValueError, "num_layers must be 1-%d", FML_MAX_LAYERS);
            return NULL;
        }
        if (K <= 0 || K > FML_MAX_K) {
            PyErr_Format(PyExc_ValueError, "K must be 1-%d", FML_MAX_K);
            return NULL;
        }

        for (int li = 0; li < num_layers; li++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, li);

            int fd = open(path, O_RDONLY);
            if (fd < 0) {
                PyErr_Format(PyExc_OSError, "Cannot open %s: %s", path, strerror(errno));
                for (int k = 0; k < li; k++) {
                    close(g_state.layer_fds[k].fd);
                    g_state.layer_fds[k].fd = -1;
                }
                return NULL;
            }
            g_state.layer_fds[li].fd = fd;
        }
    }

    /*
     * Allocate staging buffer.
     * Size: num_layers * K * expert_size (enough for full routing of all layers).
     * But we organize it differently: per (layer_entry, component), each region
     * is [K * comp_size] bytes so it maps directly to a [K, *shape] array.
     *
     * Total per layer entry: sum over comps of (K * comp.size)
     *                      = K * sum(comp.size) = K * per_expert_comp_total
     *
     * For 60 layers, K=4, expert_size ~7MB: 60 * 4 * 7MB = 1.68 GB.
     * That's too large. Instead, we allocate for the max routing size,
     * which is num_layers * K * per_expert_comp_total.
     *
     * Actually we need it all: one call loads all 60 layers.
     * 60 * 4 * 7,077,888 = ~1.6 GB. That's large but temporary.
     *
     * Optimization: we can allocate on demand or use a fixed max.
     * For safety, let's cap and grow.
     */
    if (num_layers > 0 && K > 0 && g_state.per_expert_comp_total > 0) {
        size_t needed = (size_t)num_layers * (size_t)K * g_state.per_expert_comp_total;
        /* Round up to page boundary */
        needed = (needed + FML_PAGE_SIZE - 1) & ~((size_t)(FML_PAGE_SIZE - 1));

        void *buf = NULL;
        int rc = posix_memalign(&buf, FML_PAGE_SIZE, needed);
        if (rc != 0 || !buf) {
            PyErr_Format(PyExc_MemoryError,
                         "Failed to allocate %zu byte staging buffer", needed);
            /* cleanup FDs */
            for (int li = 0; li < num_layers; li++) {
                if (g_state.layer_fds[li].fd >= 0) {
                    close(g_state.layer_fds[li].fd);
                    g_state.layer_fds[li].fd = -1;
                }
            }
            return NULL;
        }
        g_state.staging = buf;
        g_state.staging_size = needed;
    }

    /* Cache mx.array function and mx.bfloat16 dtype */
    PyObject *mx_module = PyImport_ImportModule("mlx.core");
    if (!mx_module) {
        /* Cleanup */
        if (g_state.staging) { free(g_state.staging); g_state.staging = NULL; }
        for (int li = 0; li < num_layers; li++) {
            if (g_state.layer_fds[li].fd >= 0) {
                close(g_state.layer_fds[li].fd);
                g_state.layer_fds[li].fd = -1;
            }
        }
        return NULL;
    }

    g_state.mx_array_fn = PyObject_GetAttrString(mx_module, "array");
    g_state.mx_eval_fn = PyObject_GetAttrString(mx_module, "eval");
    g_state.mx_bfloat16 = PyObject_GetAttrString(mx_module, "bfloat16");
    Py_DECREF(mx_module);

    if (!g_state.mx_array_fn || !g_state.mx_eval_fn || !g_state.mx_bfloat16) {
        Py_XDECREF(g_state.mx_array_fn);
        Py_XDECREF(g_state.mx_eval_fn);
        Py_XDECREF(g_state.mx_bfloat16);
        g_state.mx_array_fn = NULL;
        g_state.mx_eval_fn = NULL;
        g_state.mx_bfloat16 = NULL;
        if (g_state.staging) { free(g_state.staging); g_state.staging = NULL; }
        for (int li = 0; li < num_layers; li++) {
            if (g_state.layer_fds[li].fd >= 0) {
                close(g_state.layer_fds[li].fd);
                g_state.layer_fds[li].fd = -1;
            }
        }
        PyErr_SetString(PyExc_RuntimeError, "Cannot find mx.array, mx.eval, or mx.bfloat16");
        return NULL;
    }

    /* Create worker thread pool */
    pthread_mutex_init(&g_state.done_mutex, NULL);
    pthread_cond_init(&g_state.done_cond, NULL);
    atomic_store(&g_state.completed_count, 0);

    g_state.num_workers = num_workers;
    g_state.workers = (WorkerCtx *)calloc(num_workers, sizeof(WorkerCtx));
    if (!g_state.workers) {
        Py_XDECREF(g_state.mx_array_fn); g_state.mx_array_fn = NULL;
        Py_XDECREF(g_state.mx_bfloat16); g_state.mx_bfloat16 = NULL;
        if (g_state.staging) { free(g_state.staging); g_state.staging = NULL; }
        for (int li = 0; li < num_layers; li++) {
            if (g_state.layer_fds[li].fd >= 0) {
                close(g_state.layer_fds[li].fd);
                g_state.layer_fds[li].fd = -1;
            }
        }
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < num_workers; i++) {
        WorkerCtx *w = &g_state.workers[i];
        w->worker_id = i;
        w->running = 1;
        w->has_work = 0;
        w->items = NULL;
        w->item_count = 0;
        w->done_mutex = &g_state.done_mutex;
        w->done_cond = &g_state.done_cond;
        w->completed_count = &g_state.completed_count;

        pthread_mutex_init(&w->work_mutex, NULL);
        pthread_cond_init(&w->work_cond, NULL);

        int rc = pthread_create(&w->thread, NULL, worker_func, w);
        if (rc != 0) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to create worker thread %d: %s", i, strerror(rc));
            for (int j = 0; j < i; j++) {
                g_state.workers[j].running = 0;
                pthread_mutex_lock(&g_state.workers[j].work_mutex);
                g_state.workers[j].has_work = 1;
                pthread_cond_signal(&g_state.workers[j].work_cond);
                pthread_mutex_unlock(&g_state.workers[j].work_mutex);
                pthread_join(g_state.workers[j].thread, NULL);
                pthread_mutex_destroy(&g_state.workers[j].work_mutex);
                pthread_cond_destroy(&g_state.workers[j].work_cond);
            }
            free(g_state.workers);
            g_state.workers = NULL;
            Py_XDECREF(g_state.mx_array_fn); g_state.mx_array_fn = NULL;
            Py_XDECREF(g_state.mx_bfloat16); g_state.mx_bfloat16 = NULL;
            if (g_state.staging) { free(g_state.staging); g_state.staging = NULL; }
            for (int li = 0; li < num_layers; li++) {
                if (g_state.layer_fds[li].fd >= 0) {
                    close(g_state.layer_fds[li].fd);
                    g_state.layer_fds[li].fd = -1;
                }
            }
            return NULL;
        }
    }

    g_state.initialized = 1;
    g_state.total_loads = 0;
    g_state.total_bytes_read = 0;
    g_state.total_calls = 0;
    g_state.total_io_ms = 0.0;
    g_state.total_create_ms = 0.0;

    Py_RETURN_NONE;
}

/*
 * ---- load_and_assemble(routing_list) ----
 *
 * THE HOT PATH. Creates FRESH mx.arrays every call.
 *
 * Input: routing_list -- list of (layer_idx, expert_indices_list) tuples
 *   e.g. [(0, [23, 45, 120, 7]), (1, [12, 67, 200, 3]), ...]
 *
 * Steps (ALL in C):
 *   1. Build pread work items targeting the staging buffer
 *      Staging layout: per (entry, comp), each is K * comp_size contiguous bytes.
 *      Expert slot i for component c of entry e is at:
 *        staging_base + e * (num_comps * K * max_comp_stride) + ...
 *      Actually, we use a flat offset table for simplicity.
 *
 *   2. Dispatch to worker threads, release GIL, wait for completion
 *
 *   3. For each (entry, comp): create numpy array pointing at staging data,
 *      convert to mx.array, optionally apply .view(bfloat16).
 *      Build dict per layer, collect into result list.
 *
 * Returns: list of N dicts, each {comp_name: mx.array[K, *shape]}
 *
 * IMPORTANT: Every call returns NEW mx.array objects with FRESH data.
 * No stale buffer issue. The staging buffer is reused between calls
 * (it's just scratch space for pread), but the mx.arrays are always new.
 */

static PyObject *fml_load_and_assemble(PyObject *self, PyObject *args) {
    PyObject *py_routing_list = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_routing_list))
        return NULL;

    if (!g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() first");
        return NULL;
    }

    if (!PyList_Check(py_routing_list) && !PyTuple_Check(py_routing_list)) {
        PyErr_SetString(PyExc_TypeError, "routing_list must be a list or tuple");
        return NULL;
    }

    Py_ssize_t num_entries = PySequence_Size(py_routing_list);
    if (num_entries == 0) {
        return PyList_New(0);
    }

    int K = g_state.K;
    int num_comps = g_state.num_comps;

    /*
     * Staging buffer layout:
     *
     * For each entry e (0..num_entries-1):
     *   For each component c (0..num_comps-1):
     *     Region of K * comp_specs[c].size bytes, contiguous.
     *     Within this region, slot s (0..K-1) is at offset s * comp_specs[c].size.
     *
     * entry_stride = sum over c of (K * comp_specs[c].size)
     *              = K * per_expert_comp_total
     *
     * Offset for (entry e, comp c, slot s):
     *   base = e * entry_stride + comp_region_offset[c] + s * comp_specs[c].size
     *
     * Where comp_region_offset[c] = K * sum(comp_specs[0..c-1].size)
     */
    size_t entry_stride = (size_t)K * g_state.per_expert_comp_total;
    size_t comp_region_offsets[FML_MAX_COMPONENTS];
    {
        size_t off = 0;
        for (int c = 0; c < num_comps; c++) {
            comp_region_offsets[c] = off;
            off += (size_t)K * g_state.comp_specs[c].size;
        }
        /* off should equal entry_stride */
    }

    /* Check staging buffer is large enough */
    size_t total_staging_needed = (size_t)num_entries * entry_stride;
    if (total_staging_needed > g_state.staging_size) {
        /* Need to grow staging buffer */
        size_t new_size = (total_staging_needed + FML_PAGE_SIZE - 1) & ~((size_t)(FML_PAGE_SIZE - 1));
        void *new_buf = NULL;
        int rc = posix_memalign(&new_buf, FML_PAGE_SIZE, new_size);
        if (rc != 0 || !new_buf) {
            PyErr_Format(PyExc_MemoryError,
                         "Cannot grow staging buffer to %zu bytes", new_size);
            return NULL;
        }
        free(g_state.staging);
        g_state.staging = new_buf;
        g_state.staging_size = new_size;
    }

    /*
     * Build work items: one per (entry, expert_slot, component).
     * Total: num_entries * K * num_comps (max).
     */
    int max_items = (int)(num_entries * K * num_comps);
    if (max_items > FML_MAX_WORK_ITEMS) {
        PyErr_Format(PyExc_OverflowError,
                     "Too many work items: %d (max %d)", max_items, FML_MAX_WORK_ITEMS);
        return NULL;
    }

    WorkItem *items = (WorkItem *)calloc(max_items, sizeof(WorkItem));
    if (!items) {
        PyErr_NoMemory();
        return NULL;
    }

    /* Also store per-entry K values (might differ from g_state.K if less experts routed) */
    int *entry_expert_counts = (int *)calloc(num_entries, sizeof(int));
    if (!entry_expert_counts) {
        free(items);
        PyErr_NoMemory();
        return NULL;
    }

    int item_idx = 0;

    for (Py_ssize_t ei = 0; ei < num_entries; ei++) {
        PyObject *entry = PySequence_GetItem(py_routing_list, ei);
        if (!entry) { free(items); free(entry_expert_counts); return NULL; }

        /* Unpack (layer_idx, expert_indices_list) */
        PyObject *py_layer_idx = NULL;
        PyObject *py_expert_list = NULL;

        if (PyTuple_Check(entry) && PyTuple_Size(entry) == 2) {
            py_layer_idx = PyTuple_GetItem(entry, 0);
            py_expert_list = PyTuple_GetItem(entry, 1);
        } else if (PyList_Check(entry) && PyList_Size(entry) == 2) {
            py_layer_idx = PyList_GetItem(entry, 0);
            py_expert_list = PyList_GetItem(entry, 1);
        } else {
            Py_DECREF(entry);
            free(items);
            free(entry_expert_counts);
            PyErr_SetString(PyExc_TypeError,
                            "Each routing entry must be (layer_idx, [expert_indices])");
            return NULL;
        }

        int layer_idx = (int)PyLong_AsLong(py_layer_idx);
        if (layer_idx < 0 || layer_idx >= g_state.num_layers) {
            Py_DECREF(entry);
            free(items);
            free(entry_expert_counts);
            PyErr_Format(PyExc_ValueError, "layer_idx %d out of range (0-%d)",
                         layer_idx, g_state.num_layers - 1);
            return NULL;
        }

        if (!PyList_Check(py_expert_list) && !PyTuple_Check(py_expert_list)) {
            Py_DECREF(entry);
            free(items);
            free(entry_expert_counts);
            PyErr_SetString(PyExc_TypeError, "expert_indices must be a list or tuple");
            return NULL;
        }

        Py_ssize_t num_experts_this = PySequence_Size(py_expert_list);
        if (num_experts_this > K) {
            Py_DECREF(entry);
            free(items);
            free(entry_expert_counts);
            PyErr_Format(PyExc_ValueError,
                         "Too many experts %zd for layer %d (K=%d)",
                         num_experts_this, layer_idx, K);
            return NULL;
        }

        entry_expert_counts[ei] = (int)num_experts_this;
        int fd = g_state.layer_fds[layer_idx].fd;

        for (Py_ssize_t si = 0; si < num_experts_this; si++) {
            PyObject *py_eidx = PySequence_GetItem(py_expert_list, si);
            int expert_idx = (int)PyLong_AsLong(py_eidx);
            Py_DECREF(py_eidx);

            off_t expert_base = (off_t)expert_idx * (off_t)g_state.expert_size;

            for (int ci = 0; ci < num_comps; ci++) {
                ComponentSpec *cs = &g_state.comp_specs[ci];

                WorkItem *wi = &items[item_idx++];
                wi->fd = fd;
                /* Destination in staging buffer:
                 * entry offset + component region offset + slot offset */
                wi->dest = (char *)g_state.staging
                         + (size_t)ei * entry_stride
                         + comp_region_offsets[ci]
                         + (size_t)si * cs->size;
                wi->nbytes = cs->size;
                wi->file_offset = expert_base + (off_t)cs->offset;
                wi->error = 0;
                wi->bytes_read = 0;
            }
        }

        Py_DECREF(entry);
    }

    if (item_idx == 0) {
        free(items);
        free(entry_expert_counts);
        return PyList_New(0);
    }

    /* ---- Phase 1: Parallel pread into staging buffer (GIL released) ---- */

    double t_io_start = get_time_ms();

    /* Distribute work items to workers (round-robin) */
    WorkItem **worker_items = (WorkItem **)calloc(g_state.num_workers, sizeof(WorkItem *));
    int *worker_counts = (int *)calloc(g_state.num_workers, sizeof(int));
    if (!worker_items || !worker_counts) {
        free(items);
        free(entry_expert_counts);
        free(worker_items);
        free(worker_counts);
        PyErr_NoMemory();
        return NULL;
    }

    /* Count items per worker */
    int *worker_alloc = (int *)calloc(g_state.num_workers, sizeof(int));
    for (int i = 0; i < item_idx; i++) {
        worker_alloc[i % g_state.num_workers]++;
    }
    for (int w = 0; w < g_state.num_workers; w++) {
        if (worker_alloc[w] > 0) {
            worker_items[w] = (WorkItem *)calloc(worker_alloc[w], sizeof(WorkItem));
            if (!worker_items[w]) {
                for (int k = 0; k < w; k++) free(worker_items[k]);
                free(worker_items);
                free(worker_counts);
                free(worker_alloc);
                free(items);
                free(entry_expert_counts);
                PyErr_NoMemory();
                return NULL;
            }
        }
    }
    free(worker_alloc);

    for (int i = 0; i < item_idx; i++) {
        int wid = i % g_state.num_workers;
        worker_items[wid][worker_counts[wid]++] = items[i];
    }

    /* Reset completion counter */
    atomic_store(&g_state.completed_count, 0);

    /* Dispatch to workers */
    for (int w = 0; w < g_state.num_workers; w++) {
        WorkerCtx *wk = &g_state.workers[w];
        pthread_mutex_lock(&wk->work_mutex);
        wk->items = worker_items[w];
        wk->item_count = worker_counts[w];
        wk->has_work = (worker_counts[w] > 0) ? 1 : 0;
        if (wk->has_work)
            pthread_cond_signal(&wk->work_cond);
        pthread_mutex_unlock(&wk->work_mutex);
    }

    /* Release GIL and wait for all workers to complete */
    int total_expected = item_idx;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&g_state.done_mutex);
    while (atomic_load(&g_state.completed_count) < total_expected) {
        pthread_cond_wait(&g_state.done_cond, &g_state.done_mutex);
    }
    pthread_mutex_unlock(&g_state.done_mutex);
    Py_END_ALLOW_THREADS

    /* Check for errors */
    int errors = 0;
    long long bytes_loaded = 0;
    for (int w = 0; w < g_state.num_workers; w++) {
        for (int j = 0; j < worker_counts[w]; j++) {
            if (worker_items[w][j].error != 0) {
                errors++;
            } else {
                bytes_loaded += worker_items[w][j].bytes_read;
            }
        }
    }

    /* Cleanup per-worker arrays */
    for (int w = 0; w < g_state.num_workers; w++) {
        free(worker_items[w]);
    }
    free(worker_items);
    free(worker_counts);
    free(items);

    double t_io_end = get_time_ms();

    if (errors > 0) {
        free(entry_expert_counts);
        PyErr_Format(PyExc_IOError,
                     "pread failed for %d/%d work items", errors, item_idx);
        return NULL;
    }

    /* ---- Phase 2: Create FRESH mx.arrays from staging data ---- */

    double t_create_start = get_time_ms();

    PyObject *result_list = PyList_New(num_entries);
    if (!result_list) {
        free(entry_expert_counts);
        return NULL;
    }

    /*
     * Collect all mx.arrays for batch eval at the end.
     * Without eval, MLX accumulates a lazy computation graph that becomes
     * expensive to process (~470ms for 540 arrays). Batch eval keeps it ~30ms.
     */
    int total_arrays = (int)num_entries * num_comps;
    PyObject *eval_list = PyTuple_New(total_arrays);
    if (!eval_list) {
        Py_DECREF(result_list);
        free(entry_expert_counts);
        return NULL;
    }
    int eval_idx = 0;

    int create_ok = 1;

    for (Py_ssize_t ei = 0; ei < num_entries && create_ok; ei++) {
        PyObject *layer_dict = PyDict_New();
        if (!layer_dict) { create_ok = 0; break; }

        int K_this = entry_expert_counts[ei];

        for (int ci = 0; ci < num_comps && create_ok; ci++) {
            ComponentSpec *cs = &g_state.comp_specs[ci];

            /* Staging data for this (entry, component): K_this * cs->size bytes */
            char *staging_ptr = (char *)g_state.staging
                              + (size_t)ei * entry_stride
                              + comp_region_offsets[ci];

            /* Build numpy shape: [K_this, shape[0], shape[1], ...] */
            int stacked_ndim = 1 + cs->ndim;
            npy_intp np_dims[5]; /* max 1 + 4 = 5 */
            np_dims[0] = K_this;
            for (int d = 0; d < cs->ndim; d++) {
                np_dims[1 + d] = cs->shape[d];
            }

            /*
             * Create numpy array and copy staging data into it.
             * We use PyArray_SimpleNew (allocates fresh buffer) + memcpy.
             * This ensures numpy owns the memory and mx.array can safely
             * take ownership through its normal path.
             */
            PyObject *np_arr = PyArray_SimpleNew(stacked_ndim, np_dims, cs->npy_dtype);
            if (!np_arr) {
                Py_DECREF(layer_dict);
                create_ok = 0;
                break;
            }

            /* Copy staging data into numpy array */
            void *np_data = PyArray_DATA((PyArrayObject *)np_arr);
            size_t copy_size = (size_t)K_this * cs->size;
            memcpy(np_data, staging_ptr, copy_size);

            /* Convert to mx.array: mx.array(np_arr) */
            PyObject *mx_arr = PyObject_CallFunctionObjArgs(
                g_state.mx_array_fn, np_arr, NULL);
            Py_DECREF(np_arr);

            if (!mx_arr) {
                Py_DECREF(layer_dict);
                create_ok = 0;
                break;
            }

            /* Apply .view(mx.bfloat16) if needed */
            if (cs->needs_bf16_view) {
                PyObject *view_method = PyObject_GetAttrString(mx_arr, "view");
                if (!view_method) {
                    Py_DECREF(mx_arr);
                    Py_DECREF(layer_dict);
                    create_ok = 0;
                    break;
                }
                PyObject *bf16_arr = PyObject_CallFunctionObjArgs(
                    view_method, g_state.mx_bfloat16, NULL);
                Py_DECREF(view_method);
                Py_DECREF(mx_arr);

                if (!bf16_arr) {
                    Py_DECREF(layer_dict);
                    create_ok = 0;
                    break;
                }
                mx_arr = bf16_arr;
            }

            /* Insert into dict using interned string key */
            PyDict_SetItem(layer_dict, g_state.comp_name_strs[ci], mx_arr);

            /* Collect for batch eval (PyTuple_SET_ITEM steals reference) */
            Py_INCREF(mx_arr);
            PyTuple_SET_ITEM(eval_list, eval_idx++, mx_arr);

            Py_DECREF(mx_arr);
        }

        if (create_ok) {
            /* PyList_SET_ITEM steals reference */
            PyList_SET_ITEM(result_list, ei, layer_dict);
        }
    }

    free(entry_expert_counts);

    if (!create_ok) {
        Py_DECREF(eval_list);
        Py_DECREF(result_list);
        return NULL;
    }

    /*
     * Batch eval all mx.arrays to force materialization.
     * Without this, MLX's lazy graph accumulates ~540 pending operations,
     * causing ~470ms of overhead when the arrays are eventually consumed.
     * Batch eval costs ~30ms and prevents this.
     */
    if (eval_idx > 0) {
        /* Resize tuple to actual count (might be less if some layers had fewer experts) */
        if (eval_idx < total_arrays) {
            _PyTuple_Resize(&eval_list, eval_idx);
        }
        PyObject *eval_result = PyObject_Call(g_state.mx_eval_fn, eval_list, NULL);
        Py_XDECREF(eval_result);
        if (!eval_result) {
            Py_DECREF(eval_list);
            Py_DECREF(result_list);
            return NULL;
        }
    }
    Py_DECREF(eval_list);

    double t_create_end = get_time_ms();

    /* Update stats */
    g_state.total_loads += num_entries;
    g_state.total_bytes_read += bytes_loaded;
    g_state.total_calls++;
    g_state.total_io_ms += (t_io_end - t_io_start);
    g_state.total_create_ms += (t_create_end - t_create_start);

    return result_list;
}

/* ---- stats() ---- */

static PyObject *fml_stats(PyObject *self, PyObject *args) {
    return Py_BuildValue("{s:i, s:i, s:i, s:i, s:n, s:n, s:L, s:L, s:L, s:d, s:d}",
                         "initialized", g_state.initialized,
                         "num_workers", g_state.num_workers,
                         "num_layers", g_state.num_layers,
                         "K", g_state.K,
                         "expert_size", (Py_ssize_t)g_state.expert_size,
                         "staging_size", (Py_ssize_t)g_state.staging_size,
                         "total_loads", g_state.total_loads,
                         "total_bytes_read", g_state.total_bytes_read,
                         "total_calls", g_state.total_calls,
                         "total_io_ms", g_state.total_io_ms,
                         "total_create_ms", g_state.total_create_ms);
}

/* ---- shutdown() ---- */

static PyObject *fml_shutdown(PyObject *self, PyObject *args) {
    if (!g_state.initialized) {
        Py_RETURN_NONE;
    }

    /* Stop worker threads */
    if (g_state.workers) {
        for (int i = 0; i < g_state.num_workers; i++) {
            WorkerCtx *w = &g_state.workers[i];
            pthread_mutex_lock(&w->work_mutex);
            w->running = 0;
            w->has_work = 1;
            pthread_cond_signal(&w->work_cond);
            pthread_mutex_unlock(&w->work_mutex);
        }
        for (int i = 0; i < g_state.num_workers; i++) {
            pthread_join(g_state.workers[i].thread, NULL);
            pthread_mutex_destroy(&g_state.workers[i].work_mutex);
            pthread_cond_destroy(&g_state.workers[i].work_cond);
        }
        free(g_state.workers);
        g_state.workers = NULL;
    }

    /* Free staging buffer */
    if (g_state.staging) {
        free(g_state.staging);
        g_state.staging = NULL;
        g_state.staging_size = 0;
    }

    /* Release cached Python objects */
    Py_XDECREF(g_state.mx_array_fn);
    g_state.mx_array_fn = NULL;
    Py_XDECREF(g_state.mx_eval_fn);
    g_state.mx_eval_fn = NULL;
    Py_XDECREF(g_state.mx_bfloat16);
    g_state.mx_bfloat16 = NULL;

    for (int ci = 0; ci < g_state.num_comps; ci++) {
        Py_XDECREF(g_state.comp_name_strs[ci]);
        g_state.comp_name_strs[ci] = NULL;
    }

    /* Close layer file descriptors */
    for (int i = 0; i < g_state.num_layers; i++) {
        if (g_state.layer_fds[i].fd >= 0) {
            close(g_state.layer_fds[i].fd);
            g_state.layer_fds[i].fd = -1;
        }
    }

    pthread_mutex_destroy(&g_state.done_mutex);
    pthread_cond_destroy(&g_state.done_cond);

    memset(&g_state, 0, sizeof(g_state));

    Py_RETURN_NONE;
}

/* ---- Module definition ---- */

static PyMethodDef fml_methods[] = {
    {"init", (PyCFunction)fml_init, METH_VARARGS | METH_KEYWORDS,
     "init(num_workers=8, num_layers=0, K=0, components=None, packed_dir=None, expert_size=0)\n"
     "Initialize worker pool, open layer files, allocate staging buffer,\n"
     "cache mx.array function. One-shot setup."},
    {"load_and_assemble", fml_load_and_assemble, METH_VARARGS,
     "load_and_assemble(routing_list)\n"
     "THE HOT PATH. pread into C staging (GIL released, parallel), then create\n"
     "FRESH mx.arrays from staging data. No stale buffer issue.\n"
     "routing_list: [(layer_idx, [expert_indices]), ...]\n"
     "Returns: list of dicts {comp_name: mx.array[K, *shape]}"},
    {"stats", fml_stats, METH_NOARGS,
     "stats() -- Return diagnostic counters including I/O and create timings"},
    {"shutdown", fml_shutdown, METH_NOARGS,
     "shutdown() -- Stop workers, free staging buffer, close files"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fml_module = {
    PyModuleDef_HEAD_INIT,
    "fast_moe_load",
    "FRESH mx.array expert weight loading for MoE inference.\n"
    "\n"
    "pread into C staging buffer (parallel pthreads, GIL released),\n"
    "then create FRESH mx.arrays via numpy C API.\n"
    "No pre-allocated buffers, no stale data, no Python in the I/O path.\n"
    "\n"
    "Returns list of dicts ready for compute_moe_direct.",
    -1,
    fml_methods
};

PyMODINIT_FUNC PyInit_fast_moe_load(void) {
    /* Initialize numpy C API */
    import_array();

    PyObject *m = PyModule_Create(&fml_module);
    if (!m) return NULL;

    PyModule_AddIntConstant(m, "MAX_LAYERS", FML_MAX_LAYERS);
    PyModule_AddIntConstant(m, "MAX_K", FML_MAX_K);
    PyModule_AddIntConstant(m, "MAX_COMPONENTS", FML_MAX_COMPONENTS);
    PyModule_AddIntConstant(m, "MAX_WORKERS", FML_MAX_WORKERS);
    PyModule_AddIntConstant(m, "PAGE_SIZE", FML_PAGE_SIZE);

    return m;
}

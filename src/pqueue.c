#include "pqueue.h"
#include <stdlib.h>

PQueue pqueue_create(size_t initial_capacity) {
    PQueue pq;
    if (initial_capacity == 0) {
        initial_capacity = 1;
    }
    pq.entries = malloc(initial_capacity * sizeof(PQEntry));
    pq.size = 0;
    pq.capacity = pq.entries ? initial_capacity : 0;
    return pq;
}

void pqueue_free(PQueue *pq) {
    free(pq->entries);
    pq->entries = NULL;
    pq->size = 0;
    pq->capacity = 0;
}

int pqueue_is_empty(const PQueue *pq) {
    return pq->size == 0;
}

static void swap_entries(PQEntry *a, PQEntry *b) {
    PQEntry tmp = *a;
    *a = *b;
    *b = tmp;
}

static void sift_up(PQueue *pq, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (pq->entries[parent].priority <= pq->entries[i].priority) {
            break;
        }
        swap_entries(&pq->entries[parent], &pq->entries[i]);
        i = parent;
    }
}

static void sift_down(PQueue *pq, size_t i) {
    for (;;) {
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        size_t smallest = i;

        if (left < pq->size && pq->entries[left].priority < pq->entries[smallest].priority) {
            smallest = left;
        }
        if (right < pq->size && pq->entries[right].priority < pq->entries[smallest].priority) {
            smallest = right;
        }
        if (smallest == i) {
            break;
        }
        swap_entries(&pq->entries[i], &pq->entries[smallest]);
        i = smallest;
    }
}

int pqueue_push(PQueue *pq, RPNode *node, float priority) {
    if (pq->size == pq->capacity) {
        size_t new_capacity = (pq->capacity == 0) ? 1 : pq->capacity * 2;
        PQEntry *new_entries = realloc(pq->entries, new_capacity * sizeof(PQEntry));
        if (new_entries == NULL) {
            return 0;
        }
        pq->entries = new_entries;
        pq->capacity = new_capacity;
    }

    pq->entries[pq->size].node = node;
    pq->entries[pq->size].priority = priority;
    sift_up(pq, pq->size);
    pq->size++;
    return 1;
}

int pqueue_pop(PQueue *pq, RPNode **out_node, float *out_priority) {
    if (pq->size == 0) {
        return 0;
    }

    *out_node = pq->entries[0].node;
    *out_priority = pq->entries[0].priority;

    pq->size--;
    pq->entries[0] = pq->entries[pq->size];
    sift_down(pq, 0);

    return 1;
}
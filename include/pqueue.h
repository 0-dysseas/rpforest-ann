#ifndef PQUEUE_H
#define PQUEUE_H

#include "tree.h"
#include <stddef.h>

typedef struct {
    RPNode *node;
    float priority; // smaller = explored sooner
} PQEntry;

typedef struct {
    PQEntry *entries;
    size_t size;
    size_t capacity;
} PQueue;

// Allocates a min-heap with room for initial_capacity entries.
// On allocation failure, the returned PQueue has entries == NULL, capacity == 0.
PQueue pqueue_create(size_t initial_capacity);

void pqueue_free(PQueue *pq);

int pqueue_is_empty(const PQueue *pq);

// Inserts (node, priority), doubling the backing array if full.
// Returns 1 on success, 0 on allocation failure.
int pqueue_push(PQueue *pq, RPNode *node, float priority);

// Removes the entry with the smallest priority into *out_node/*out_priority.
// Returns 1 if an entry was popped, 0 if the queue was empty.
int pqueue_pop(PQueue *pq, RPNode **out_node, float *out_priority);

#endif
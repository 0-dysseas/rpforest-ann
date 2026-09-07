CC      ?= gcc
CFLAGS  ?= -std=c11 -Wall -Wextra -Wpedantic -O2
DBGFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g -O0 -fsanitize=address,undefined
LDFLAGS ?= -lm

SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build
BIN       := $(BUILD_DIR)/rpforest-ann

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

LIB_OBJS := $(filter-out $(BUILD_DIR)/main.o, $(OBJS))

TEST_GENERATOR_SRC := tests/test_generator.c
TEST_GENERATOR_BIN := $(BUILD_DIR)/test_generator

TEST_TREE_SRC := tests/test_tree.c
TEST_TREE_BIN := $(BUILD_DIR)/test_tree

TEST_SEARCH_SRC := tests/test_search.c
TEST_SEARCH_BIN := $(BUILD_DIR)/test_search

.PHONY: all debug clean run test

all: $(BIN)

$(BIN): $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

debug: CFLAGS := $(DBGFLAGS)
debug: clean all

run: all
	./$(BIN)

clean:
	rm -rf $(BUILD_DIR)

test: $(TEST_GENERATOR_BIN) $(TEST_TREE_BIN) $(TEST_SEARCH_BIN)
	./$(TEST_GENERATOR_BIN)
	./$(TEST_TREE_BIN)
	./$(TEST_SEARCH_BIN)

$(TEST_GENERATOR_BIN): $(TEST_GENERATOR_SRC) $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) $(TEST_GENERATOR_SRC) $(LIB_OBJS) -o $@ $(LDFLAGS)

$(TEST_TREE_BIN): $(TEST_TREE_SRC) $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) $(TEST_TREE_SRC) $(LIB_OBJS) -o $@ $(LDFLAGS)

$(TEST_SEARCH_BIN): $(TEST_SEARCH_SRC) $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) $(TEST_SEARCH_SRC) $(LIB_OBJS) -o $@ $(LDFLAGS)
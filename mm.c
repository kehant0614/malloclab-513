/**
 * @file mm.c
 * @brief A 64-bit struct-based implicit free list memory allocator
 *
 * 15-213: Introduction to Computer Systems
 *
 * TODO: insert your documentation here. :)
 *
 *************************************************************************
 *
 * ADVICE FOR STUDENTS.
 * - Step 0: Please read the writeup!
 * - Step 1: Write your heap checker.
 * - Step 2: Write contracts / debugging assert statements.
 * - Good luck, and have fun!
 *
 *************************************************************************
 *
 * @author Taiming Liu <taimingl@andrew.cmu.edu>
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* Do not change the following! */

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/* You can change anything from here onward */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printf(...) ((void)printf(__VA_ARGS__))
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, these should emit no code whatsoever,
 * not even from evaluation of argument expressions.  However,
 * argument expressions should still be syntax-checked and should
 * count as uses of any variables involved.  This used to use a
 * straightforward hack involving sizeof(), but that can sometimes
 * provoke warnings about misuse of sizeof().  I _hope_ that this
 * newer, less straightforward hack will be more robust.  Technically
 * it only works for EXPRs for which (0 && (EXPR)) is valid, but I
 * cannot think of any EXPR usable as a _function parameter_ that
 * doesn't qualify.
 *
 * The "diagnostic push/pop/ignored" pragmas are required to prevent
 * clang from issuing "unused value" warnings about most of the
 * arguments to dbg_printf / dbg_printheap (the argument list is being
 * treated, in this case, as a chain of uses of the comma operator).
 * Yes, these apparently GCC-specific pragmas work with clang,
 * I checked.
 *   -zw 2022-07-15
 */
#define dbg_discard_expr_(expr)                                                \
    (_Pragma("GCC diagnostic push") _Pragma(                                   \
        "GCC diagnostic ignored \"-Wunused-value\"")(void)(0 && (expr))        \
         _Pragma("GCC diagnostic pop"))
#define dbg_requires(expr) dbg_discard_expr_(expr)
#define dbg_assert(expr) dbg_discard_expr_(expr)
#define dbg_ensures(expr) dbg_discard_expr_(expr)
#define dbg_printf(...) dbg_discard_expr_((__VA_ARGS__))
#define dbg_printheap(...) dbg_discard_expr_((__VA_ARGS__))
#endif

/* Basic constants */

typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = 2 * dsize;

/**
 * Size of a chunk of memory to be requested from the system
 * to extend the heap.
 * (Must be divisible by dsize)
 */
static const size_t chunksize = (1 << 12);

/**
 * Status bit in block header.
 * 1 - allocated. 0 - free
 */
static const word_t alloc_mask = 0x1;

/**
 * Bits in block header masking the size of current block.
 */
static const word_t size_mask = ~(word_t)0xF;

/** @brief Represents the pointers to the next/prev free blocks */
typedef struct fblocks {
    struct block *fnext;
    struct block *fprev;
} fblocks_t;

/** @brief Union represents the payload or free block pointers */
union Data {
    fblocks_t fblocks;
    char payload[0];
};

/** @brief Represents the header and payload of one block in the heap */
typedef struct block {
    /** @brief Header contains size + allocation flag */
    word_t header;

    /**
     * @brief A pointer to the block payload.
     *
     * TODO: feel free to delete this comment once you've read it carefully.
     * We don't know what the size of the payload will be, so we will declare
     * it as a zero-length array, which is a GNU compiler extension. This will
     * allow us to obtain a pointer to the start of the payload. (The similar
     * standard-C feature of "flexible array members" won't work here because
     * those are not allowed to be members of a union.)
     *
     * WARNING: A zero-length array must be the last element in a struct, so
     * there should not be any struct fields after it. For this lab, we will
     * allow you to include a zero-length array in a union, as long as the
     * union is the last field in its containing struct. However, this is
     * compiler-specific behavior and should be avoided in general.
     *
     * WARNING: DO NOT cast this pointer to/from other types! Instead, you
     * should use a union to alias this zero-length array with another struct,
     * in order to store additional types of data in the payload memory.
     */
    union Data data;

    /*
     * TODO: delete or replace this comment once you've thought about it.
     * Why can't we declare the block footer here as part of the struct?
     * Why do we even have footers -- will the code work fine without them?
     * which functions actually use the data contained in footers?
     */
} block_t;

/* Global variables */

/** @brief Pointer to first block in the heap */
static block_t *heap_start = NULL;

/** @brief Pointer to last free block on the explicit free list */
static block_t *curr_fblock = NULL;

/** @brief int to count the number of free blocks */
static int fcount = 0;

/*
 *****************************************************************************
 * The functions below are short wrapper functions to perform                *
 * bit manipulation, pointer arithmetic, and other helper operations.        *
 *                                                                           *
 * We've given you the function header comments for the functions below      *
 * to help you understand how this baseline code works.                      *
 *                                                                           *
 * Note that these function header comments are short since the functions    *
 * they are describing are short as well; you will need to provide           *
 * adequate details for the functions that you write yourself!               *
 *****************************************************************************
 */

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, data));
}

/**
 * @brief Given a fblocks pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] fblocks A pointer to a free block's fnext
 * @return The corresponding block
 */
static block_t *fblocks_to_header(fblocks_t *fblocks) {
    return (block_t *)((char *)fblocks - offsetof(block_t, data));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 * @pre The block must be an allocated block.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    // dbg_requires(get_alloc(block));
    // return (void *)&(block->data);
    return (void *)(block->data.payload);
    // return (void *)(block->data.payload);
}

/**
 * @brief Given a free block pointer, returns a pointer to the corresponding
 *        fblocks struct.
 * @param[in] block
 * @return A pointer to the block's fblocks struct.
 * @pre The block must be a valid block, not a boundary tag.
 * @pre The block must be a free block.
 */
static fblocks_t *header_to_fblocks(block_t *block) {
    dbg_requires(get_size(block) != 0);
    dbg_requires(!get_alloc(block));
    // return (fblocks_t *)&(block->data);
    return (fblocks_t *)&(block->data.fblocks);
    // return (fblocks_t *)&(block->data.fblocks);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    size_t size = get_size(block);
    dbg_requires(size != 0 && "Called header_to_footer on the epilogue block");
    // return (word_t *)(block->data.payload + get_size(block) - dsize);
    return (word_t *)((char *)block + size - wsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not a boundary tag.
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block
 * @return The size of the block's payload.
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - dsize;
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[out] block The location to write the epilogue header.
 * @pre block address is not null.
 * @pre block address is exactly one wsize below heap break point.
 */
static void write_epilogue(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == (char *)mem_heap_hi() - 7);
    block->header = pack(0, true);
}

/**
 * @brief Add the block to explicit free list
 *
 * @param[out] block the block to be added
 * @pre block address is not null.
 * @pre a free block
 */
static void add_to_flist(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(!get_alloc(block) &&
                 "Error: Adding an alloc block to free list");

    if (fcount == 1) {
        curr_fblock->data.fblocks.fnext = block;
        curr_fblock->data.fblocks.fprev = block;
        block->data.fblocks.fnext = curr_fblock;
        block->data.fblocks.fprev = curr_fblock;
    }
    if (fcount > 1) {
        block_t *temp = curr_fblock->data.fblocks.fnext;
        curr_fblock->data.fblocks.fnext = block;
        block->data.fblocks.fprev = curr_fblock;
        block->data.fblocks.fnext = temp;
        temp->data.fblocks.fprev = block;
    }
    curr_fblock = block;
    fcount++;
}

/**
 * @brief Remove the block from the explicit free list
 *
 * @param[out] block the block to be removed
 * @pre block address is not null.
 * @pre an alloc block
 */
static void remove_from_flist(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_alloc(block) &&
                 "Error: Removing a free block to free list");

    if (fcount == 0) {
        return;
    }
    if (fcount == 1) {
        dbg_requires(block == curr_fblock);
        curr_fblock = NULL;
    } else {
        block_t *prev = block->data.fblocks.fprev;
        block_t *next = block->data.fblocks.fnext;
        prev->data.fblocks.fnext = next;
        next->data.fblocks.fprev = prev;
        if (curr_fblock == block) {
            curr_fblock = next;
        }
    }
    fcount--;
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes both a header and footer, where the location of the
 * footer is computed in relation to the header.
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 * @pre The block address needs to be within the heap range.
 * @pre The size needs to be at least greater than minimum required size.
 */
static void write_block(block_t *block, size_t size, bool alloc) {
    dbg_requires(block != NULL);
    dbg_requires(size > 0);

    // ?? - double check correct offset from prologue and epilogue
    // Conditions
    dbg_requires((char *)block < (char *)mem_heap_hi() - 7);
    dbg_requires((char *)block + size > (char *)mem_heap_lo() + 7);
    dbg_requires(size >= dsize);

    bool alloc_before = get_alloc(block);
    block->header = pack(size, alloc);
    word_t *footerp = header_to_footer(block);
    *footerp = pack(size, alloc);

    // add/remove the block to/from free block list
    if (!alloc) {
        add_to_flist(block);
    } else {
        if (!alloc_before) {
            remove_from_flist(block);
        }
    }
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Finds the next consecutive block on the explicit free list.
 *
 * This function accesses the next free block on the heap by following
 * the fnext pointer stored on current block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive free block on the heap
 * @pre The block is not the epilogue
 * @pre The block is a free block
 */
static block_t *find_next_fblock(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next_fblock on the last block in the heap");
    dbg_requires(!get_alloc(block) &&
                 "Called find_next_fblock on an allocated block");
    // return (block_t *)block->data.fblocks.fnext;
    return block->data.fblocks.fnext;
}

/**
 * @brief Finds the previous consecutive block on the explicit free list.
 *
 * This function accesses the previous free block on the heap by following
 * the fprev pointer stored on current block.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive free block on the heap
 * @pre The block is not the epilogue
 * @pre The block is a free block
 */
static block_t *find_prev_fblock(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_prev_fblock on the last block in the heap");
    dbg_requires(!get_alloc(block) &&
                 "Called find_prev_fblock on an allocated block");
    // return (block_t *)block->data.fblocks.fprev;
    return block->data.fblocks.fprev;
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    // Compute previous footer position as one word before the header
    return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 *
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 *
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    word_t *footerp = find_prev_footer(block);

    // Return NULL if called on first block in the heap
    if (extract_size(*footerp) == 0) {
        return NULL;
    }

    return footer_to_header(footerp);
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

static void clean_heap() {
    block_t *block = heap_start;
    for (; get_size(block) != 0; block = find_next(block)) {
        block->data.fblocks.fnext = NULL;
        block->data.fblocks.fprev = NULL;
        // block->data.payload = NULL;
        // block->header = NULL;
    }

    heap_start = NULL;
}

static void pheap() {
    if (heap_start != NULL) {
        printf("--- Heap ---\n");
        block_t *block;
        int idx = 0;
        for (block = heap_start; get_size(block) != 0;
             block = find_next(block)) {
            printf("block: %d: %s, size: %zu,   \taddr: %p\n", idx++,
                   get_alloc(block) ? "a" : "f", get_size(block),
                   (void *)block);
        }
        printf("\n");
    }
}

static void pfl() {

    if (fcount > 0) {
        printf("--- Free List ---\n");
        int idx = 0;
        block_t *block = curr_fblock;
        for (; idx < fcount; idx++) {
            printf("block: %d: %s, size: %zu,   \taddr: %p\n", idx,
                       get_alloc(block) ? "a" : "f", get_size(block),
                       (void *)block);
            block = find_next_fblock(block);
        }
        printf("\n\n\n");
    }
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] block
 * @return
 */
static block_t *coalesce_block(block_t *block) {
    /*
     * TODO: delete or replace this comment once you're done.
     *
     * Before you start, it will be helpful to review the "Dynamic Memory
     * Allocation: Basic" lecture, and especially the four coalescing
     * cases that are described.
     *
     * The actual content of the function will probably involve a call to
     * find_prev(), and multiple calls to write_block(). For examples of how
     * to use write_block(), take a look at split_block().
     *
     * Please do not reference code from prior semesters for this, including
     * old versions of the 213 website. We also discourage you from looking
     * at the malloc code in CS:APP and K&R, which make heavy use of macros
     * and which we no longer consider to be good style.
     */
    return block;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] size
 * @return
 */
static block_t *extend_heap(size_t size) {
    void *bp;

    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk((intptr_t)size)) == (void *)-1) {
        return NULL;
    }

    /*
     * TODO: delete or replace this comment once you've thought about it.
     * Think about what bp represents. Why do we write the new block
     * starting one word BEFORE bp, but with the same size that we
     * originally requested?
     *
     * bp represents the position of break before it was increased by size
     * by mem_sbrk.
     * New block starts one word BEFORE bp becaues this one word was the
     * previous epilogue that's reserverd. Since break moved up, we need to
     * retract and use the space.
     */

    // Initialize free block header/footer
    block_t *block = payload_to_header(bp);
    write_block(block, size, false);

    // Create new epilogue header
    block_t *block_next = find_next(block);
    write_epilogue(block_next);

    // Coalesce in case the previous block was free
    block = coalesce_block(block);

    return block;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] block
 * @param[in] asize
 */
static void split_block(block_t *block, size_t asize) {
    dbg_requires(get_alloc(block));

    /**
     * asize needs to be bigger than the minimum size required
     * and smaller than block's current size
     */
    size_t block_size = get_size(block);
    dbg_ensures(asize >= min_block_size &&
                "split_block called without meeting minimum required size");
    // dbg_ensures(asize < block_size &&
    //             "split_block called without enough space");

    if ((block_size - asize) >= min_block_size) {
        block_t *block_next;
        write_block(block, asize, true);

        block_next = find_next(block);
        write_block(block_next, block_size - asize, false);
    }

    dbg_ensures(get_alloc(block));
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] asize
 * @return
 */
static block_t *find_fit(size_t asize) {

    int idx = 0;
    block_t *block = curr_fblock;
    for (; idx < fcount; idx++) {
        if (asize <= get_size(block)) {
            return block;
        }
        block = find_next_fblock(block);
    }

    // if (curr_fblock != NULL) {
    //     if (curr_fblock->data.fblocks.fnext == NULL ||
    //         curr_fblock->data.fblocks.fnext == curr_fblock) {
    //         if (asize <= get_size(curr_fblock)) {
    //             return curr_fblock;
    //         }
    //     } else {
    //         // starting and ending points of scanning loop
    //         block_t *block = find_next_fblock(curr_fblock);
    //         while (block != curr_fblock) {
    //             if (asize <= get_size(block)) {
    //                 return block;
    //             }
    //             block = find_next_fblock(block);
    //         }
    //         if (asize <= get_size(curr_fblock)) {
    //             return curr_fblock;
    //         }
    //     }
    // }

    return NULL; // no fit found
}

/**
 * @brief Helper function to check heap's prologue and epilogue
 *
 * Prologue and epilogue must have size 0 and alloc set to true.
 *
 * @param[in] block pointer of type block_t to prologue/epilogue
 * @return true if block is valid, false otherwise
 */
static bool pro_epilogue_check(block_t *block) {
    if (get_size(block) != 0) {
        fprintf(stderr, "Error: heap prologue/epilogue wrong size.\n");
        return false;
    }
    if (!get_alloc(block)) {
        fprintf(stderr, "Error: heap prologue/epilogue wrong alloc flag.\n");
        return false;
    }

    return true;
}

/**
 * @brief Helper function to check block's address is aligned and within
 * boundary.
 *
 * An address is aligned in this malloc interface when the address is
 * multiple of 16 (dsize).
 *
 * @param[in] block pointer of type block_t to the address to be checked.
 * @return true if address is aligned to dsize, false otherwise
 */
static bool addr_check(block_t *block) {
    uintptr_t block_addr = (uintptr_t)(char *)block;
    // check if within boundary
    if (block_addr < (uintptr_t)mem_heap_lo() ||
        block_addr > ((uintptr_t)mem_heap_hi() - 7)) {
        fprintf(stderr, "Error: Block address is out heap boundaries\n");
        return false;
    }
    uintptr_t data_addr;
    if (get_alloc(block)) {
        data_addr = (uintptr_t)((char *)header_to_payload(block));
    } else {
        data_addr = (uintptr_t)((char *)header_to_fblocks(block));
    }
    // check if aligned
    if ((data_addr % dsize) != 0) {
        fprintf(stderr, "Error: Block address is not aligned\n");
        return false;
    }

    return true;
}

/**
 * @brief Helper function to valid a block.
 *
 * A valid block on the heap needs to meet below criteria:
 * - Implicit list
 *  1. size is greater than minimum size
 *  2. prev/next allocate/free bit consistency
 *  3. header and footer match each other
 * TODO:
 * - Explicit list
 *  4. all prev/next pointers are consistency (A's next is B, A is B's prev)
 *  5. all free blocks are between heap boundaries
 *  6. count free blocks iteratively and match with free block list
 *  7. (seglist) all blocks in each list bucket fall within bucket size range
 *
 * @param[in] block pointer of type block_t to the address to be checked.
 * @return true if address is aligned to dsize, false otherwise
 * @pre The block must not be a boundary tag.
 */
static bool block_ck(block_t *block) {
    // check 1.
    if (get_size(block) < min_block_size) {
        fprintf(stderr, "Error: Block invalid - Not enough size\n");
        return false;
    }

    // check 2.
    /**
     * TODO: Add prev/next check after changing to explicit list
     */

    // check 3.
    word_t *footer = header_to_footer(block);
    if (extract_size(block->header) != extract_size(*footer)) {
        fprintf(stderr,
                "Error: Block invalid - header footer size mismatch.\n");
        return false;
    }
    if (extract_alloc(block->header) != extract_alloc(*footer)) {
        fprintf(stderr,
                "Error: Block invalid - header footer alloc mismatch.\n");
        return false;
    }

    return true;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] line
 * @return
 */
bool mm_checkheap(int line) {
    /*
     * As a filler: one guacamole is equal to 6.02214086 x 10**23 guacas.
     * One might even call it...  the avocado's number.
     *
     * Internal use only: If you mix guacamole on your bibimbap,
     * do you eat it with a pair of chopsticks, or with a spoon?
     *
     * I always use spoon for bibimbap - with or without guacamole
     */

    /**
     * Checking heap with implicit list
     */

    // 0. check if the heap is initialized
    if (!line) {
        fprintf(stderr, "Error: line number not provided\n");
        return false;
    }
    if (!heap_start) {
        fprintf(stderr, "Error: heap is not initialized\n");
        return false;
    }

    // 1. check for prologue and epilogue blocks
    block_t *prologue = (block_t *)((char *)mem_heap_lo());
    block_t *epilogue = (block_t *)((char *)mem_heap_hi() - 7);
    if (!pro_epilogue_check(prologue)) {
        return false;
    }
    if (!pro_epilogue_check(epilogue)) {
        return false;
    }

    block_t *block;
    for (block = heap_start; get_size(block) > 0; block = find_next(block)) {
        // 2. check each block's address alignment
        if (!addr_check(block)) {
            return false;
        }
        // 3. check each block's size, header, and footer
        if (!block_ck(block)) {
            return false;
        }
    }

    return true;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @return
 */
bool mm_init(void) {
    // int n = fcount;
    for (int i = 0; i < fcount; i++) {
        // block_t *block = find_next_fblock(curr_fblock);
        curr_fblock = (block_t *)(mem_sbrk(min_block_size));
        curr_fblock->data.fblocks.fnext = curr_fblock;
        curr_fblock->data.fblocks.fprev = curr_fblock;
        // curr_fblock = block;
    }
    // curr_fblock = (block_t *)(mem_sbrk(min_block_size));
    // curr_fblock->data.fblocks.fnext = curr_fblock;
    // curr_fblock->data.fblocks.fprev = curr_fblock;
    fcount = 0;

    // Create the initial empty heap
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    /*
     * TODO: delete or replace this comment once you've thought about it.
     * Think about why we need a heap prologue and epilogue. Why do
     * they correspond to a block footer and header respectively?
     */

    start[0] = pack(0, true); // Heap prologue (block footer)
    start[1] = pack(0, true); // Heap epilogue (block header)

    // Heap starts with first "block header", currently the epilogue
    heap_start = (block_t *)&(start[1]);

    // Extend the empty heap with a free block of chunksize bytes
    if (extend_heap(chunksize) == NULL) {
        return false;
    }

    return true;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] size
 * @return
 */
void *malloc(size_t size) {
    dbg_requires(mm_checkheap(__LINE__));

    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if (heap_start == NULL) {
        mm_init();
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust block size to include overhead and to meet alignment requirements
    asize = round_up(size + dsize, dsize);

    // Search the free list for a fit
    block = find_fit(asize);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize);
        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    }

    // The block should be marked as free
    dbg_assert(!get_alloc(block));

    // Mark block as allocated
    size_t block_size = get_size(block);
    write_block(block, block_size, true);

    // Try to split the block if too large
    split_block(block, asize);

    bp = header_to_payload(block);

    // DEBUG: print heap and free_list 
    pheap();
    pfl();

    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] bp
 */
void free(void *bp) {
    dbg_requires(mm_checkheap(__LINE__));

    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Mark the block as free
    write_block(block, size, false);

    // Try to coalesce the block with its neighbors
    coalesce_block(block);

    dbg_ensures(mm_checkheap(__LINE__));

    // DEBUG: print heap and free_list 
    pheap();
    pfl();
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] ptr
 * @param[in] size
 * @return
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief
 *
 * <What does this function do?>
 * <What are the function's arguments?>
 * <What is the function's return value?>
 * <Are there any preconditions or postconditions?>
 *
 * @param[in] elements
 * @param[in] size
 * @return
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
 *****************************************************************************
 * Do not delete the following super-secret(tm) lines!                       *
 *                                                                           *
 * 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
 *                                                                           *
 * 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
 * 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
 * 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
 *                                                                           *
 * 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
 * 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
 * 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
 *                                                                           *
 *****************************************************************************
 */

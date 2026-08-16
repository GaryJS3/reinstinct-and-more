#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define RI_MTMD_API __declspec(dllexport)
#else
#  define RI_MTMD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RI_MTMD_ABI_VERSION 2u

typedef struct ri_mtmd_context ri_mtmd_context;
typedef struct ri_mtmd_result ri_mtmd_result;

typedef enum ri_mtmd_chunk_type {
    RI_MTMD_CHUNK_TEXT = 0,
    RI_MTMD_CHUNK_IMAGE = 1,
} ri_mtmd_chunk_type;

typedef struct ri_mtmd_decoder_pos {
    uint32_t t;
    uint32_t x;
    uint32_t y;
    uint32_t z;
} ri_mtmd_decoder_pos;

RI_MTMD_API uint32_t ri_mtmd_abi_version(void);
RI_MTMD_API ri_mtmd_context *ri_mtmd_create(const char *model_path,
                                             const char *mmproj_path,
                                             size_t embedding_dim,
                                             int use_gpu,
                                             int threads,
                                             char *error,
                                             size_t error_size);
RI_MTMD_API void ri_mtmd_destroy(ri_mtmd_context *ctx);
RI_MTMD_API int ri_mtmd_process(ri_mtmd_context *ctx,
                                 const char *formatted_prompt,
                                 const uint8_t *encoded_image,
                                 size_t encoded_image_size,
                                 ri_mtmd_result **out_result,
                                 char *error,
                                 size_t error_size);
RI_MTMD_API void ri_mtmd_result_destroy(ri_mtmd_result *result);

RI_MTMD_API size_t ri_mtmd_result_chunk_count(const ri_mtmd_result *result);
RI_MTMD_API int ri_mtmd_result_chunk_type(const ri_mtmd_result *result, size_t index);
RI_MTMD_API size_t ri_mtmd_result_chunk_n_tokens(const ri_mtmd_result *result, size_t index);
RI_MTMD_API size_t ri_mtmd_result_chunk_n_pos(const ri_mtmd_result *result, size_t index);
RI_MTMD_API const uint32_t *ri_mtmd_result_chunk_tokens(const ri_mtmd_result *result, size_t index);
RI_MTMD_API const float *ri_mtmd_result_chunk_embeddings(const ri_mtmd_result *result, size_t index);
RI_MTMD_API size_t ri_mtmd_result_chunk_embedding_dim(const ri_mtmd_result *result, size_t index);
RI_MTMD_API const ri_mtmd_decoder_pos *ri_mtmd_result_chunk_positions(const ri_mtmd_result *result, size_t index);
RI_MTMD_API int ri_mtmd_result_uses_mrope(const ri_mtmd_result *result);
RI_MTMD_API int ri_mtmd_result_chunk_uses_non_causal(const ri_mtmd_result *result, size_t index);
RI_MTMD_API double ri_mtmd_result_decode_ms(const ri_mtmd_result *result);
RI_MTMD_API double ri_mtmd_result_tokenize_ms(const ri_mtmd_result *result);
RI_MTMD_API double ri_mtmd_result_encode_ms(const ri_mtmd_result *result);
RI_MTMD_API double ri_mtmd_result_copy_ms(const ri_mtmd_result *result);

#ifdef __cplusplus
}
#endif

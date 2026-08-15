#include "reinstinct_mtmd.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

struct Chunk {
    ri_mtmd_chunk_type type = RI_MTMD_CHUNK_TEXT;
    size_t n_tokens = 0;
    size_t n_pos = 0;
    std::vector<uint32_t> tokens;
    std::vector<float> embeddings;
    std::vector<ri_mtmd_decoder_pos> positions;
    size_t embedding_dim = 0;
    bool non_causal = false;
};

void set_error(char *out, size_t out_size, const char *message) {
    if (!out || out_size == 0) return;
    std::strncpy(out, message, out_size - 1);
    out[out_size - 1] = '\0';
}

bool checked_product(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

} // namespace

struct ri_mtmd_context {
    llama_model *model = nullptr;
    mtmd_context *mtmd = nullptr;
    size_t embedding_dim = 0;

    ~ri_mtmd_context() {
        if (mtmd) mtmd_free(mtmd);
        if (model) llama_model_free(model);
    }
};

struct ri_mtmd_result {
    bool uses_mrope = false;
    std::vector<Chunk> chunks;
};

extern "C" uint32_t ri_mtmd_abi_version(void) { return RI_MTMD_ABI_VERSION; }

extern "C" ri_mtmd_context *ri_mtmd_create(const char *model_path,
                                              const char *mmproj_path,
                                              size_t embedding_dim,
                                              int use_gpu,
                                              int threads,
                                              char *error,
                                              size_t error_size) {
    if (!model_path || !mmproj_path || embedding_dim == 0 || threads < 1) {
        set_error(error, error_size, "model path, mmproj path, positive embedding dimension, and positive thread count are required");
        return nullptr;
    }
    try {
        auto result = std::make_unique<ri_mtmd_context>();
        auto model_params = llama_model_default_params();
        model_params.vocab_only = true;
        result->model = llama_model_load_from_file(model_path, model_params);
        if (!result->model) {
            set_error(error, error_size, "llama.cpp could not load the GGUF vocabulary metadata");
            return nullptr;
        }
        // vocab_only leaves llama.cpp's embedding-size getters unset on this
        // revision. ReInstinct parses the authoritative GGUF hidden size
        // while loading its own model, so retain that explicit dimension.
        result->embedding_dim = embedding_dim;
        auto params = mtmd_context_params_default();
        params.use_gpu = use_gpu != 0;
        params.n_threads = threads;
        params.print_timings = false;
        result->mtmd = mtmd_init_from_file(mmproj_path, result->model, params);
        if (!result->mtmd || !mtmd_support_vision(result->mtmd)) {
            set_error(error, error_size, "mmproj does not initialize vision support for this GGUF");
            return nullptr;
        }
        return result.release();
    } catch (const std::bad_alloc &) {
        set_error(error, error_size, "out of memory creating the mtmd context");
    } catch (...) {
        set_error(error, error_size, "unexpected exception creating the mtmd context");
    }
    return nullptr;
}

extern "C" void ri_mtmd_destroy(ri_mtmd_context *ctx) { delete ctx; }

extern "C" int ri_mtmd_process(ri_mtmd_context *ctx,
                                  const char *formatted_prompt,
                                  const uint8_t *encoded_image,
                                  size_t encoded_image_size,
                                  ri_mtmd_result **out_result,
                                  char *error,
                                  size_t error_size) {
    if (!ctx || !formatted_prompt || !encoded_image || encoded_image_size == 0 || !out_result) {
        set_error(error, error_size, "context, prompt, non-empty image bytes, and output pointer are required");
        return -1;
    }
    *out_result = nullptr;
    try {
        const auto wrapper = mtmd_helper_bitmap_init_from_buf(
            ctx->mtmd, encoded_image, encoded_image_size, false);
        if (!wrapper.bitmap) {
            set_error(error, error_size, "libmtmd could not decode the encoded image buffer");
            return -1;
        }
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(wrapper.bitmap, mtmd_bitmap_free);
        std::unique_ptr<mtmd_helper_video, decltype(&mtmd_helper_video_free)> video(wrapper.video_ctx, mtmd_helper_video_free);
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            mtmd_input_chunks_init(), mtmd_input_chunks_free);
        if (!chunks) {
            set_error(error, error_size, "libmtmd could not allocate input chunks");
            return -1;
        }
        const mtmd_input_text text { formatted_prompt, true, true };
        const mtmd_bitmap *bitmaps[] = { bitmap.get() };
        if (mtmd_tokenize(ctx->mtmd, chunks.get(), &text, bitmaps, 1) != 0) {
            set_error(error, error_size, "libmtmd tokenization failed; the prompt must contain exactly one media marker");
            return -1;
        }

        auto result = std::make_unique<ri_mtmd_result>();
        result->uses_mrope = mtmd_decode_use_mrope(ctx->mtmd);
        const size_t count = mtmd_input_chunks_size(chunks.get());
        result->chunks.reserve(count);
        llama_pos logical_pos = 0;
        for (size_t i = 0; i < count; ++i) {
            const mtmd_input_chunk *input = mtmd_input_chunks_get(chunks.get(), i);
            if (!input) { set_error(error, error_size, "libmtmd returned a null input chunk"); return -1; }
            Chunk output;
            output.n_tokens = mtmd_input_chunk_get_n_tokens(input);
            const auto n_pos = mtmd_input_chunk_get_n_pos(input);
            if (n_pos < 0) { set_error(error, error_size, "libmtmd returned a negative logical position count"); return -1; }
            output.n_pos = static_cast<size_t>(n_pos);
            const auto type = mtmd_input_chunk_get_type(input);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                output.type = RI_MTMD_CHUNK_TEXT;
                size_t n_text = 0;
                const llama_token *tokens = mtmd_input_chunk_get_tokens_text(input, &n_text);
                if (n_text != output.n_tokens || (n_text && !tokens)) {
                    set_error(error, error_size, "libmtmd text chunk has inconsistent token storage"); return -1;
                }
                output.tokens.assign(tokens, tokens + n_text);
            } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                output.type = RI_MTMD_CHUNK_IMAGE;
                output.embedding_dim = ctx->embedding_dim;
                output.non_causal = mtmd_decode_use_non_causal(ctx->mtmd, input);
                if (output.non_causal) {
                    set_error(error, error_size, "this projector requests non-causal attention, which ReInstinct does not yet support"); return -1;
                }
                if (mtmd_encode_chunk(ctx->mtmd, input) != 0) {
                    set_error(error, error_size, "libmtmd vision encode/projector failed"); return -1;
                }
                size_t floats = 0;
                if (!checked_product(output.n_tokens, output.embedding_dim, &floats)) {
                    set_error(error, error_size, "image embedding size overflows size_t"); return -1;
                }
                const float *embeddings = mtmd_get_output_embd(ctx->mtmd);
                if (floats && !embeddings) {
                    set_error(error, error_size, "libmtmd returned null image embeddings"); return -1;
                }
                output.embeddings.assign(embeddings, embeddings + floats);
                if (!std::all_of(output.embeddings.begin(), output.embeddings.end(), [](float v) { return std::isfinite(v); })) {
                    set_error(error, error_size, "libmtmd produced a NaN or infinite image embedding"); return -1;
                }
                if (result->uses_mrope) {
                    const mtmd_image_tokens *image_tokens = mtmd_input_chunk_get_tokens_image(input);
                    if (!image_tokens || mtmd_image_tokens_get_n_tokens(image_tokens) != output.n_tokens) {
                        set_error(error, error_size, "libmtmd image token metadata does not match embedding count"); return -1;
                    }
                    output.positions.resize(output.n_tokens);
                    for (size_t p = 0; p < output.n_tokens; ++p) {
                        const auto pos = mtmd_image_tokens_get_decoder_pos(image_tokens, logical_pos, p);
                        output.positions[p] = { pos.t, pos.x, pos.y, pos.z };
                    }
                }
            } else {
                set_error(error, error_size, "audio chunks are unsupported by the ReInstinct bridge"); return -1;
            }
            if (output.n_pos > static_cast<size_t>(std::numeric_limits<llama_pos>::max()) - static_cast<size_t>(logical_pos)) {
                set_error(error, error_size, "logical position count overflows llama_pos"); return -1;
            }
            logical_pos += static_cast<llama_pos>(output.n_pos);
            result->chunks.push_back(std::move(output));
        }
        *out_result = result.release();
        return 0;
    } catch (const std::bad_alloc &) {
        set_error(error, error_size, "out of memory processing the image");
    } catch (...) {
        set_error(error, error_size, "unexpected exception processing the image");
    }
    return -1;
}

extern "C" void ri_mtmd_result_destroy(ri_mtmd_result *result) { delete result; }
extern "C" size_t ri_mtmd_result_chunk_count(const ri_mtmd_result *r) { return r ? r->chunks.size() : 0; }
extern "C" int ri_mtmd_result_chunk_type(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() ? r->chunks[i].type : -1; }
extern "C" size_t ri_mtmd_result_chunk_n_tokens(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() ? r->chunks[i].n_tokens : 0; }
extern "C" size_t ri_mtmd_result_chunk_n_pos(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() ? r->chunks[i].n_pos : 0; }
extern "C" const uint32_t *ri_mtmd_result_chunk_tokens(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() && !r->chunks[i].tokens.empty() ? r->chunks[i].tokens.data() : nullptr; }
extern "C" const float *ri_mtmd_result_chunk_embeddings(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() && !r->chunks[i].embeddings.empty() ? r->chunks[i].embeddings.data() : nullptr; }
extern "C" size_t ri_mtmd_result_chunk_embedding_dim(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() ? r->chunks[i].embedding_dim : 0; }
extern "C" const ri_mtmd_decoder_pos *ri_mtmd_result_chunk_positions(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() && !r->chunks[i].positions.empty() ? r->chunks[i].positions.data() : nullptr; }
extern "C" int ri_mtmd_result_uses_mrope(const ri_mtmd_result *r) { return r && r->uses_mrope; }
extern "C" int ri_mtmd_result_chunk_uses_non_causal(const ri_mtmd_result *r, size_t i) { return r && i < r->chunks.size() && r->chunks[i].non_causal; }

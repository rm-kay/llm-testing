// End-to-end demo of the inference pipeline: char-level tokenize -> forward ->
// greedy argmax -> append -> repeat.
//
// NOTE: the model is randomly initialized (untrained), so the output is
// gibberish. The point is to show how the pieces fit together, not to produce
// meaningful text.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "llm/config.hpp"
#include "llm/model.hpp"

using namespace llm;

// The tiniest possible tokenizer: map printable ASCII [32, 128) to ids [0, 96).
static int   encode_char(char c) { return (int)(unsigned char)c - 32; }
static char  decode_id(int id)   { return (char)(id + 32); }

int main() {
    Config cfg;               // vocab_size=96 matches the printable-ASCII range
    GPT model(cfg);
    model.random_init(1234);

    std::string prompt = "Hello, world! ";
    std::vector<int> tokens;
    for (char c : prompt) {
        const int id = encode_char(c);
        if (id >= 0 && id < cfg.vocab_size) tokens.push_back(id);
    }

    const int max_new_tokens = 32;
    for (int step = 0; step < max_new_tokens; ++step) {
        if ((int)tokens.size() >= cfg.n_ctx) break;

        Tensor logits = model.forward(tokens);  // [T, vocab]

        // Greedy: pick the argmax of the last position's logits.
        const int last = logits.rows() - 1;
        int best = 0;
        float best_v = logits.at(last, 0);
        for (int v = 1; v < cfg.vocab_size; ++v) {
            if (logits.at(last, v) > best_v) { best_v = logits.at(last, v); best = v; }
        }
        tokens.push_back(best);
    }

    std::string out;
    for (int id : tokens) out.push_back(decode_id(id));
    std::printf("%s\n", out.c_str());
    return 0;
}

#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <windows.h>
#ifdef small
#undef small
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "engines/stockfish/stockfish/src/bitboard.h"
#include "engines/stockfish/stockfish/src/misc.h"
#include "engines/stockfish/stockfish/src/movegen.h"
#include "engines/stockfish/stockfish/src/position.h"
#include "engines/stockfish/stockfish/src/syzygy/tbprobe.h"
#include "engines/stockfish/stockfish/src/types.h"

using namespace Stockfish;

namespace {

constexpr const char* kRealStartFen =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

using Clock = std::chrono::steady_clock;

struct HarnessConfig {
    std::string model_path =
      "SSChess_training/models/SSChess_12M[1600-2600]/weights_bf16.pth";
    std::string value_model_path;
    std::string stockfish_path;
    double      normal_time       = 5.0;
    double      panic_time        = 10.0;
    double      stockfish_time    = 0.1;
    int         stockfish_elo     = 2500;
    int         max_plies         = 512;
    double      cpuct             = 0.9;
    double      cpuct_init        = 19652.0;
    double      cpuct_scale       = 1.0;
    double      virtual_loss      = 0.001;
    int         eval_batch_size   = 128;
    double      progress_interval = 0.5;
    int         games             = 3;
    int         seed              = 0;
    bool        verbose_search    = false;
    int         cache_capacity    = 50000;
    int         collect_dup_limit = 128;
    int         min_sims          = 4000;
    int         max_sims          = 50000;
    int         opening_index     = -1;
    int         opening_cache_mb = 0;
    int         opening_cache_max_entries = 0;
    int         opening_cache_max_ply = 8;
    int         opening_cache_full_ply = 1;
    int         opening_cache_branching = 8;
    double      opening_cache_max_seconds = 3600.0;
    int         opening_book_sims = 256;
    int         opening_branch_visit_cap = 0;
    std::string opening_cache_file;
    bool        opening_cache_build_only = false;
    std::string opening_graph_file;
    int         opening_graph_sims = 0;
    int         opening_graph_max_ply = 16;
    double      opening_graph_max_seconds = 0.0;
    int         opening_graph_load_min_visits = 1;
    int         opening_graph_load_max_children = 0;
    int         opening_graph_load_max_nodes = 1000000;
    bool        opening_graph_build_only = false;
    bool        opening_cache_all_white_roots = false;
    bool        dynamic_cpuct      = true;
    bool        uci               = false;
    bool        fp32              = false;
    bool        value_fp32        = false;
    std::vector<int> checkpoint_sims;
    int         checkpoint_topn   = 1;
    int         root_probe_children = 0;
    int         root_probe_visits_per_child = 0;
    int         cpuct_warmup_visits = 0;
    double      cpuct_warmup_value = 0.0;
    bool        syzygy_enable = false;
    std::string syzygy_path;
    int         syzygy_max_pieces = 5;
    int         syzygy_cache_capacity = 8192;
    bool        greedy_first_bot_move = false;
    bool        reuse_tree = true;
};

struct SearchStats {
    int         sims          = 0;
    int         seeded_sims   = 0;
    double      elapsed       = 0.0;
    std::string best_move;
    int         best_visits   = 0;
    float       best_q        = 0.0f;
    double      avg_depth     = 0.0;
    int         max_depth     = 0;
    int         root_children = 0;
    int         leaf_batches  = 0;
    int         cache_hits    = 0;
    int         cache_misses  = 0;
    int         cache_flushes = 0;
    int         deduped       = 0;
    int         cache_size    = 0;
};

struct EncodedPosition {
    std::array<int64_t, 64> pcs{};
    std::array<int64_t, 2>  meta{};
};

struct EvalCacheEntry {
    std::vector<uint16_t> moves;
    std::vector<float>    priors;
    float                 value = 0.0f;
};

struct EvalCacheSlot {
    EvalCacheEntry entry;
    std::uint64_t  stamp = 0;
};

struct EvalCacheTouch {
    Key           key   = 0;
    std::uint64_t stamp = 0;
};

struct TablebaseProbeInfo {
    bool  ok = false;
    int   wdl = 0;
    bool  has_dtz = false;
    int   dtz = 0;
    float value = 0.0f;
};

struct TablebaseCacheSlot {
    TablebaseProbeInfo entry;
    std::uint64_t      stamp = 0;
};

struct TablebaseCacheTouch {
    Key           key = 0;
    std::uint64_t stamp = 0;
};

struct OpeningBookChild {
    uint16_t move_raw    = 0;
    uint32_t visits      = 0;
    float    value_sum   = 0.0f;
    float    prior       = 0.0f;
};

struct OpeningBookEntry {
    float                        value        = 0.0f;
    uint32_t                     total_visits = 0;
    std::vector<OpeningBookChild> children;
};

struct SelectedBatchEval {
    std::vector<std::vector<float>> priors;
    std::vector<float>              values;
};

struct ChildLink {
    uint16_t move_raw   = 0;
    int      node_index = -1;
    float    prior      = 0.0f;
};

struct SearchNode {
    int                    visit_count = 0;
    float                  value_sum   = 0.0f;
    int                    in_flight   = 0;
    bool                   is_expanded = false;
    std::vector<ChildLink> children;

    explicit SearchNode(float p = 0.0f) {
        (void)p;
    }

    float q_value() const {
        return visit_count == 0 ? 0.0f : (value_sum / static_cast<float>(visit_count));
    }
};

struct UniqueMoves {
    std::vector<uint16_t> moves;
    std::vector<int>      tokens;
};

struct LeafGroup {
    Key                           key = 0;
    EncodedPosition               enc;
    std::vector<uint16_t>         moves;
    std::vector<int>              tokens;
    std::vector<std::vector<int>> paths;
};

struct TerminalInfo {
    bool  is_terminal = false;
    float value       = 0.0f;
};

struct OpeningTask {
    std::vector<uint16_t> history;
    int                   ply = 0;
};

struct OpeningBatchItem {
    Key                   key = 0;
    std::vector<uint16_t> history;
    std::vector<uint16_t> moves;
    std::vector<int>      tokens;
    int                   ply = 0;
    EncodedPosition       enc;
};

struct StoredGraphEdge {
    uint16_t move_raw   = 0;
    uint32_t child_id   = 0;
    uint32_t visits     = 0;
    float    value_sum  = 0.0f;
    float    prior      = 0.0f;
};

struct StoredGraphNode {
    Key                         key = 0;
    uint32_t                    visits = 0;
    float                       value_sum = 0.0f;
    std::vector<StoredGraphEdge> children;
};

std::string move_to_uci(const Position& pos, Move move);

class GameState {
  public:
    explicit GameState(std::size_t max_states = 2048) : states_(max_states) {
        reset();
    }

    void reset() {
        pos_.set(kRealStartFen, false, &states_[0]);
        ply_ = 0;
        piece_count_ = count_pieces(pos_);
        piece_count_history_.clear();
        history_raw_.clear();
        history_uci_.clear();
    }

    void push(Move move, bool record_uci = true) {
        ensure_capacity();
        piece_count_history_.push_back(piece_count_);
        if (pos_.capture(move)) {
            --piece_count_;
        }
        if (record_uci) {
            history_uci_.push_back(move_to_uci(pos_, move));
        }
        history_raw_.push_back(move.raw());
        pos_.do_move(move, states_[++ply_]);
    }

    void pop(bool had_uci = true) {
        Move move(history_raw_.back());
        pos_.undo_move(move);
        history_raw_.pop_back();
        if (!piece_count_history_.empty()) {
            piece_count_ = piece_count_history_.back();
            piece_count_history_.pop_back();
        } else {
            piece_count_ = count_pieces(pos_);
        }
        if (had_uci && !history_uci_.empty()) {
            history_uci_.pop_back();
        }
        --ply_;
    }

    void set_from_raw_history(const std::vector<uint16_t>& raw_moves, bool record_uci = false) {
        reset();
        for (uint16_t raw : raw_moves) {
            push(Move(raw), record_uci);
        }
    }

    void set_fen(const std::string& fen) {
        pos_.set(fen, false, &states_[0]);
        ply_ = 0;
        piece_count_ = count_pieces(pos_);
        piece_count_history_.clear();
        history_raw_.clear();
        history_uci_.clear();
    }

    Position& pos() { return pos_; }
    const Position& pos() const { return pos_; }
    int ply() const { return ply_; }
    int piece_count() const { return piece_count_; }
    const std::vector<uint16_t>& history_raw() const { return history_raw_; }
    const std::vector<std::string>& history_uci() const { return history_uci_; }

    StateInfo& state_info(int i) { return states_[i]; }

  private:
    static int count_pieces(const Position& pos) {
        return popcount(pos.pieces());
    }

    void ensure_capacity() {
        if (ply_ + 2 >= static_cast<int>(states_.size())) {
            states_.resize(states_.size() * 2);
        }
    }

    Position               pos_;
    std::deque<StateInfo>  states_;
    std::vector<uint16_t>    history_raw_;
    std::vector<std::string> history_uci_;
    std::vector<int>         piece_count_history_;
    int                      ply_ = 0;
    int                      piece_count_ = 32;
};

std::vector<std::vector<std::string>> openings() {
    return {
      {"e2e4", "e7e5"}, {"d2d4", "d7d5"}, {"c2c4", "e7e5"}, {"g1f3", "d7d5"},
      {"e2e4", "c7c5"}, {"d2d4", "g8f6"}, {"c2c4", "g8f6"}, {"g1f3", "g8f6"},
      {"e2e4", "e7e6"}, {"d2d4", "e7e6"}, {"e2e4", "c7c6"}, {"d2d4", "c7c6"},
      {"e2e3", "d7d5"}, {"d2d3", "d7d5"}, {"c2c3", "d7d5"}, {"g2g3", "d7d5"},
    };
}

double elapsed_seconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string square_to_uci(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string move_to_uci(const Position& pos, Move move) {
    if (move == Move::none()) {
        return "(none)";
    }
    if (move == Move::null()) {
        return "0000";
    }

    Square from = move.from_sq();
    Square to   = move.to_sq();
    if (move.type_of() == CASTLING && !pos.is_chess960()) {
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));
    }

    std::string out = square_to_uci(from) + square_to_uci(to);
    if (move.type_of() == PROMOTION) {
        switch (move.promotion_type()) {
            case KNIGHT: out.push_back('n'); break;
            case BISHOP: out.push_back('b'); break;
            case ROOK: out.push_back('r'); break;
            case QUEEN: out.push_back('q'); break;
            default: break;
        }
    }
    return out;
}

Move parse_uci_move(const Position& pos, const std::string& uci) {
    for (const auto& ext : MoveList<LEGAL>(pos)) {
        Move move = ext;
        if (move_to_uci(pos, move) == uci) {
            return move;
        }
    }
    return Move::none();
}

int piece_to_id(Piece piece, bool flipped) {
    if (piece == NO_PIECE) {
        return 0;
    }
    Color color = color_of(piece);
    if (flipped) {
        color = ~color;
    }
    switch (type_of(piece)) {
        case PAWN: return color == WHITE ? 1 : 7;
        case KNIGHT: return color == WHITE ? 2 : 8;
        case BISHOP: return color == WHITE ? 3 : 9;
        case ROOK: return color == WHITE ? 4 : 10;
        case QUEEN: return color == WHITE ? 5 : 11;
        case KING: return color == WHITE ? 6 : 12;
        default: return 0;
    }
}

void encode_position(const Position& pos, EncodedPosition& out) {
    out = EncodedPosition{};
    bool flipped    = pos.side_to_move() == BLACK;
    int  square_xor = flipped ? 0x38 : 0;

    for (int sq = 0; sq < 64; ++sq) {
        Square mapped = Square(sq ^ square_xor);
        out.pcs[mapped] = piece_to_id(pos.piece_on(Square(sq)), flipped);
    }

    int cast = 0;
    if (!flipped) {
        cast |= pos.can_castle(WHITE_OO) ? 1 : 0;
        cast |= pos.can_castle(WHITE_OOO) ? 2 : 0;
        cast |= pos.can_castle(BLACK_OO) ? 4 : 0;
        cast |= pos.can_castle(BLACK_OOO) ? 8 : 0;
    } else {
        cast |= pos.can_castle(BLACK_OO) ? 1 : 0;
        cast |= pos.can_castle(BLACK_OOO) ? 2 : 0;
        cast |= pos.can_castle(WHITE_OO) ? 4 : 0;
        cast |= pos.can_castle(WHITE_OOO) ? 8 : 0;
    }
    out.meta[0] = cast;

    Square ep = pos.ep_square();
    out.meta[1] = ep == SQ_NONE ? 64 : (static_cast<int>(ep) ^ square_xor);
}

UniqueMoves unique_legal_moves(const Position& pos) {
    UniqueMoves result;
    result.moves.reserve(64);
    result.tokens.reserve(64);

    bool flipped    = pos.side_to_move() == BLACK;
    int  square_xor = flipped ? 0x38 : 0;

    std::array<int, 4096> token_index;
    token_index.fill(-1);

    for (const auto& ext : MoveList<LEGAL>(pos)) {
        Move move  = ext;
        int  from  = static_cast<int>(move.from_sq()) ^ square_xor;
        int  to    = static_cast<int>(move.to_sq()) ^ square_xor;
        int  token = from * 64 + to;
        int  slot  = token_index[token];
        if (slot < 0) {
            token_index[token] = static_cast<int>(result.moves.size());
            result.moves.push_back(move.raw());
            result.tokens.push_back(token);
            continue;
        }
        Move current(result.moves[slot]);
        if (move.type_of() == PROMOTION && move.promotion_type() == QUEEN
            && (current.type_of() != PROMOTION || current.promotion_type() != QUEEN)) {
            result.moves[slot] = move.raw();
        }
    }

    return result;
}

bool bishops_all_same_color(const Position& pos) {
    std::optional<int> color;
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        Piece piece = pos.piece_on(sq);
        if (piece == NO_PIECE || type_of(piece) != BISHOP) {
            continue;
        }
        int sq_color = (static_cast<int>(file_of(sq)) + static_cast<int>(rank_of(sq))) & 1;
        if (!color.has_value()) {
            color = sq_color;
        } else if (*color != sq_color) {
            return false;
        }
    }
    return true;
}

bool insufficient_material(const Position& pos) {
    if (pos.count<PAWN>() || pos.count<ROOK>() || pos.count<QUEEN>()) {
        return false;
    }

    int bishops = pos.count<BISHOP>();
    int knights = pos.count<KNIGHT>();
    int minors  = bishops + knights;
    if (minors == 0 || minors == 1) {
        return true;
    }
    return knights == 0 && bishops > 0 && bishops_all_same_color(pos);
}

bool is_draw_claim(const Position& pos, int search_ply = 0) {
    return pos.is_draw(search_ply) || insufficient_material(pos);
}

TerminalInfo terminal_info(const Position& pos, const UniqueMoves& legal, int search_ply = 0) {
    if (is_draw_claim(pos, search_ply)) {
        return {true, 0.0f};
    }
    if (!legal.moves.empty()) {
        return {false, 0.0f};
    }
    return pos.checkers() ? TerminalInfo{true, -1.0f} : TerminalInfo{true, 0.0f};
}

float tablebase_value_from_wdl(int wdl) {
    if (wdl >= static_cast<int>(Tablebases::WDLWin)) {
        return 1.0f;
    }
    if (wdl <= static_cast<int>(Tablebases::WDLLoss)) {
        return -1.0f;
    }
    return 0.0f;
}

std::string material_code_for_color(const Position& pos, Color color) {
    std::string out = "K";
    auto append = [&](PieceType pt, char ch) {
        int count = 0;
        switch (pt) {
            case QUEEN: count = pos.count<QUEEN>(color); break;
            case ROOK: count = pos.count<ROOK>(color); break;
            case BISHOP: count = pos.count<BISHOP>(color); break;
            case KNIGHT: count = pos.count<KNIGHT>(color); break;
            case PAWN: count = pos.count<PAWN>(color); break;
            default: break;
        }
        out.append(static_cast<std::size_t>(count), ch);
    };
    append(QUEEN, 'Q');
    append(ROOK, 'R');
    append(BISHOP, 'B');
    append(KNIGHT, 'N');
    append(PAWN, 'P');
    return out;
}

std::pair<std::string, std::string> material_signatures(const Position& pos) {
    std::string white = material_code_for_color(pos, WHITE);
    std::string black = material_code_for_color(pos, BLACK);
    return {white + "v" + black, black + "v" + white};
}

float sanitize_value(float value) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    return std::clamp(value, -1.0f, 1.0f);
}

std::vector<float> softmax_selected(const float* logits, const std::vector<int>& tokens) {
    std::vector<float> out(tokens.size(), 0.0f);
    if (tokens.empty()) {
        return out;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int token : tokens) {
        float logit = logits[token];
        if (std::isfinite(logit)) {
            max_logit = std::max(max_logit, logit);
        }
    }
    if (!std::isfinite(max_logit)) {
        float uniform = 1.0f / static_cast<float>(tokens.size());
        std::fill(out.begin(), out.end(), uniform);
        return out;
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        float logit = logits[tokens[i]];
        out[i] = std::isfinite(logit) ? std::exp(logit - max_logit) : 0.0f;
        sum += out[i];
    }

    if (!(sum > 0.0) || !std::isfinite(sum)) {
        float uniform = 1.0f / static_cast<float>(tokens.size());
        std::fill(out.begin(), out.end(), uniform);
        return out;
    }

    float denom = static_cast<float>(sum);
    for (float& value : out) {
        value /= denom;
    }
    return out;
}

std::size_t estimate_eval_entry_bytes(const EvalCacheEntry& entry) {
    return sizeof(Key) + sizeof(float) + sizeof(uint16_t)
      + entry.moves.size() * sizeof(uint16_t) + entry.priors.size() * sizeof(float);
}

std::size_t estimate_opening_book_entry_bytes(const OpeningBookEntry& entry) {
    return sizeof(Key) + sizeof(float) + sizeof(uint16_t)
      + entry.children.size() * (sizeof(uint16_t) + sizeof(float));
}

struct RootBestSnapshot {
    std::string best_move = "-";
    int         best_visits = 0;
    float       best_q = 0.0f;
};

struct RootCandidateSnapshot {
    std::string move = "-";
    int         visits = 0;
    double      visit_share = 0.0;
    float       q = 0.0f;
    float       prior = 0.0f;
};

RootBestSnapshot root_best_snapshot(
  const SearchNode& root,
  const std::vector<SearchNode>& nodes,
  const Position& pos) {
    RootBestSnapshot snapshot;
    if (!root.children.empty()) {
        const ChildLink* best = nullptr;
        for (const auto& child : root.children) {
            if (!best) {
                best = &child;
                continue;
            }
            const SearchNode& lhs = nodes[child.node_index];
            const SearchNode& rhs = nodes[best->node_index];
            if (lhs.visit_count > rhs.visit_count
                || (lhs.visit_count == rhs.visit_count && lhs.q_value() > rhs.q_value())) {
                best = &child;
            }
        }
        if (best) {
            snapshot.best_move   = move_to_uci(pos, Move(best->move_raw));
            snapshot.best_visits = nodes[best->node_index].visit_count;
            snapshot.best_q      = -nodes[best->node_index].q_value();
        }
    }
    return snapshot;
}

std::vector<RootCandidateSnapshot> root_candidate_snapshots(
  const SearchNode& root,
  const std::vector<SearchNode>& nodes,
  const Position& pos,
  int topn) {
    std::vector<RootCandidateSnapshot> snapshots;
    if (topn <= 0 || root.children.empty()) {
        return snapshots;
    }

    std::vector<const ChildLink*> children;
    children.reserve(root.children.size());
    int total_child_visits = 0;
    for (const auto& child : root.children) {
        if (child.node_index < 0 || child.node_index >= static_cast<int>(nodes.size())) {
            continue;
        }
        children.push_back(&child);
        total_child_visits += std::max(0, nodes[child.node_index].visit_count);
    }

    std::sort(children.begin(), children.end(), [&](const ChildLink* lhs, const ChildLink* rhs) {
        const SearchNode& lhs_node = nodes[lhs->node_index];
        const SearchNode& rhs_node = nodes[rhs->node_index];
        if (lhs_node.visit_count != rhs_node.visit_count) {
            return lhs_node.visit_count > rhs_node.visit_count;
        }
        return lhs_node.q_value() > rhs_node.q_value();
    });

    int limit = std::min<int>(topn, static_cast<int>(children.size()));
    snapshots.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        const ChildLink* child = children[i];
        const SearchNode& child_node = nodes[child->node_index];
        RootCandidateSnapshot snapshot;
        snapshot.move = move_to_uci(pos, Move(child->move_raw));
        snapshot.visits = child_node.visit_count;
        snapshot.visit_share =
          total_child_visits > 0 ? static_cast<double>(std::max(0, child_node.visit_count)) / total_child_visits : 0.0;
        snapshot.q = -child_node.q_value();
        snapshot.prior = child->prior;
        snapshots.push_back(snapshot);
    }
    return snapshots;
}

std::string progress_line(
  const SearchNode& root,
  const std::vector<SearchNode>& nodes,
  const Position& pos,
  const Clock::time_point& start,
  int sims,
  double current_limit,
  double avg_depth,
  int max_depth,
  bool verbose,
  std::optional<float> q_override = std::nullopt) {
    double elapsed = elapsed_seconds(start);
    if (elapsed <= 0.0) {
        return "[search] warming up";
    }

    RootBestSnapshot best = root_best_snapshot(root, nodes, pos);
    if (q_override.has_value()) {
        best.best_q = *q_override;
    }

    std::ostringstream ss;
    ss << "[search] " << std::fixed << std::setprecision(2) << elapsed << "/" << current_limit
       << "s sims=" << sims << " nps=" << static_cast<int>(sims / std::max(elapsed, 1e-6))
       << " best=" << best.best_move << " visits=" << best.best_visits
       << " q=" << std::showpos << std::fixed << std::setprecision(3) << best.best_q << std::noshowpos
       << " depth=" << std::fixed << std::setprecision(1) << avg_depth << "/" << max_depth;

    if (verbose && !root.children.empty()) {
        ss << " root_children=" << root.children.size();
    }
    return ss.str();
}

std::string checkpoint_line(
  const SearchNode& root,
  const std::vector<SearchNode>& nodes,
  const Position& pos,
  int target_sims,
  int actual_sims,
  double elapsed,
  double avg_depth,
  int max_depth) {
    RootBestSnapshot best = root_best_snapshot(root, nodes, pos);
    std::ostringstream ss;
    ss << "[checkpoint] target=" << target_sims
       << " sims=" << actual_sims
       << " elapsed=" << std::fixed << std::setprecision(3) << elapsed
       << " best=" << best.best_move
       << " visits=" << best.best_visits
       << " q=" << std::showpos << std::fixed << std::setprecision(3) << best.best_q << std::noshowpos
       << " depth=" << std::fixed << std::setprecision(1) << avg_depth << "/" << max_depth
       << " root_children=" << root.children.size();
    return ss.str();
}

std::vector<std::string> root_candidate_lines(
  const SearchNode& root,
  const std::vector<SearchNode>& nodes,
  const Position& pos,
  int target_sims,
  int actual_sims,
  int topn) {
    std::vector<std::string> lines;
    auto snapshots = root_candidate_snapshots(root, nodes, pos, topn);
    lines.reserve(snapshots.size());
    for (std::size_t i = 0; i < snapshots.size(); ++i) {
        const auto& candidate = snapshots[i];
        std::ostringstream ss;
        ss << "[root] target=" << target_sims
           << " sims=" << actual_sims
           << " rank=" << (i + 1)
           << " move=" << candidate.move
           << " visits=" << candidate.visits
           << " share=" << std::fixed << std::setprecision(4) << candidate.visit_share
           << " q=" << std::showpos << std::fixed << std::setprecision(3) << candidate.q << std::noshowpos
           << " prior=" << std::fixed << std::setprecision(6) << candidate.prior;
        lines.push_back(ss.str());
    }
    return lines;
}

class TorchModel {
  public:
    explicit TorchModel(const std::string& model_path, bool use_fp32, int search_batch_size)
        : use_fp32_(use_fp32), search_batch_size_(std::max(1, search_batch_size)) {
        try {
            if (!torch::cuda::is_available()) {
                 std::cerr << "[warning] CUDA is not reported as available by LibTorch. This will likely crash." << std::endl;
            }
            torch::jit::getProfilingMode() = false;
            torch::jit::getExecutorMode() = false;

            // Load the TorchScript model natively into C++
            module_ = torch::jit::load(model_path);
            
            // Move to CUDA and convert to requested precision
            if (use_fp32) {
                module_.to(torch::kCUDA, torch::kFloat32);
                module_.eval();
                std::cout << "[model] Loaded LibTorch model natively (FP32): " << model_path << std::endl;
            } else {
                module_.to(torch::kCUDA, torch::kBFloat16);
                module_.eval();
                std::cout << "[model] Loaded LibTorch model natively (BF16): " << model_path << std::endl;
            }
            warmup();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading the model\n" + std::string(e.what_without_backtrace()));
        }
        std::cout << "[model] ready" << std::endl;
    }

    std::pair<torch::Tensor, torch::Tensor> evaluate_batch(const std::vector<EncodedPosition>& batch) {
        c10::InferenceMode inference_mode;
        
        int bsz = static_cast<int>(batch.size());
        if (bsz == 0) {
            auto logits = torch::empty({0, 4096}, torch::TensorOptions().dtype(torch::kFloat32));
            auto values = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32));
            return {logits, values};
        }

        int padded_bsz = padded_batch_size(bsz);
        auto [pcs_tensor, meta_tensor] = encode_batch(batch, padded_bsz);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pcs_tensor);
        inputs.push_back(meta_tensor);

        // Run inference entirely on the GPU via C++!
        auto output = module_.forward(inputs).toTuple();

        // Extract outputs, move back to CPU, and ensure they are standard float32 for your MCTS
        torch::Tensor logits =
          output->elements()[0].toTensor().slice(0, 0, bsz).to(torch::kCPU, torch::kFloat32);
        torch::Tensor values =
          output->elements()[1].toTensor().view({padded_bsz}).slice(0, 0, bsz).to(torch::kCPU, torch::kFloat32);

        return {logits, values};
    }

    std::pair<torch::Tensor, torch::Tensor> evaluate_single(const EncodedPosition& encoded) {
        return evaluate_batch(std::vector<EncodedPosition>{encoded});
    }

    SelectedBatchEval evaluate_batch_selected(
      const std::vector<EncodedPosition>& batch,
      const std::vector<const std::vector<int>*>& token_sets) {
        c10::InferenceMode inference_mode;

        int bsz = static_cast<int>(batch.size());
        SelectedBatchEval out;
        out.priors.resize(bsz);
        out.values.resize(bsz, 0.0f);
        if (bsz == 0) {
            return out;
        }
        if (static_cast<int>(token_sets.size()) != bsz) {
            throw std::runtime_error("evaluate_batch_selected token set size mismatch");
        }

        int padded_bsz = padded_batch_size(bsz);
        int max_tokens = 0;
        for (const auto* tokens : token_sets) {
            if (!tokens) {
                throw std::runtime_error("evaluate_batch_selected got null token set");
            }
            max_tokens = std::max(max_tokens, static_cast<int>(tokens->size()));
        }

        auto [pcs_tensor, meta_tensor] = encode_batch(batch, padded_bsz);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pcs_tensor);
        inputs.push_back(meta_tensor);

        auto output = module_.forward(inputs).toTuple();
        torch::Tensor logits_gpu = output->elements()[0].toTensor();
        torch::Tensor values_cpu =
          output->elements()[1].toTensor().to(torch::kCPU, torch::kFloat32).view({padded_bsz});

        auto values_ptr = values_cpu.data_ptr<float>();
        for (int i = 0; i < bsz; ++i) {
            out.values[i] = values_ptr[i];
        }

        if (max_tokens <= 0) {
            return out;
        }

        token_flat_.assign(static_cast<std::size_t>(padded_bsz) * max_tokens, 0);
        token_mask_.assign(static_cast<std::size_t>(padded_bsz) * max_tokens, 0);

        for (int i = 0; i < bsz; ++i) {
            const auto& tokens = *token_sets[i];
            for (std::size_t j = 0; j < tokens.size(); ++j) {
                std::size_t idx = static_cast<std::size_t>(i) * max_tokens + j;
                token_flat_[idx] = tokens[j];
                token_mask_[idx] = 1;
            }
        }

        if (padded_bsz > bsz) {
            const auto& pad_tokens = *token_sets.front();
            for (int i = bsz; i < padded_bsz; ++i) {
                for (std::size_t j = 0; j < pad_tokens.size(); ++j) {
                    std::size_t idx = static_cast<std::size_t>(i) * max_tokens + j;
                    token_flat_[idx] = pad_tokens[j];
                    token_mask_[idx] = 1;
                }
            }
        }

        auto token_tensor =
          torch::from_blob(token_flat_.data(), {padded_bsz, max_tokens}, torch::kInt64).to(torch::kCUDA);
        auto mask_tensor =
          torch::from_blob(token_mask_.data(), {padded_bsz, max_tokens}, torch::TensorOptions().dtype(torch::kUInt8))
            .to(torch::kCUDA)
            .to(torch::kBool);

        auto selected_logits = logits_gpu.gather(1, token_tensor).to(torch::kFloat32);
        selected_logits = selected_logits.masked_fill(mask_tensor.logical_not(), -1.0e30f);

        torch::Tensor probs_cpu = torch::softmax(selected_logits, 1).to(torch::kCPU, torch::kFloat32).contiguous();
        const float* probs_ptr = probs_cpu.data_ptr<float>();

        for (int i = 0; i < bsz; ++i) {
            const auto& tokens = *token_sets[i];
            auto& priors = out.priors[i];
            priors.assign(
              probs_ptr + static_cast<std::size_t>(i) * max_tokens,
              probs_ptr + static_cast<std::size_t>(i) * max_tokens + tokens.size());
        }
        return out;
    }

  private:
    void warmup() {
        std::vector<int> batch_sizes{1};
        if (search_batch_size_ > 1) {
            batch_sizes.push_back(std::max(1, std::min(search_batch_size_, 32)));
            batch_sizes.push_back(search_batch_size_);
        }

        EncodedPosition encoded;
        std::vector<int> tokens;
        tokens.reserve(64);
        for (int token = 0; token < 64; ++token) {
            tokens.push_back(token);
        }

        for (int batch_size : batch_sizes) {
            std::vector<EncodedPosition> batch(static_cast<std::size_t>(batch_size), encoded);
            std::vector<const std::vector<int>*> token_sets(static_cast<std::size_t>(batch_size), &tokens);
            (void)evaluate_batch_selected(batch, token_sets);
        }

        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
    }

    static int next_power_of_two(int value) {
        int out = 1;
        while (out < value) {
            out <<= 1;
        }
        return out;
    }

    int padded_batch_size(int batch_size) const {
        if (batch_size <= 0 || batch_size >= search_batch_size_) {
            return batch_size;
        }

        constexpr int kMinGpuBatch = 32;
        int target = std::max(kMinGpuBatch, next_power_of_two(batch_size));
        return std::min(search_batch_size_, target);
    }

    std::pair<torch::Tensor, torch::Tensor> encode_batch(const std::vector<EncodedPosition>& batch, int padded_bsz) {
        int bsz = static_cast<int>(batch.size());
        pcs_flat_.resize(static_cast<std::size_t>(padded_bsz) * 64);
        meta_flat_.resize(static_cast<std::size_t>(padded_bsz) * 2);

        for (int i = 0; i < bsz; ++i) {
            std::copy(batch[i].pcs.begin(), batch[i].pcs.end(), pcs_flat_.begin() + static_cast<std::size_t>(i) * 64);
            std::copy(batch[i].meta.begin(), batch[i].meta.end(), meta_flat_.begin() + static_cast<std::size_t>(i) * 2);
        }

        if (padded_bsz > bsz) {
            for (int i = bsz; i < padded_bsz; ++i) {
                std::copy(batch[0].pcs.begin(), batch[0].pcs.end(), pcs_flat_.begin() + static_cast<std::size_t>(i) * 64);
                std::copy(batch[0].meta.begin(), batch[0].meta.end(), meta_flat_.begin() + static_cast<std::size_t>(i) * 2);
            }
        }

        auto pcs_tensor = torch::from_blob(pcs_flat_.data(), {padded_bsz, 64}, torch::kInt64).to(torch::kCUDA);
        auto meta_tensor = torch::from_blob(meta_flat_.data(), {padded_bsz, 2}, torch::kInt64).to(torch::kCUDA);
        return {pcs_tensor, meta_tensor};
    }

    torch::jit::script::Module module_;
    bool                       use_fp32_ = false;
    int                        search_batch_size_ = 128;
    std::vector<int64_t>       pcs_flat_;
    std::vector<int64_t>       meta_flat_;
    std::vector<int64_t>       token_flat_;
    std::vector<uint8_t>       token_mask_;
};

class PUCTPlayer {
  public:
    PUCTPlayer(TorchModel& model, HarnessConfig config, TorchModel* value_model = nullptr)
        : model_(model), value_model_(value_model), config_(std::move(config)), root_state_(4096) {
        root_index_ = new_node(1.0f);
        init_tablebases();
        if (!config_.opening_cache_file.empty()
            && std::filesystem::exists(config_.opening_cache_file)
            && !config_.opening_cache_all_white_roots) {
            load_opening_cache(config_.opening_cache_file);
        } else if (config_.opening_cache_mb > 0 || config_.opening_cache_max_entries > 0) {
            build_opening_cache();
            if (!config_.opening_cache_file.empty()) {
                save_opening_cache(config_.opening_cache_file);
            }
        }
        if (!config_.opening_graph_build_only
            && !config_.opening_graph_file.empty()
            && std::filesystem::exists(config_.opening_graph_file)) {
            load_opening_graph_as_tree(config_.opening_graph_file);
        }
    }

    std::size_t opening_cache_size() const { return opening_cache_.size(); }
    std::size_t opening_cache_bytes() const { return opening_cache_bytes_; }

    struct OpeningGraphSummary {
        std::size_t nodes = 0;
        std::size_t edges = 0;
        int         sims = 0;
        int         max_ply = 0;
    };

    // New state variable to track the "Winning confidence" from the last turn
    float persistent_target_q = 0.0f;

    void set_position_from_history(const std::vector<uint16_t>& raw_moves) {
        if (config_.reuse_tree) {
            sync_position_from_history(raw_moves);
        } else {
            reset_tree_to_history(raw_moves, false);
        }
        cache_hits_ = 0;
        cache_misses_ = 0;
        cache_flushes_ = 0;
        tb_hits_ = 0;
        tb_misses_ = 0;
        tb_skipped_by_piece_count_ = 0;
        tb_skipped_by_material_ = 0;
        persistent_target_q = 0.0f;
        root_allowed_moves_.clear();
    }

    void set_position_from_fen(const std::string& fen) {
        root_state_.set_fen(fen);
        nodes_.clear();
        root_index_ = new_node(1.0f);
        cache_hits_ = 0;
        cache_misses_ = 0;
        cache_flushes_ = 0;
        tb_hits_ = 0;
        tb_misses_ = 0;
        tb_skipped_by_piece_count_ = 0;
        tb_skipped_by_material_ = 0;
        persistent_target_q = 0.0f;
        root_allowed_moves_.clear();
    }

    void set_root_allowed_moves(const std::vector<uint16_t>& moves) {
        root_allowed_moves_.clear();
        for (uint16_t move : moves) {
            if (move != 0) {
                root_allowed_moves_.insert(move);
            }
        }
    }

    void clear_root_allowed_moves() {
        root_allowed_moves_.clear();
    }

    void set_root_prior_overrides(const std::vector<std::pair<uint16_t, float>>& priors) {
        root_prior_overrides_.clear();
        for (const auto& item : priors) {
            if (item.first != 0 && item.second > 0.0f && std::isfinite(item.second)) {
                root_prior_overrides_[item.first] = item.second;
            }
        }
    }

    void clear_root_prior_overrides() {
        root_prior_overrides_.clear();
    }

    void advance_root(Move move) {
        root_state_.push(move, true);
        int next_root = -1;
        if (root_index_ >= 0 && root_index_ < static_cast<int>(nodes_.size())) {
            for (const auto& child : nodes_[root_index_].children) {
                if (child.move_raw == move.raw()) {
                    next_root = child.node_index;
                    break;
                }
            }
        }

        if (next_root >= 0 && next_root < static_cast<int>(nodes_.size())) {
            root_index_ = next_root;
            compact_to_reachable_root();
            return;
        }

        nodes_.clear();
        root_index_ = new_node(1.0f);
    }

    std::optional<Move> choose_move() {
        last_stats_ = SearchStats{};
        SearchNode& root = nodes_[root_index_];

        const Clock::time_point start      = Clock::now();
        const Clock::time_point hard_limit = start + std::chrono::duration_cast<Clock::duration>(
                                                      std::chrono::duration<double>(config_.panic_time));

        int    ply = static_cast<int>(root_state_.history_raw().size());
        double scale = 1.0;
        // Time scaling disabled — always use full normal_time
        // if (ply <= 5) {
        //     scale = 0.4;
        // } else if (ply <= 15) {
        //     scale = 0.7;
        // }

        const double base_time_limit = std::max(config_.normal_time * scale, 1.0);
        double       current_time_limit = base_time_limit;
        auto         deadline = start + std::chrono::duration_cast<Clock::duration>(
                                          std::chrono::duration<double>(current_time_limit));

        int start_cache_hits    = cache_hits_;
        int start_cache_misses  = cache_misses_;
        int start_cache_flushes = cache_flushes_;

        std::optional<uint16_t> best_move_tracker;
        bool                    is_panic              = false;
        bool                    is_defensive_boosted  = false;
        bool                    is_sharp_drop_boosted = false;
        bool                    is_instability_boosted = false;
        bool                    flat_tree_extended    = false;
        std::optional<float>    target_q;
        std::optional<float>    initial_q;
        std::optional<float>    peak_best_q;
        int                     seeded_sims = 0;
        int                     last_best_change_sims = 0;
        double                  last_best_change_elapsed = 0.0;

        int       sims         = 0;
        int       leaf_batches = 0;
        int       deduped      = 0;
        long long depth_sum    = 0;
        int       max_depth    = 0;
        std::unordered_set<uint16_t> original_root_allowed_moves = root_allowed_moves_;
        std::unordered_map<uint16_t, int> root_tb_wdl_by_move;
        bool tb_root_filter_active = false;
        int effective_min_sims = config_.min_sims;
        int effective_max_sims = config_.max_sims;
        
        // 1. Capture the piece count at the root!
        int initial_root_pieces = root_state_.piece_count();

        if (auto mate_move = choose_immediate_mate_root(start)) {
            return mate_move;
        }

        // DELETED: We completely removed choose_tablebase_root_move() 
        // so the engine is FORCED to use MCTS in the endgame.

        if (auto tb_filter = build_tablebase_root_filter()) {
            tb_root_filter_active = true;
            root_tb_wdl_by_move = std::move(tb_filter->wdl_by_move);
            if (!tb_filter->allowed_moves.empty()) {
                if (root_allowed_moves_.empty()) {
                    root_allowed_moves_ = tb_filter->allowed_moves;
                } else {
                    for (auto it = root_allowed_moves_.begin(); it != root_allowed_moves_.end();) {
                        if (tb_filter->allowed_moves.find(*it) == tb_filter->allowed_moves.end()) {
                            it = root_allowed_moves_.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
            
            // 2. We removed the 256/512 override. The engine will now use 
            // full config_.min_sims (e.g. 4000) for deep endgame search!
            
            std::cout << "[syzygy] root filter category=" << tb_filter->category
                      << " allowed=" << tb_filter->allowed_moves.size()
                      << " candidates=" << tb_filter->candidate_count
                      << " min_sims=" << effective_min_sims
                      << " max_sims=" << effective_max_sims
                      << " rule50=" << root_state_.pos().rule50_count()
                      << std::endl;
            root = SearchNode(1.0f);
        }

        auto next_progress = start + std::chrono::duration_cast<Clock::duration>(
                                       std::chrono::duration<double>(config_.progress_interval));

        if (!root.is_expanded) {
            UniqueMoves  legal = unique_legal_moves(root_state_.pos());
            TerminalInfo term  = terminal_info(root_state_.pos(), legal);
            if (term.is_terminal) {
                return std::nullopt;
            }

            Key key = root_state_.pos().key();
            if (const OpeningBookEntry* opening_entry = lookup_opening_entry(key)) {
                seed_node_from_opening_entry(root_index_, *opening_entry, legal);
                ++cache_hits_;
                initial_q = nodes_[root_index_].q_value();
            } else {
                EvalCacheEntry entry;
                const EvalCacheEntry* cached = lookup_eval(key);
                if (!cached) {
                    EncodedPosition encoded;
                    encode_position(root_state_.pos(), encoded);
                    std::vector<EncodedPosition> batch{encoded};
                    std::vector<const std::vector<int>*> token_sets{&legal.tokens};
                    auto selected = evaluate_batch_selected_for_search(batch, token_sets);
                    entry.moves  = legal.moves;
                    entry.priors = std::move(selected.priors[0]);
                    entry.value  = sanitize_value(selected.values[0]);
                    remember_eval(key, entry);
                    ++cache_misses_;
                } else {
                    entry = *cached;
                    ++cache_hits_;
                }
                expand_from_entry(root_index_, entry);
                nodes_[root_index_].value_sum += entry.value;
                nodes_[root_index_].visit_count += 1;
                initial_q = nodes_[root_index_].q_value();
            }
        } else if (nodes_[root_index_].visit_count > 0) {
            seeded_sims = nodes_[root_index_].visit_count;
            initial_q = nodes_[root_index_].q_value();
        }



        auto keep_searching = [&](int current_sims) {
            auto now = Clock::now();
            if (effective_max_sims > 0 && (current_sims + seeded_sims) >= effective_max_sims) {
                return false;
            }
            if (now < deadline) {
                return true;
            }
            // Respect min_sims only up to hard_limit to prevent "hanging" on slow models
            if (effective_min_sims > 0 && (current_sims + seeded_sims) < effective_min_sims && now < hard_limit) {
                return true;
            }
            return false;
        };
        auto has_sim_budget = [&]() {
            return effective_max_sims <= 0 || (sims + seeded_sims) < effective_max_sims;
        };

        const Key root_key = root_state_.pos().key();
        const int root_ply = root_state_.ply();

        std::vector<int> checkpoint_targets = config_.checkpoint_sims;
        std::sort(checkpoint_targets.begin(), checkpoint_targets.end());
        checkpoint_targets.erase(
          std::unique(checkpoint_targets.begin(), checkpoint_targets.end()),
          checkpoint_targets.end());
        checkpoint_targets.erase(
          std::remove_if(checkpoint_targets.begin(), checkpoint_targets.end(), [](int value) { return value <= 0; }),
          checkpoint_targets.end());
        std::size_t next_checkpoint_index = 0;
        auto emit_due_checkpoints = [&]() {
            const int total_sims = sims + seeded_sims;
            while (
              next_checkpoint_index < checkpoint_targets.size()
              && total_sims >= checkpoint_targets[next_checkpoint_index]) {
                std::cout << checkpoint_line(
                               nodes_[root_index_],
                               nodes_,
                               root_state_.pos(),
                               checkpoint_targets[next_checkpoint_index],
                               total_sims,
                               elapsed_seconds(start),
                               sims > 0 ? static_cast<double>(depth_sum) / sims : 0.0,
                               max_depth)
                          << std::endl;
                for (const auto& line : root_candidate_lines(
                       nodes_[root_index_],
                       nodes_,
                       root_state_.pos(),
                       checkpoint_targets[next_checkpoint_index],
                       total_sims,
                       config_.checkpoint_topn)) {
                    std::cout << line << std::endl;
                }
                ++next_checkpoint_index;
            }
        };
        emit_due_checkpoints();

        // Warm up/Pre-allocate nodes vector
        if (nodes_.capacity() < 1000000) {
            nodes_.reserve(1000000);
        }
        auto rewind_search = [&](int& pushed, const char* stage) {
            while (pushed > 0) {
                root_state_.pop(false);
                --pushed;
            }
            if (root_state_.ply() != root_ply || root_state_.pos().key() != root_key
                || !root_state_.pos().pos_is_ok()) {
                std::ostringstream ss;
                ss << "search state corrupted after " << stage << " root_ply=" << root_ply
                   << " current_ply=" << root_state_.ply();
                throw std::runtime_error(ss.str());
            }
        };

        while (keep_searching(sims)) {
            std::vector<LeafGroup>       leaves;
            std::unordered_map<Key, int> leaf_group_index;
            int                          duplicate_streak = 0;
            int                          collect_attempts = 0;
            const int                    min_useful_batch =
              std::max(1, std::min(config_.eval_batch_size / 4, 16));
            const int                    max_collect_attempts =
              std::max(config_.eval_batch_size * std::max(1, config_.collect_dup_limit), config_.eval_batch_size * 4);

            while (
              static_cast<int>(leaves.size()) < config_.eval_batch_size
              && keep_searching(sims)
              && collect_attempts < max_collect_attempts) {
                ++collect_attempts;
                int              node_index = root_index_;
                std::vector<int> path{node_index};
                int              pushed = 0;

                while (nodes_[node_index].is_expanded && !nodes_[node_index].children.empty()) {
                    auto [move_raw, child_index] =
                      select_child(
                        node_index,
                        node_index == root_index_ ? config_.opening_branch_visit_cap : 0,
                        node_index == root_index_ ? root_probe_target_for_child_count(nodes_[node_index].children.size()) : 0);
                    root_state_.push(Move(move_raw), false);
                    ++pushed;
                    node_index = child_index;
                    nodes_[node_index].in_flight += 1;
                    path.push_back(node_index);
                }

                int depth = static_cast<int>(path.size()) - 1;
                depth_sum += depth;
                max_depth = std::max(max_depth, depth);

                UniqueMoves  legal = unique_legal_moves(root_state_.pos());
                TerminalInfo term  = terminal_info(root_state_.pos(), legal, depth);
                if (term.is_terminal) {
                    if (!has_sim_budget()) {
                        rewind_search(pushed, "terminal-budget");
                        break;
                    }
                    backup(path, term.value);
                    ++sims;
                    rewind_search(pushed, "terminal");
                    continue;
                }

                
                if (initial_root_pieces > config_.syzygy_max_pieces) {
                    if (auto tb = probe_tablebase(root_state_.pos(), root_state_.piece_count(), false)) {
                        if (!has_sim_budget()) {
                            rewind_search(pushed, "tablebase-budget");
                            break;
                        }
                        backup(path, tb->value);
                        ++sims;
                        rewind_search(pushed, "tablebase-hit");
                        continue;
                    }
                }

                Key key = root_state_.pos().key();
                if (const OpeningBookEntry* opening_entry = lookup_opening_entry(key)) {
                    if (!has_sim_budget()) {
                        rewind_search(pushed, "opening-book-budget");
                        break;
                    }
                    ++cache_hits_;
                    seed_node_from_opening_entry(node_index, *opening_entry, legal);
                    backup(path, opening_entry->value);
                    ++sims;
                    rewind_search(pushed, "opening-book-hit");
                    continue;
                }

                const EvalCacheEntry* cached = lookup_eval(key);
                if (cached != nullptr) {
                    if (!has_sim_budget()) {
                        rewind_search(pushed, "cache-budget");
                        break;
                    }
                    ++cache_hits_;
                    expand_from_entry(node_index, *cached);
                    backup(path, cached->value);
                    ++sims;
                    rewind_search(pushed, "cache-hit");
                    continue;
                }

                ++cache_misses_;
                auto group_it = leaf_group_index.find(key);
                if (group_it != leaf_group_index.end()) {
                    leaves[group_it->second].paths.push_back(std::move(path));
                    ++deduped;
                    ++duplicate_streak;
                    rewind_search(pushed, "dedup");
                    if (
                      static_cast<int>(leaves.size()) >= min_useful_batch
                      && duplicate_streak >= config_.collect_dup_limit) {
                        break;
                    }
                    continue;
                }

                EncodedPosition encoded;
                encode_position(root_state_.pos(), encoded);

                LeafGroup leaf;
                leaf.key    = key;
                leaf.enc    = encoded;
                leaf.moves  = std::move(legal.moves);
                leaf.tokens = std::move(legal.tokens);
                leaf.paths.push_back(std::move(path));
                leaf_group_index[key] = static_cast<int>(leaves.size());
                leaves.push_back(std::move(leaf));
                duplicate_streak = 0;

                rewind_search(pushed, "leaf-collect");
            }

            if (!leaves.empty()) {
                std::vector<EncodedPosition> batch;
                std::vector<const std::vector<int>*> token_sets;
                batch.reserve(leaves.size());
                token_sets.reserve(leaves.size());
                for (const auto& leaf : leaves) {
                    batch.push_back(leaf.enc);
                    token_sets.push_back(&leaf.tokens);
                }

                auto selected = evaluate_batch_selected_for_search(batch, token_sets);

                ++leaf_batches;
                for (std::size_t i = 0; i < leaves.size(); ++i) {
                    EvalCacheEntry entry;
                    entry.moves  = leaves[i].moves;
                    entry.priors = std::move(selected.priors[i]);
                    entry.value  = sanitize_value(selected.values[i]);
                    remember_eval(leaves[i].key, entry);

                    for (const auto& path : leaves[i].paths) {
                        if (!has_sim_budget()) {
                            break;
                        }
                        expand_from_entry(path.back(), entry);
                        backup(path, entry.value);
                        ++sims;
                    }
                }
            }

            emit_due_checkpoints();

            auto current = Clock::now();
            if (current >= next_progress) {
                std::cout << progress_line(
                               nodes_[root_index_],
                               nodes_,
                               root_state_.pos(),
                               start,
                               sims + seeded_sims,
                               current_time_limit,
                               sims > 0 ? static_cast<double>(depth_sum) / sims : 0.0,
                               max_depth,
                               config_.verbose_search)
                          << std::endl;
                next_progress = current + std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double>(config_.progress_interval));
            }


            SearchNode& root_ref = nodes_[root_index_];
            if (root_ref.children.empty()) {
                continue;
            }

            double elapsed = elapsed_seconds(start);
            if (root_ref.children.size() == 1 && elapsed > 0.1) {
                break;
            }

            std::vector<int> sorted_child_indices;
            sorted_child_indices.reserve(root_state_.ply() < 512 ? nodes_[root_index_].children.size() : 0);
            for (const auto& child : nodes_[root_index_].children) {
                sorted_child_indices.push_back(child.node_index);
            }
            
            if (sorted_child_indices.empty()) {
                continue;
            }

            std::sort(
              sorted_child_indices.begin(),
              sorted_child_indices.end(),
              [&](int a_idx, int b_idx) {
                  const SearchNode& lhs = nodes_[a_idx];
                  const SearchNode& rhs = nodes_[b_idx];
                  if (lhs.visit_count != rhs.visit_count) return lhs.visit_count > rhs.visit_count;
                  return lhs.q_value() > rhs.q_value();
              });

            int best_node_idx = sorted_child_indices[0];
            int second_node_idx = sorted_child_indices.size() > 1 ? sorted_child_indices[1] : -1;
            
            float best_q = -nodes_[best_node_idx].q_value();
            
            uint16_t best_move_raw = 0;
            for (const auto& child : nodes_[root_index_].children) {
                if (child.node_index == best_node_idx) {
                    best_move_raw = child.move_raw;
                    break;
                }
            }

            if (!best_move_tracker.has_value()) {
                best_move_tracker = best_move_raw;
                last_best_change_sims = sims + seeded_sims;
                last_best_change_elapsed = elapsed;
            }

            if (!peak_best_q.has_value() || best_q > *peak_best_q) {
                peak_best_q = best_q;
            }

            bool best_move_changed = best_move_tracker.has_value() && best_move_raw != *best_move_tracker;
            if (best_move_changed) {
                best_move_tracker = best_move_raw;
                last_best_change_sims = sims + seeded_sims;
                last_best_change_elapsed = elapsed;
            }

            // 1. Time Management States
            bool is_slipping = (sims > 800 && persistent_target_q >= 0.4f && best_q < (persistent_target_q - 0.02f));
            bool is_locally_unstable =
              sims > 800
              && (
                best_move_changed
                || (peak_best_q.has_value() && best_q < (*peak_best_q - 0.05f))
                || (initial_q.has_value() && best_q < (*initial_q - 0.08f)));

            if (!is_panic && best_q < -0.30f) {
                is_panic = true;
                current_time_limit = config_.panic_time;
                deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(current_time_limit));
            } else if (!is_panic && !is_instability_boosted && is_locally_unstable && current_time_limit < config_.panic_time) {
                current_time_limit = std::min(current_time_limit * 2.0, config_.panic_time);
                deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(current_time_limit));
                is_instability_boosted = true;
            } else if (!is_panic && best_q > 0.30f) {
                if (!target_q.has_value() || best_q > *target_q) target_q = best_q;
                
                if (!is_defensive_boosted && sims > 300 && target_q.has_value() && best_q < (*target_q - 0.10f) && current_time_limit < config_.panic_time) {
                    current_time_limit = std::min(base_time_limit * 2.0, config_.panic_time);
                    deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(current_time_limit));
                    is_defensive_boosted = true;
                }
            } else if (!is_panic && !is_sharp_drop_boosted && initial_q.has_value() && sims > 300 && best_q < (*initial_q - 0.30f) && current_time_limit < config_.panic_time) {
                current_time_limit = std::min(base_time_limit * 2.0, config_.panic_time);
                deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(current_time_limit));
                is_sharp_drop_boosted = true;
            }

            // 2. Dynamic Drip-Feed Extensions (Triggers at 85% of current limit)
            if (!is_panic && elapsed > current_time_limit * 0.85) {
                double ceiling = current_time_limit;
                bool should_extend = false;

                double total_visits = static_cast<double>(std::max(1, nodes_[root_index_].visit_count));

                if (is_slipping && current_time_limit < config_.panic_time) {
                    should_extend = true;
                    ceiling = config_.panic_time;
                } else if (best_move_changed && current_time_limit < config_.panic_time) {
                    should_extend = true;
                    ceiling = config_.panic_time;
                } else if (!is_instability_boosted && !flat_tree_extended && (static_cast<double>(nodes_[best_node_idx].visit_count) / total_visits) < 0.20 && current_time_limit < config_.panic_time) {
                    should_extend = true;
                    ceiling = config_.panic_time;
                    flat_tree_extended = true;
                }

                if (should_extend) {
                    current_time_limit = std::min(current_time_limit * 2.0, ceiling);
                    deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(current_time_limit));
                }
            }

            // 3. Early Stop
            if (elapsed > std::min(0.5, current_time_limit * 0.1) && (sims + seeded_sims) >= effective_min_sims) {
                double nps      = sims / std::max(elapsed, 1e-6);
                double rem_time = std::max(0.0, std::chrono::duration<double>(deadline - current).count());
                bool can_early_stop = true;
                if (is_slipping && current_time_limit < config_.panic_time) {
                    can_early_stop = false;
                }
                if (best_move_tracker.has_value()) {
                    int sims_since_best_change = (sims + seeded_sims) - last_best_change_sims;
                    double seconds_since_best_change = elapsed - last_best_change_elapsed;
                    if (sims_since_best_change < 1500 || seconds_since_best_change < 1.5) {
                        can_early_stop = false;
                    }
                }

                if (can_early_stop) {
                    int second_visits = second_node_idx >= 0 ? nodes_[second_node_idx].visit_count : 0;
                    const SearchNode& best_child_ref = nodes_[best_node_idx];
                    
                    // Uncatchable gap
                    if (best_child_ref.visit_count > second_visits + static_cast<int>(nps * rem_time * 1.5)) {
                        break;
                    }
                    // Obvious ratio
                    if (elapsed > current_time_limit * 0.2) {
                        double total_visits  = static_cast<double>(std::max(1, nodes_[root_index_].visit_count));
                        double win_bonus     = best_q > 0.85f ? 0.05 : 0.0;
                        double obvious_ratio = (ply < 30 ? 0.45 : 0.65) - win_bonus;
                        
                        if ((best_child_ref.visit_count / total_visits) > obvious_ratio) {
                            break;
                        }
                    }
                }
            }
        }

        int best_node_final = -1;
        int max_visits = -1;
        uint16_t best_move_raw_final = 0;
        bool have_non_draw_candidate = false;
        for (const auto& child : nodes_[root_index_].children) {
            if (!root_allowed_moves_.empty() && root_allowed_moves_.find(child.move_raw) == root_allowed_moves_.end()) {
                continue;
            }
            float child_q_for_stm = -nodes_[child.node_index].q_value();
            if (child_q_for_stm > 0.03f && !root_move_claims_draw(Move(child.move_raw))) {
                have_non_draw_candidate = true;
                break;
            }
        }
        
        for (const auto& child : nodes_[root_index_].children) {
            if (!root_allowed_moves_.empty() && root_allowed_moves_.find(child.move_raw) == root_allowed_moves_.end()) {
                continue;
            }
            float child_q_for_stm = -nodes_[child.node_index].q_value();
            if (have_non_draw_candidate && child_q_for_stm > 0.03f && root_move_claims_draw(Move(child.move_raw))) {
                continue;
            }
            if (nodes_[child.node_index].visit_count > max_visits) {
                max_visits = nodes_[child.node_index].visit_count;
                best_node_final = child.node_index;
                best_move_raw_final = child.move_raw;
            }
        }

        if (best_node_final == -1) {
            root_allowed_moves_ = std::move(original_root_allowed_moves);
            return std::nullopt;
        }

        last_stats_.sims             = sims;
        last_stats_.seeded_sims      = seeded_sims;
        last_stats_.elapsed          = elapsed_seconds(start);
        last_stats_.best_move        = move_to_uci(root_state_.pos(), Move(best_move_raw_final));
        last_stats_.best_visits      = nodes_[best_node_final].visit_count;
        last_stats_.best_q           = -nodes_[best_node_final].q_value();
        if (tb_root_filter_active) {
            auto tb_it = root_tb_wdl_by_move.find(best_move_raw_final);
            if (tb_it != root_tb_wdl_by_move.end()) {
                last_stats_.best_q = static_cast<float>(tb_it->second);
            }
        }
        last_stats_.avg_depth        = sims > 0 ? static_cast<double>(depth_sum) / sims : 0.0;
        last_stats_.max_depth        = max_depth;
        last_stats_.root_children    = static_cast<int>(nodes_[root_index_].children.size());
        last_stats_.leaf_batches     = leaf_batches;
        last_stats_.cache_hits       = cache_hits_ - start_cache_hits;
        last_stats_.cache_misses     = cache_misses_ - start_cache_misses;
        last_stats_.cache_flushes    = cache_flushes_ - start_cache_flushes;
        last_stats_.deduped          = deduped;
        last_stats_.cache_size       = static_cast<int>(eval_cache_.size());

        std::cout << progress_line(
                       nodes_[root_index_],
                       nodes_,
                       root_state_.pos(),
                       start,
                       sims + seeded_sims,
                       current_time_limit,
                       last_stats_.avg_depth,
                       last_stats_.max_depth,
                       config_.verbose_search,
                       tb_root_filter_active ? std::optional<float>(last_stats_.best_q) : std::nullopt)
                  << std::endl;

        persistent_target_q = last_stats_.best_q;
        root_allowed_moves_ = std::move(original_root_allowed_moves);
        return Move(best_move_raw_final);
    }

    const SearchStats& last_stats() const { return last_stats_; }

    OpeningGraphSummary build_opening_graph_file(
      const std::string& path,
      int target_sims,
      int max_ply,
      double max_seconds = 0.0) {
        reset_tree_to_history({}, true);
        int completed_sims =
          run_fixed_root_search(std::max(1, target_sims), std::max(1, max_ply), std::max(0.0, max_seconds));
        std::cout << "[opening-graph] search complete sims=" << completed_sims
                  << " live_nodes=" << nodes_.size()
                  << " eval_cache=" << eval_cache_.size() << std::endl;
        if (!config_.opening_cache_file.empty()) {
            export_opening_cache_from_current_tree(config_.opening_cache_file, std::max(1, max_ply));
        }
        return export_opening_graph(path, std::max(1, max_ply), completed_sims);
    }

    OpeningGraphSummary build_opening_cache_all_white_roots(
      const std::string& path,
      int target_sims,
      int max_ply,
      double max_seconds = 0.0) {
        if (!config_.opening_cache_file.empty() && std::filesystem::exists(config_.opening_cache_file)) {
            load_opening_cache(config_.opening_cache_file);
            for (const auto& [key, entry] : opening_cache_) {
                EvalCacheEntry eval;
                eval.value = entry.value;
                eval.moves.reserve(entry.children.size());
                eval.priors.reserve(entry.children.size());
                for (const auto& child : entry.children) {
                    eval.moves.push_back(child.move_raw);
                    eval.priors.push_back(std::max(1e-6f, child.prior));
                }
                float total = 0.0f;
                for (float prior : eval.priors) {
                    total += prior;
                }
                if (total > 0.0f) {
                    for (float& prior : eval.priors) {
                        prior /= total;
                    }
                }
                remember_eval(key, eval);
            }
        }

        GameState start(128);
        UniqueMoves roots = unique_legal_moves(start.pos());
        const int per_root_sims = std::max(1, target_sims);
        const double per_root_seconds =
          max_seconds > 0.0 ? std::max(1.0, max_seconds / static_cast<double>(std::max<std::size_t>(1, roots.moves.size()))) : 0.0;

        int total_sims = 0;
        std::size_t max_live_nodes = 0;
        auto started = Clock::now();

        for (std::size_t i = 0; i < roots.moves.size(); ++i) {
            Move root_move(roots.moves[i]);
            std::vector<uint16_t> history{roots.moves[i]};
            reset_tree_to_history(history, false);
            int sims = run_fixed_root_search(per_root_sims, std::max(1, max_ply - 1), per_root_seconds);
            total_sims += sims;
            max_live_nodes = std::max(max_live_nodes, nodes_.size());
            merge_opening_cache_from_current_tree(std::max(1, max_ply - 1));
            std::cout << "[opening-cache] root " << (i + 1) << "/" << roots.moves.size()
                      << " move=" << move_to_uci(start.pos(), root_move)
                      << " sims=" << sims
                      << " cache_entries=" << opening_cache_.size()
                      << " elapsed=" << std::fixed << std::setprecision(1)
                      << elapsed_seconds(started) << "s" << std::endl;
        }

        opening_cache_bytes_ = 0;
        for (auto& [key, entry] : opening_cache_) {
            std::sort(entry.children.begin(), entry.children.end(), [](const OpeningBookChild& lhs, const OpeningBookChild& rhs) {
                if (lhs.prior != rhs.prior) {
                    return lhs.prior > rhs.prior;
                }
                return lhs.move_raw < rhs.move_raw;
            });
            opening_cache_bytes_ += estimate_opening_book_entry_bytes(entry);
        }
        save_opening_cache(path);

        return {opening_cache_.size(), max_live_nodes, total_sims, max_ply};
    }

    OpeningGraphSummary load_opening_graph_as_tree(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open opening graph for read: " + path);
        }

        std::string tag;
        int         stored_sims = 0;
        int         stored_max_ply = 0;
        std::size_t node_count = 0;
        std::size_t edge_count = 0;

        in >> tag;
        if (tag != "SCG1") {
            throw std::runtime_error("unsupported opening graph file: " + path);
        }
        in >> tag >> stored_sims;
        if (tag != "sims") {
            throw std::runtime_error("bad opening graph sims header: " + path);
        }
        in >> tag >> stored_max_ply;
        if (tag != "max_ply") {
            throw std::runtime_error("bad opening graph max_ply header: " + path);
        }
        in >> tag >> node_count;
        if (tag != "nodes") {
            throw std::runtime_error("bad opening graph nodes header: " + path);
        }
        in >> tag >> edge_count;
        if (tag != "edges") {
            throw std::runtime_error("bad opening graph edges header: " + path);
        }

        std::vector<StoredGraphNode> graph(node_count);
        for (std::size_t n = 0; n < node_count; ++n) {
            std::string key_hex;
            std::size_t node_id = 0;
            std::size_t child_count = 0;
            in >> tag >> node_id >> key_hex >> graph[n].visits >> graph[n].value_sum >> child_count;
            if (!in || tag != "node" || node_id != n) {
                throw std::runtime_error("bad opening graph node payload: " + path);
            }
            graph[n].key = static_cast<Key>(std::stoull(key_hex, nullptr, 16));
            graph[n].children.resize(child_count);
            for (std::size_t e = 0; e < child_count; ++e) {
                StoredGraphEdge edge;
                in >> tag >> edge.move_raw >> edge.child_id >> edge.visits >> edge.value_sum >> edge.prior;
                if (!in || tag != "edge" || edge.child_id >= node_count) {
                    throw std::runtime_error("bad opening graph edge payload: " + path);
                }
                graph[n].children[e] = edge;
            }
        }

        GameState start_board(128);
        if (graph.empty() || graph.front().key != start_board.pos().key()) {
            throw std::runtime_error("opening graph root is not startpos: " + path);
        }

        reset_tree_to_history({}, false);
        nodes_.clear();
        nodes_.reserve(graph.size());
        for (const auto& stored : graph) {
            int node_index = new_node(0.0f);
            SearchNode& node = nodes_[node_index];
            node.visit_count = static_cast<int>(std::min<std::uint32_t>(
              stored.visits, static_cast<std::uint32_t>(std::numeric_limits<int>::max())));
            node.value_sum = stored.value_sum;
            node.is_expanded = true;
        }

        std::size_t imported_edges = 0;
        for (std::size_t graph_id = 0; graph_id < graph.size(); ++graph_id) {
            SearchNode& node = nodes_[graph_id];
            node.children.reserve(graph[graph_id].children.size());
            for (const auto& edge : graph[graph_id].children) {
                node.children.push_back(
                  {edge.move_raw, static_cast<int>(edge.child_id), edge.prior});
                ++imported_edges;
            }
        }
        root_index_ = 0;

        std::cout << "[opening-graph] loaded as graph file=" << path
                  << " stored_nodes=" << graph.size()
                  << " live_nodes=" << nodes_.size()
                  << " edges=" << imported_edges
                  << " sims=" << stored_sims
                  << " max_ply=" << stored_max_ply << std::endl;
        return {nodes_.size(), imported_edges, stored_sims, stored_max_ply};
    }

  private:
    static constexpr std::uint32_t kOpeningCacheMagic = 0x31424353; // SCB1

    template <typename T>
    static void write_pod(std::ostream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!out) {
            throw std::runtime_error("failed writing opening cache");
        }
    }

    template <typename T>
    static void read_pod(std::istream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) {
            throw std::runtime_error("failed reading opening cache");
        }
    }

    template <typename T>
    static std::string pod_to_hex(const T& value) {
        std::ostringstream ss;
        ss << std::hex << std::nouppercase << value;
        return ss.str();
    }

    void reset_tree_to_history(const std::vector<uint16_t>& raw_moves, bool clear_eval_cache) {
        root_state_.set_from_raw_history(raw_moves);
        if (clear_eval_cache) {
            eval_cache_.clear();
            eval_cache_order_.clear();
            next_eval_cache_stamp_ = 0;
        }
        nodes_.clear();
        root_index_ = new_node(1.0f);
    }

    void reset_tree_only_to_history(const std::vector<uint16_t>& raw_moves) {
        root_state_.set_from_raw_history(raw_moves);
        nodes_.clear();
        root_index_ = new_node(1.0f);
    }

    void compact_to_reachable_root() {
        if (root_index_ < 0 || root_index_ >= static_cast<int>(nodes_.size())) {
            nodes_.clear();
            root_index_ = new_node(1.0f);
            return;
        }

        std::vector<int> remap(nodes_.size(), -1);
        std::vector<int> old_order;
        old_order.reserve(nodes_.size());
        std::deque<int> queue;

        remap[root_index_] = 0;
        old_order.push_back(root_index_);
        queue.push_back(root_index_);

        while (!queue.empty()) {
            int old_index = queue.front();
            queue.pop_front();
            for (const auto& child : nodes_[old_index].children) {
                if (child.node_index < 0 || child.node_index >= static_cast<int>(nodes_.size())) {
                    continue;
                }
                if (remap[child.node_index] >= 0) {
                    continue;
                }
                remap[child.node_index] = static_cast<int>(old_order.size());
                old_order.push_back(child.node_index);
                queue.push_back(child.node_index);
            }
        }

        std::vector<SearchNode> compact;
        compact.reserve(old_order.size());
        for (int old_index : old_order) {
            SearchNode node = nodes_[old_index];
            std::vector<ChildLink> kept;
            kept.reserve(node.children.size());
            for (const auto& child : node.children) {
                if (child.node_index < 0 || child.node_index >= static_cast<int>(remap.size())) {
                    continue;
                }
                int mapped_child = remap[child.node_index];
                if (mapped_child < 0) {
                    continue;
                }
                kept.push_back({child.move_raw, mapped_child, child.prior});
            }
            node.children = std::move(kept);
            compact.push_back(std::move(node));
        }

        std::size_t old_size = nodes_.size();
        nodes_ = std::move(compact);
        root_index_ = 0;
        std::cout << "[opening-graph] compacted reachable_nodes=" << nodes_.size()
                  << " dropped_nodes=" << (old_size - nodes_.size()) << std::endl;
    }

    static bool is_history_prefix(
      const std::vector<uint16_t>& prefix,
      const std::vector<uint16_t>& full) {
        return prefix.size() <= full.size()
          && std::equal(prefix.begin(), prefix.end(), full.begin());
    }

    void sync_position_from_history(const std::vector<uint16_t>& raw_moves) {
        const auto& current = root_state_.history_raw();
        if (current == raw_moves) {
            return;
        }

        if (is_history_prefix(current, raw_moves)) {
            for (std::size_t i = current.size(); i < raw_moves.size(); ++i) {
                advance_root(Move(raw_moves[i]));
            }
            return;
        }

        if (raw_moves.empty()
            && !config_.opening_graph_file.empty()
            && std::filesystem::exists(config_.opening_graph_file)) {
            load_opening_graph_as_tree(config_.opening_graph_file);
            return;
        }

        reset_tree_only_to_history(raw_moves);
    }

    const EvalCacheEntry* lookup_eval(Key key) {
        auto it = eval_cache_.find(key);
        if (it != eval_cache_.end()) {
            return &it->second.entry;
        }
        return nullptr;
    }

    const OpeningBookEntry* lookup_opening_entry(Key key) const {
        auto it = opening_cache_.find(key);
        if (it != opening_cache_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    void save_opening_cache(const std::string& path) const {
        std::filesystem::path target(path);
        if (target.has_parent_path()) {
            std::filesystem::create_directories(target.parent_path());
        }

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open opening cache for write: " + path);
        }

        write_pod(out, kOpeningCacheMagic);
        std::uint32_t version = 3;
        write_pod(out, version);
        std::uint64_t entry_count = static_cast<std::uint64_t>(opening_cache_.size());
        write_pod(out, entry_count);

        for (const auto& [key, entry] : opening_cache_) {
            write_pod(out, key);
            write_pod(out, entry.value);
            std::uint16_t child_count = static_cast<std::uint16_t>(std::min<std::size_t>(entry.children.size(), 65535));
            write_pod(out, child_count);
            for (std::size_t i = 0; i < child_count; ++i) {
                write_pod(out, entry.children[i].move_raw);
                write_pod(out, entry.children[i].prior);
            }
            if (!out) {
                throw std::runtime_error("failed writing opening cache payload");
            }
        }
        out.flush();
        if (!out) {
            throw std::runtime_error("failed finalizing opening cache file");
        }

        std::uintmax_t bytes = std::filesystem::file_size(target);
        std::cout << "[opening-book] saved file=" << target.string()
                  << " size_mb=" << std::fixed << std::setprecision(1)
                  << (static_cast<double>(bytes) / (1024.0 * 1024.0))
                  << " entries=" << opening_cache_.size() << std::endl;
    }

    void load_opening_cache(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open opening cache for read: " + path);
        }

        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        std::uint64_t entry_count = 0;
        read_pod(in, magic);
        read_pod(in, version);
        read_pod(in, entry_count);

        if (magic != kOpeningCacheMagic || (version != 2 && version != 3)) {
            throw std::runtime_error("unsupported opening cache file: " + path);
        }

        opening_cache_.clear();
        opening_cache_bytes_ = 0;
        opening_cache_.reserve(static_cast<std::size_t>(entry_count));

        for (std::uint64_t i = 0; i < entry_count; ++i) {
            Key key = 0;
            OpeningBookEntry entry;
            std::uint16_t child_count = 0;

            read_pod(in, key);
            read_pod(in, entry.value);
            if (version == 2) {
                read_pod(in, entry.total_visits);
            }
            read_pod(in, child_count);
            entry.children.resize(child_count);
            for (std::size_t j = 0; j < child_count; ++j) {
                read_pod(in, entry.children[j].move_raw);
                if (version == 2) {
                    read_pod(in, entry.children[j].visits);
                    read_pod(in, entry.children[j].value_sum);
                    entry.children[j].prior = 0.0f;
                } else {
                    read_pod(in, entry.children[j].prior);
                }
            }

            opening_cache_bytes_ += estimate_opening_book_entry_bytes(entry);
            opening_cache_.emplace(key, std::move(entry));
        }

        std::cout << "[opening-book] loaded file=" << path
                  << " entries=" << opening_cache_.size()
                  << " size_mb=" << std::fixed << std::setprecision(1)
                  << (static_cast<double>(opening_cache_bytes_) / (1024.0 * 1024.0))
                  << std::endl;
    }

    void build_opening_cache() {
        const std::size_t budget_bytes = static_cast<std::size_t>(std::max(0, config_.opening_cache_mb)) * 1024ull * 1024ull;
        const std::size_t max_entries =
          config_.opening_cache_max_entries > 0 ? static_cast<std::size_t>(config_.opening_cache_max_entries) : std::numeric_limits<std::size_t>::max();
        if (budget_bytes == 0 && config_.opening_cache_max_entries <= 0) {
            return;
        }

        opening_cache_.clear();
        opening_cache_bytes_ = 0;
        opening_cache_.reserve(std::max<std::size_t>(4096, budget_bytes / 768));

        std::cout << "[opening-book] building budget=" << config_.opening_cache_mb
                  << "MB max_entries="
                  << (config_.opening_cache_max_entries > 0 ? std::to_string(config_.opening_cache_max_entries) : std::string("unlimited"))
                  << " max_ply=" << config_.opening_cache_max_ply
                  << " full_ply=" << config_.opening_cache_full_ply
                  << " branching=" << config_.opening_cache_branching
                  << " sims=" << config_.opening_book_sims << std::endl;

        auto started = Clock::now();
        auto deadline = started + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(config_.opening_cache_max_seconds));

        HarnessConfig builder_cfg = config_;
        builder_cfg.opening_cache_mb = 0;
        builder_cfg.opening_cache_max_entries = 0;
        builder_cfg.opening_cache_file.clear();
        builder_cfg.opening_cache_build_only = false;
        builder_cfg.dynamic_cpuct = false;
        PUCTPlayer worker(model_, builder_cfg);

        std::deque<OpeningTask> queue;
        queue.push_back(OpeningTask{});
        for (const auto& line : openings()) {
            GameState board(128);
            for (const auto& uci : line) {
                Move move = parse_uci_move(board.pos(), uci);
                if (move == Move::none()) {
                    break;
                }
                board.push(move);
            }
            OpeningTask task;
            task.ply = board.ply();
            task.history = board.history_raw();
            queue.push_back(std::move(task));
        }

        int expanded = 0;
        while (!queue.empty()
               && (budget_bytes == 0 || opening_cache_bytes_ < budget_bytes)
               && opening_cache_.size() < max_entries
               && Clock::now() < deadline) {
            OpeningTask task = std::move(queue.front());
            queue.pop_front();

            GameState board(128);
            board.set_from_raw_history(task.history);
            Key key = board.pos().key();
            if (opening_cache_.find(key) != opening_cache_.end()) {
                continue;
            }

            UniqueMoves legal = unique_legal_moves(board.pos());
            TerminalInfo term = terminal_info(board.pos(), legal);
            if (term.is_terminal) {
                continue;
            }

            worker.reset_tree_to_history(task.history, false);
            OpeningBookEntry entry =
              worker.build_opening_book_entry_for_current_root(config_.opening_book_sims, config_.opening_cache_branching);

            std::size_t est = estimate_opening_book_entry_bytes(entry);
            if ((budget_bytes > 0 && opening_cache_bytes_ + est > budget_bytes)
                || opening_cache_.size() >= max_entries) {
                queue.clear();
                break;
            }

            opening_cache_.emplace(key, entry);
            opening_cache_bytes_ += est;
            ++expanded;

            if (task.ply >= config_.opening_cache_max_ply) {
                continue;
            }

            int expand_count = static_cast<int>(entry.children.size());
            if (task.ply >= config_.opening_cache_full_ply) {
                expand_count = std::min(expand_count, config_.opening_cache_branching);
            }
            for (int j = 0; j < expand_count; ++j) {
                OpeningTask next;
                next.history = task.history;
                next.history.push_back(entry.children[j].move_raw);
                next.ply = task.ply + 1;
                queue.push_back(std::move(next));
            }

            if (expanded > 0 && expanded % 4096 == 0) {
                double elapsed = std::max(elapsed_seconds(started), 1e-6);
                std::cout << "[opening-book] entries=" << expanded
                          << " queued=" << queue.size()
                          << " size_mb=" << std::fixed << std::setprecision(1)
                          << (static_cast<double>(opening_cache_bytes_) / (1024.0 * 1024.0))
                          << " eps=" << static_cast<int>(expanded / elapsed) << std::endl;
            }
        }

        std::cout << "[opening-book] ready entries=" << opening_cache_.size()
                  << " size_mb=" << std::fixed << std::setprecision(1)
                  << (static_cast<double>(opening_cache_bytes_) / (1024.0 * 1024.0))
                  << " build_time=" << std::setprecision(1) << elapsed_seconds(started) << "s"
                  << std::endl;
    }

    int new_node(float prior) {
        nodes_.emplace_back(prior);
        return static_cast<int>(nodes_.size()) - 1;
    }

    void remember_eval(Key key, const EvalCacheEntry& entry) {
        if (config_.cache_capacity <= 0) {
            return;
        }

        std::uint64_t stamp = ++next_eval_cache_stamp_;
        auto& slot = eval_cache_[key];
        slot.entry = entry;
        slot.stamp = stamp;
        eval_cache_order_.push_back({key, stamp});

        bool evicted = false;
        while (static_cast<int>(eval_cache_.size()) > config_.cache_capacity && !eval_cache_order_.empty()) {
            EvalCacheTouch touch = eval_cache_order_.front();
            eval_cache_order_.pop_front();
            auto it = eval_cache_.find(touch.key);
            if (it == eval_cache_.end() || it->second.stamp != touch.stamp) {
                continue;
            }
            eval_cache_.erase(it);
            evicted = true;
        }
        if (evicted) {
            ++cache_flushes_;
        }
    }

    float evaluate_leaf_value(const Position& pos, const std::vector<int>& tokens) {
        Key key = pos.key();
        if (const EvalCacheEntry* cached = lookup_eval(key)) {
            ++cache_hits_;
            return cached->value;
        }

        EncodedPosition encoded;
        encode_position(pos, encoded);
        std::vector<EncodedPosition> batch{encoded};
        std::vector<const std::vector<int>*> token_sets{&tokens};
        auto selected = evaluate_batch_selected_for_search(batch, token_sets);

        EvalCacheEntry entry;
        entry.priors = std::move(selected.priors[0]);
        entry.value = sanitize_value(selected.values[0]);
        ++cache_misses_;
        remember_eval(key, entry);
        return entry.value;
    }

    SelectedBatchEval evaluate_batch_selected_for_search(
      const std::vector<EncodedPosition>& batch,
      const std::vector<const std::vector<int>*>& token_sets) {
        SelectedBatchEval selected = model_.evaluate_batch_selected(batch, token_sets);
        if (value_model_ != nullptr) {
            SelectedBatchEval value_selected = value_model_->evaluate_batch_selected(batch, token_sets);
            selected.values = std::move(value_selected.values);
        }
        return selected;
    }

    void initialize_root_from_current_position() {
        SearchNode& root = nodes_[root_index_];
        if (root.is_expanded) {
            apply_root_prior_overrides();
            return;
        }

        UniqueMoves legal = unique_legal_moves(root_state_.pos());
        TerminalInfo term = terminal_info(root_state_.pos(), legal);
        if (term.is_terminal) {
            root.visit_count = 1;
            root.value_sum = term.value;
            root.is_expanded = true;
            return;
        }

        Key key = root_state_.pos().key();
        if (const OpeningBookEntry* opening_entry = lookup_opening_entry(key)) {
            seed_node_from_opening_entry(root_index_, *opening_entry, legal);
            ++cache_hits_;
            return;
        }

        EvalCacheEntry entry;
        const EvalCacheEntry* cached = lookup_eval(key);
        if (!cached) {
            EncodedPosition encoded;
            encode_position(root_state_.pos(), encoded);
            std::vector<EncodedPosition> batch{encoded};
            std::vector<const std::vector<int>*> token_sets{&legal.tokens};
            auto selected = evaluate_batch_selected_for_search(batch, token_sets);
            entry.moves = legal.moves;
            entry.priors = std::move(selected.priors[0]);
            entry.value = sanitize_value(selected.values[0]);
            remember_eval(key, entry);
            ++cache_misses_;
        } else {
            entry = *cached;
            ++cache_hits_;
        }
        expand_from_entry(root_index_, entry);
        apply_root_prior_overrides();
        nodes_[root_index_].value_sum += entry.value;
        nodes_[root_index_].visit_count += 1;
    }

    int run_fixed_root_search(int target_sims, int max_ply, double max_seconds = 0.0) {
        initialize_root_from_current_position();
        int sims = 0;
        max_ply = std::max(1, max_ply);
        int next_report = 1000;
        const Key root_key = root_state_.pos().key();
        const int root_ply = root_state_.ply();
        const auto started = Clock::now();
        const auto deadline =
          started + std::chrono::duration_cast<Clock::duration>(
                      std::chrono::duration<double>(std::max(0.0, max_seconds)));
        auto keep_searching = [&]() {
            if (sims >= target_sims) {
                return false;
            }
            return max_seconds <= 0.0 || Clock::now() < deadline;
        };

        auto rewind_search = [&](int& pushed, const char* stage) {
            while (pushed > 0) {
                root_state_.pop(false);
                --pushed;
            }
            if (root_state_.ply() != root_ply || root_state_.pos().key() != root_key
                || !root_state_.pos().pos_is_ok()) {
                std::ostringstream ss;
                ss << "opening graph search state corrupted after " << stage
                   << " root_ply=" << root_ply
                   << " current_ply=" << root_state_.ply();
                throw std::runtime_error(ss.str());
            }
        };

        while (keep_searching()) {
            std::vector<LeafGroup>       leaves;
            std::unordered_map<Key, int> leaf_group_index;
            int                          duplicate_streak = 0;
            int                          collect_attempts = 0;
            const int                    min_useful_batch =
              std::max(1, std::min(config_.eval_batch_size / 4, 16));
            const int                    max_collect_attempts =
              std::max(config_.eval_batch_size * std::max(1, config_.collect_dup_limit), config_.eval_batch_size * 4);

            while (
              static_cast<int>(leaves.size()) < config_.eval_batch_size
              && keep_searching()
              && collect_attempts < max_collect_attempts) {
                ++collect_attempts;
                int              node_index = root_index_;
                std::vector<int> path{node_index};
                int              pushed = 0;

                while (nodes_[node_index].is_expanded && !nodes_[node_index].children.empty()) {
                    auto [move_raw, child_index] =
                      select_child(
                        node_index,
                        node_index == root_index_ ? config_.opening_branch_visit_cap : 0,
                        node_index == root_index_ ? root_probe_target_for_child_count(nodes_[node_index].children.size()) : 0);
                    root_state_.push(Move(move_raw), false);
                    ++pushed;
                    node_index = child_index;
                    nodes_[node_index].in_flight += 1;
                    path.push_back(node_index);
                }

                int depth = static_cast<int>(path.size()) - 1;
                UniqueMoves legal = unique_legal_moves(root_state_.pos());
                TerminalInfo term = terminal_info(root_state_.pos(), legal, depth);
                if (term.is_terminal) {
                    backup(path, term.value);
                    ++sims;
                    if (sims >= next_report) {
                        std::cout << "[opening-graph] sims=" << sims
                                  << " live_nodes=" << nodes_.size()
                                  << " cache=" << eval_cache_.size() << std::endl;
                        next_report += 1000;
                    }
                    rewind_search(pushed, "terminal");
                    continue;
                }

                if (depth >= max_ply) {
                    float value = evaluate_leaf_value(root_state_.pos(), legal.tokens);
                    backup(path, value);
                    ++sims;
                    if (sims >= next_report) {
                        std::cout << "[opening-graph] sims=" << sims
                                  << " live_nodes=" << nodes_.size()
                                  << " cache=" << eval_cache_.size() << std::endl;
                        next_report += 1000;
                    }
                    rewind_search(pushed, "depth-cap");
                    continue;
                }

                Key key = root_state_.pos().key();
                const EvalCacheEntry* cached = lookup_eval(key);
                if (cached != nullptr) {
                    ++cache_hits_;
                    expand_from_entry(node_index, *cached);
                    backup(path, cached->value);
                    ++sims;
                    if (sims >= next_report) {
                        std::cout << "[opening-graph] sims=" << sims
                                  << " live_nodes=" << nodes_.size()
                                  << " cache=" << eval_cache_.size() << std::endl;
                        next_report += 1000;
                    }
                    rewind_search(pushed, "cache-hit");
                    continue;
                }

                auto group_it = leaf_group_index.find(key);
                if (group_it != leaf_group_index.end()) {
                    leaves[group_it->second].paths.push_back(std::move(path));
                    ++duplicate_streak;
                    rewind_search(pushed, "dedup");
                    if (
                      static_cast<int>(leaves.size()) >= min_useful_batch
                      && duplicate_streak >= config_.collect_dup_limit) {
                        break;
                    }
                    continue;
                }

                EncodedPosition encoded;
                encode_position(root_state_.pos(), encoded);

                LeafGroup leaf;
                leaf.key = key;
                leaf.enc = encoded;
                leaf.moves = std::move(legal.moves);
                leaf.tokens = std::move(legal.tokens);
                leaf.paths.push_back(std::move(path));
                leaf_group_index[key] = static_cast<int>(leaves.size());
                leaves.push_back(std::move(leaf));
                duplicate_streak = 0;

                rewind_search(pushed, "leaf-collect");
            }

            if (leaves.empty()) {
                break;
            }

            std::vector<EncodedPosition> batch;
            std::vector<const std::vector<int>*> token_sets;
            batch.reserve(leaves.size());
            token_sets.reserve(leaves.size());
            for (const auto& leaf : leaves) {
                batch.push_back(leaf.enc);
                token_sets.push_back(&leaf.tokens);
            }

            auto selected = evaluate_batch_selected_for_search(batch, token_sets);
            for (std::size_t i = 0; i < leaves.size() && sims < target_sims; ++i) {
                EvalCacheEntry entry;
                entry.moves = leaves[i].moves;
                entry.priors = std::move(selected.priors[i]);
                entry.value = sanitize_value(selected.values[i]);
                remember_eval(leaves[i].key, entry);

                for (const auto& path : leaves[i].paths) {
                    expand_from_entry(path.back(), entry);
                    backup(path, entry.value);
                    ++sims;
                    if (sims >= next_report) {
                        std::cout << "[opening-graph] sims=" << sims
                                  << " live_nodes=" << nodes_.size()
                                  << " cache=" << eval_cache_.size() << std::endl;
                        next_report += 1000;
                    }
                    if (sims >= target_sims) {
                        break;
                    }
                }
            }
        }
        return sims;
    }

    OpeningGraphSummary export_opening_graph(const std::string& path, int max_ply, int sims) {
        std::filesystem::path target(path);
        if (target.has_parent_path()) {
            std::filesystem::create_directories(target.parent_path());
        }

        std::vector<StoredGraphNode> graph_nodes;
        std::unordered_map<Key, uint32_t> key_to_id;
        std::vector<std::unordered_map<std::uint64_t, std::size_t>> edge_index;

        auto assign_node = [&](Key key) -> uint32_t {
            auto it = key_to_id.find(key);
            if (it != key_to_id.end()) {
                return it->second;
            }
            uint32_t id = static_cast<uint32_t>(graph_nodes.size());
            key_to_id.emplace(key, id);
            graph_nodes.push_back(StoredGraphNode{key});
            edge_index.emplace_back();
            return id;
        };

        GameState board(4096);
        board.set_from_raw_history(root_state_.history_raw(), false);
        assign_node(board.pos().key());
        std::function<void(int, int)> dfs = [&](int node_index, int depth) {
            SearchNode& live_node = nodes_[node_index];
            Key key = board.pos().key();
            uint32_t graph_id = assign_node(key);
            StoredGraphNode& stored = graph_nodes[graph_id];
            stored.visits += static_cast<uint32_t>(std::max(0, live_node.visit_count));
            stored.value_sum += live_node.value_sum;

            if (depth >= max_ply) {
                return;
            }

            for (const auto& live_edge : live_node.children) {
                const SearchNode& live_child = nodes_[live_edge.node_index];
                Move move(live_edge.move_raw);
                board.push(move, false);
                Key child_key = board.pos().key();
                uint32_t child_id = assign_node(child_key);
                StoredGraphNode& stored_after_assign = graph_nodes[graph_id];
                std::uint64_t edge_key = (static_cast<std::uint64_t>(live_edge.move_raw) << 32) | child_id;
                auto& edge_map = edge_index[graph_id];
                auto edge_it = edge_map.find(edge_key);
                if (edge_it == edge_map.end()) {
                    StoredGraphEdge edge;
                    edge.move_raw = live_edge.move_raw;
                    edge.child_id = child_id;
                    edge.visits = static_cast<uint32_t>(std::max(0, live_child.visit_count));
                    edge.value_sum = live_child.value_sum;
                    edge.prior = live_edge.prior;
                    edge_map.emplace(edge_key, stored_after_assign.children.size());
                    stored_after_assign.children.push_back(edge);
                } else {
                    StoredGraphEdge& edge = stored_after_assign.children[edge_it->second];
                    edge.visits += static_cast<uint32_t>(std::max(0, live_child.visit_count));
                    edge.value_sum += live_child.value_sum;
                    edge.prior = std::max(edge.prior, live_edge.prior);
                }

                dfs(live_edge.node_index, depth + 1);
                board.pop(false);
            }
        };
        dfs(root_index_, 0);
        std::cout << "[opening-graph] export collected nodes=" << graph_nodes.size() << std::endl;

        std::size_t edge_count = 0;
        for (auto& node : graph_nodes) {
            std::sort(node.children.begin(), node.children.end(), [](const StoredGraphEdge& lhs, const StoredGraphEdge& rhs) {
                if (lhs.visits != rhs.visits) {
                    return lhs.visits > rhs.visits;
                }
                return lhs.move_raw < rhs.move_raw;
            });
            edge_count += node.children.size();
        }

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open opening graph for write: " + path);
        }
        out << "SCG1\n";
        out << "sims " << sims << "\n";
        out << "max_ply " << max_ply << "\n";
        out << "nodes " << graph_nodes.size() << "\n";
        out << "edges " << edge_count << "\n";
        for (std::size_t node_id = 0; node_id < graph_nodes.size(); ++node_id) {
            const auto& node = graph_nodes[node_id];
            out << "node " << node_id << ' ' << pod_to_hex(node.key) << ' ' << node.visits << ' '
                << std::setprecision(9) << node.value_sum << ' ' << node.children.size() << "\n";
            for (const auto& edge : node.children) {
                out << "edge " << edge.move_raw << ' ' << edge.child_id << ' ' << edge.visits << ' '
                    << std::setprecision(9) << edge.value_sum << ' ' << edge.prior << "\n";
            }
        }
        if (!out) {
            throw std::runtime_error("failed writing opening graph");
        }

        if (!graph_nodes.empty()) {
            std::cout << "[opening-graph] root children:" << std::endl;
            const auto& root = graph_nodes.front();
            for (std::size_t i = 0; i < std::min<std::size_t>(12, root.children.size()); ++i) {
                const auto& edge = root.children[i];
                std::cout << "  " << move_to_uci(root_state_.pos(), Move(edge.move_raw))
                          << " visits=" << edge.visits
                          << " prior=" << std::fixed << std::setprecision(4) << edge.prior
                          << " q=" << std::setprecision(4)
                          << (edge.visits > 0 ? (-edge.value_sum / static_cast<float>(edge.visits)) : 0.0f)
                          << std::endl;
            }
        }

        std::cout << "[opening-graph] saved file=" << path
                  << " nodes=" << graph_nodes.size()
                  << " edges=" << edge_count
                  << " sims=" << sims
                  << " max_ply=" << max_ply << std::endl;

        return {graph_nodes.size(), edge_count, sims, max_ply};
    }

    void export_opening_cache_from_current_tree(const std::string& path, int max_ply) {
        merge_opening_cache_from_current_tree(max_ply);
        save_opening_cache(path);
    }

    void merge_opening_cache_from_current_tree(int max_ply) {
        max_ply = std::max(1, max_ply);

        GameState board(4096);
        board.set_from_raw_history(root_state_.history_raw(), false);

        std::function<void(int, int)> dfs = [&](int node_index, int depth) {
            if (node_index < 0 || node_index >= static_cast<int>(nodes_.size())) {
                return;
            }

            const SearchNode& live_node = nodes_[node_index];
            Key key = board.pos().key();
            const EvalCacheEntry* cached_eval = lookup_eval(key);
            if (cached_eval == nullptr || cached_eval->moves.empty() || cached_eval->priors.empty()) {
                return;
            }

            OpeningBookEntry& entry = opening_cache_[key];
            entry.total_visits = 1;
            entry.value = cached_eval->value;
            entry.children.clear();
            const std::size_t move_count = std::min(cached_eval->moves.size(), cached_eval->priors.size());
            entry.children.reserve(move_count);
            for (std::size_t i = 0; i < move_count; ++i) {
                entry.children.push_back({cached_eval->moves[i], 0, 0.0f, cached_eval->priors[i]});
            }

            if (depth >= max_ply) {
                return;
            }

            for (const auto& link : live_node.children) {
                if (link.node_index < 0 || link.node_index >= static_cast<int>(nodes_.size())) {
                    continue;
                }
                board.push(Move(link.move_raw), false);
                dfs(link.node_index, depth + 1);
                board.pop(false);
            }
        };

        dfs(root_index_, 0);

        opening_cache_bytes_ = 0;
        for (auto& [key, entry] : opening_cache_) {
            std::sort(entry.children.begin(), entry.children.end(), [](const OpeningBookChild& lhs, const OpeningBookChild& rhs) {
                if (lhs.prior != rhs.prior) {
                    return lhs.prior > rhs.prior;
                }
                return lhs.move_raw < rhs.move_raw;
            });
            opening_cache_bytes_ += estimate_opening_book_entry_bytes(entry);
        }

    }

    void expand_from_entry(int node_index, const EvalCacheEntry& entry) {
        if (entry.moves.empty()) {
            nodes_[node_index].is_expanded = true;
            return;
        }

        for (std::size_t i = 0; i < entry.moves.size(); ++i) {
            bool found = false;
            // Access children directly from nodes_ array because new_node can reallocate
            for (auto& child : nodes_[node_index].children) {
                if (child.move_raw == entry.moves[i]) {
                    child.prior = entry.priors[i];
                    found = true;
                    break;
                }
            }
            if (!found) {
                int child_index = new_node(entry.priors[i]);
                // Re-fetch parent after new_node because of potential reallocation
                nodes_[node_index].children.push_back({entry.moves[i], child_index, entry.priors[i]});
            }
        }
        nodes_[node_index].is_expanded = true;
    }

    void apply_root_prior_overrides() {
        if (root_prior_overrides_.empty() || root_index_ < 0 || root_index_ >= static_cast<int>(nodes_.size())) {
            return;
        }
        SearchNode& root = nodes_[root_index_];
        float total = 0.0f;
        for (const auto& child : root.children) {
            auto it = root_prior_overrides_.find(child.move_raw);
            if (it != root_prior_overrides_.end()) {
                total += std::max(1e-8f, it->second);
            }
        }
        if (total <= 0.0f) {
            return;
        }
        const float missing_prior = 1e-8f;
        float adjusted_total = 0.0f;
        for (const auto& child : root.children) {
            auto it = root_prior_overrides_.find(child.move_raw);
            adjusted_total += (it != root_prior_overrides_.end()) ? std::max(1e-8f, it->second) : missing_prior;
        }
        for (auto& child : root.children) {
            auto it = root_prior_overrides_.find(child.move_raw);
            float prior = (it != root_prior_overrides_.end()) ? std::max(1e-8f, it->second) : missing_prior;
            child.prior = prior / adjusted_total;
        }
    }

    void seed_node_from_opening_entry(int node_index, const OpeningBookEntry& entry, const UniqueMoves& legal) {
        if (nodes_[node_index].is_expanded) {
            return;
        }

        std::unordered_map<uint16_t, float> cached_priors;
        cached_priors.reserve(entry.children.size());
        float prior_total = 0.0f;
        for (const auto& child : entry.children) {
            if (child.prior > 0.0f) {
                cached_priors[child.move_raw] = child.prior;
                prior_total += child.prior;
            }
        }
        const float fallback_prior =
          legal.moves.empty() ? 1.0f : 1.0f / static_cast<float>(legal.moves.size());
        for (uint16_t move_raw : legal.moves) {
            float prior = fallback_prior;
            auto prior_it = cached_priors.find(move_raw);
            if (prior_it != cached_priors.end() && prior_total > 0.0f) {
                prior = std::max(1e-6f, prior_it->second / prior_total);
            }
            int child_index = -1;
            for (const auto& child : nodes_[node_index].children) {
                if (child.move_raw == move_raw) {
                    child_index = child.node_index;
                    break;
                }
            }
            if (child_index < 0) {
                child_index = new_node(prior);
                nodes_[node_index].children.push_back({move_raw, child_index, prior});
            } else {
                for (auto& child : nodes_[node_index].children) {
                    if (child.node_index == child_index) {
                        child.prior = prior;
                        break;
                    }
                }
            }
        }

        // The opening cache is a bounded value/coverage hint. Live PUCT owns
        // visit counts, so the saved tree cannot force a move or inflate NPS.
        nodes_[node_index].visit_count = 1;
        nodes_[node_index].value_sum = entry.value;
        nodes_[node_index].is_expanded = true;
    }

    OpeningBookEntry build_opening_book_entry_for_current_root(int target_live_sims, int max_children) {
        (void)max_children;
        UniqueMoves legal = unique_legal_moves(root_state_.pos());
        TerminalInfo term = terminal_info(root_state_.pos(), legal);
        if (term.is_terminal) {
            OpeningBookEntry entry;
            entry.value = term.value;
            entry.total_visits = 1;
            return entry;
        }

        Key key = root_state_.pos().key();
        const EvalCacheEntry* cached = lookup_eval(key);
        EvalCacheEntry root_eval;
        if (cached) {
            root_eval = *cached;
        } else {
            EncodedPosition encoded;
            encode_position(root_state_.pos(), encoded);
            std::vector<EncodedPosition> batch{encoded};
            std::vector<const std::vector<int>*> token_sets{&legal.tokens};
            auto selected = evaluate_batch_selected_for_search(batch, token_sets);
            root_eval.moves = legal.moves;
            root_eval.priors = std::move(selected.priors[0]);
            root_eval.value = sanitize_value(selected.values[0]);
            remember_eval(key, root_eval);
        }
        expand_from_entry(root_index_, root_eval);
        nodes_[root_index_].visit_count = 1;
        nodes_[root_index_].value_sum = root_eval.value;

        int live_sims = 0;
        while (live_sims < target_live_sims) {
            std::vector<LeafGroup>       leaves;
            std::unordered_map<Key, int> leaf_group_index;
            int                          duplicate_streak = 0;
            int                          collect_attempts = 0;
            const int                    min_useful_batch =
              std::max(1, std::min(config_.eval_batch_size / 4, 16));
            const int                    max_collect_attempts =
              std::max(config_.eval_batch_size * std::max(1, config_.collect_dup_limit), config_.eval_batch_size * 4);

            while (
              static_cast<int>(leaves.size()) < config_.eval_batch_size
              && live_sims < target_live_sims
              && collect_attempts < max_collect_attempts) {
                ++collect_attempts;
                int              node_index = root_index_;
                std::vector<int> path{node_index};
                int              pushed = 0;

                while (nodes_[node_index].is_expanded && !nodes_[node_index].children.empty()) {
                    auto [move_raw, child_index] = select_child(node_index, 0, node_index == root_index_ ? root_probe_target_for_child_count(nodes_[node_index].children.size()) : 0);
                    root_state_.push(Move(move_raw), false);
                    ++pushed;
                    node_index = child_index;
                    nodes_[node_index].in_flight += 1;
                    path.push_back(node_index);
                }

                int loop_depth = static_cast<int>(path.size()) - 1;
                UniqueMoves loop_legal = unique_legal_moves(root_state_.pos());
                TerminalInfo loop_term = terminal_info(root_state_.pos(), loop_legal, loop_depth);
                if (loop_term.is_terminal) {
                    backup(path, loop_term.value);
                    ++live_sims;
                    while (pushed-- > 0) {
                        root_state_.pop(false);
                    }
                    continue;
                }

                Key loop_key = root_state_.pos().key();
                const EvalCacheEntry* loop_cached = lookup_eval(loop_key);
                if (loop_cached != nullptr) {
                    expand_from_entry(node_index, *loop_cached);
                    backup(path, loop_cached->value);
                    ++live_sims;
                    while (pushed-- > 0) {
                        root_state_.pop(false);
                    }
                    continue;
                }

                auto group_it = leaf_group_index.find(loop_key);
                if (group_it != leaf_group_index.end()) {
                    leaves[group_it->second].paths.push_back(std::move(path));
                    ++duplicate_streak;
                    while (pushed-- > 0) {
                        root_state_.pop(false);
                    }
                    if (
                      static_cast<int>(leaves.size()) >= min_useful_batch
                      && duplicate_streak >= config_.collect_dup_limit) {
                        break;
                    }
                    continue;
                }

                EncodedPosition encoded;
                encode_position(root_state_.pos(), encoded);

                LeafGroup leaf;
                leaf.key = loop_key;
                leaf.enc = encoded;
                leaf.moves = std::move(loop_legal.moves);
                leaf.tokens = std::move(loop_legal.tokens);
                leaf.paths.push_back(std::move(path));
                leaf_group_index[loop_key] = static_cast<int>(leaves.size());
                leaves.push_back(std::move(leaf));
                duplicate_streak = 0;

                while (pushed-- > 0) {
                    root_state_.pop(false);
                }
            }

            if (leaves.empty()) {
                break;
            }

            std::vector<EncodedPosition> batch;
            std::vector<const std::vector<int>*> token_sets;
            batch.reserve(leaves.size());
            token_sets.reserve(leaves.size());
            for (const auto& leaf : leaves) {
                batch.push_back(leaf.enc);
                token_sets.push_back(&leaf.tokens);
            }

            auto selected = evaluate_batch_selected_for_search(batch, token_sets);
            for (std::size_t i = 0; i < leaves.size() && live_sims < target_live_sims; ++i) {
                EvalCacheEntry entry;
                entry.moves = leaves[i].moves;
                entry.priors = std::move(selected.priors[i]);
                entry.value = sanitize_value(selected.values[i]);
                remember_eval(leaves[i].key, entry);

                for (const auto& path : leaves[i].paths) {
                    expand_from_entry(path.back(), entry);
                    backup(path, entry.value);
                    ++live_sims;
                    if (live_sims >= target_live_sims) {
                        break;
                    }
                }
            }
        }

        OpeningBookEntry entry;
        entry.value = root_eval.value;
        entry.total_visits = 1;
        const std::size_t move_count = std::min(root_eval.moves.size(), root_eval.priors.size());
        entry.children.reserve(move_count);
        for (std::size_t i = 0; i < move_count; ++i) {
            entry.children.push_back({root_eval.moves[i], 0, 0.0f, root_eval.priors[i]});
        }
        return entry;
    }

    void backup(const std::vector<int>& path, float value) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            SearchNode& node = nodes_[*it];
            node.visit_count += 1;
            node.value_sum += value;
            if (node.in_flight > 0) {
                node.in_flight -= 1;
            }
            value = -value;
        }
    }

    int root_probe_target_for_child_count(std::size_t child_count) const {
        if (config_.root_probe_children <= 0 || config_.root_probe_visits_per_child <= 0 || child_count == 0) {
            return 0;
        }
        return std::min<int>(config_.root_probe_children, static_cast<int>(child_count));
    }

    std::pair<uint16_t, int> select_child(int node_index, int child_visit_cap = 0, int root_probe_children = 0) {
        const SearchNode& node = nodes_[node_index];
        if (root_probe_children > 0) {
            const ChildLink* best_probe_child = nullptr;
            int              best_probe_visits = std::numeric_limits<int>::max();
            float            best_probe_prior = -1.0f;
            int              seen = 0;
            for (const auto& child : node.children) {
                if (node_index == root_index_
                    && !root_allowed_moves_.empty()
                    && root_allowed_moves_.find(child.move_raw) == root_allowed_moves_.end()) {
                    continue;
                }
                if (seen++ >= root_probe_children) {
                    break;
                }
                const SearchNode& child_node = nodes_[child.node_index];
                int effective_visits = child_node.visit_count + child_node.in_flight;
                if (effective_visits >= config_.root_probe_visits_per_child) {
                    continue;
                }
                if (
                  effective_visits < best_probe_visits
                  || (effective_visits == best_probe_visits && child.prior > best_probe_prior)) {
                    best_probe_child = &child;
                    best_probe_visits = effective_visits;
                    best_probe_prior = child.prior;
                }
            }
            if (best_probe_child != nullptr) {
                return {best_probe_child->move_raw, best_probe_child->node_index};
            }
        }

        float total_sqrt = std::sqrt(static_cast<float>(std::max(1, node.visit_count + node.in_flight)));
        float current_cpuct = static_cast<float>(config_.cpuct);
        if (
          config_.cpuct_warmup_visits > 0
          && config_.cpuct_warmup_value > 0.0
          && node.visit_count < config_.cpuct_warmup_visits) {
            current_cpuct = static_cast<float>(config_.cpuct_warmup_value);
        } else if (config_.dynamic_cpuct) {
            int effective_visits = node.visit_count;
            if (config_.cpuct_warmup_visits > 0 && config_.cpuct_warmup_value > 0.0) {
                effective_visits = std::max(0, node.visit_count - config_.cpuct_warmup_visits);
            }
            float cpuct_val =
              static_cast<float>(
                config_.cpuct
                + config_.cpuct_scale * std::log((effective_visits + config_.cpuct_init) / config_.cpuct_init));
            current_cpuct = std::min(3.0f, cpuct_val);
        }

        const ChildLink* best_child = nullptr;
        float            best_score = -1e10f;

        // FPU: Unvisited nodes inherit parent evaluation
        float parent_q = node.q_value();

        for (const auto& child : node.children) {
            if (node_index == root_index_
                && !root_allowed_moves_.empty()
                && root_allowed_moves_.find(child.move_raw) == root_allowed_moves_.end()) {
                continue;
            }
            const SearchNode& child_node = nodes_[child.node_index];
            if (child_visit_cap > 0 && child_node.visit_count >= child_visit_cap) {
                continue;
            }
            
            float q;
            if (child_node.visit_count > 0) {
                q = -child_node.q_value();
            } else {
                q = parent_q; // First Play Urgency
            }
            
            q -= static_cast<float>(config_.virtual_loss * child_node.in_flight);
            float u = current_cpuct * child.prior * total_sqrt / (1.0f + child_node.visit_count + child_node.in_flight);
            float score = q + u;

            if (score > best_score) {
                best_score = score;
                best_child = &child;
            }
        }
        if (!best_child) {
            for (const auto& child : node.children) {
                if (node_index == root_index_
                    && !root_allowed_moves_.empty()
                    && root_allowed_moves_.find(child.move_raw) == root_allowed_moves_.end()) {
                    continue;
                }
                best_child = &child;
                break;
            }
            if (child_visit_cap > 0) {
                for (const auto& child : node.children) {
                    if (node_index == root_index_
                        && !root_allowed_moves_.empty()
                        && root_allowed_moves_.find(child.move_raw) == root_allowed_moves_.end()) {
                        continue;
                    }
                    const SearchNode& child_node = nodes_[child.node_index];
                    if (child_node.visit_count < child_visit_cap) {
                        best_child = &child;
                        break;
                    }
                }
            }
        }
        if (!best_child) {
            best_child = &node.children.front();
        }
        return {best_child->move_raw, best_child->node_index};
    }

    void init_tablebases() {
        if (!config_.syzygy_enable) {
            return;
        }
        if (config_.syzygy_path.empty() || !std::filesystem::exists(config_.syzygy_path)) {
            std::cout << "[syzygy] disabled path_missing=" << config_.syzygy_path << std::endl;
            config_.syzygy_enable = false;
            return;
        }

        scan_tablebase_signatures();
        Tablebases::init(config_.syzygy_path);
        syzygy_available_ = Tablebases::MaxCardinality > 0 && !syzygy_materials_.empty();
        std::cout << "[syzygy] "
                  << (syzygy_available_ ? "ready" : "disabled")
                  << " path=" << config_.syzygy_path
                  << " max_cardinality=" << Tablebases::MaxCardinality
                  << " signatures=" << syzygy_materials_.size()
                  << std::endl;
    }

    void scan_tablebase_signatures() {
        syzygy_materials_.clear();
        try {
            for (const auto& entry : std::filesystem::directory_iterator(config_.syzygy_path)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                if (ext != ".rtbw") {
                    continue;
                }
                syzygy_materials_.insert(entry.path().stem().string());
            }
        } catch (const std::exception& e) {
            std::cout << "[syzygy] signature scan failed: " << e.what() << std::endl;
        }
    }

    bool material_is_available(const Position& pos) {
        if (syzygy_materials_.empty()) {
            return false;
        }
        auto [first, second] = material_signatures(pos);
        return syzygy_materials_.find(first) != syzygy_materials_.end()
            || syzygy_materials_.find(second) != syzygy_materials_.end();
    }

    std::optional<TablebaseProbeInfo> lookup_tablebase_cache(Key key, bool need_dtz) {
        auto it = tb_cache_.find(key);
        if (it == tb_cache_.end() || !it->second.entry.ok) {
            return std::nullopt;
        }
        if (need_dtz && !it->second.entry.has_dtz) {
            return std::nullopt;
        }
        return it->second.entry;
    }

    void remember_tablebase(Key key, const TablebaseProbeInfo& entry) {
        if (config_.syzygy_cache_capacity <= 0) {
            return;
        }
        std::uint64_t stamp = ++next_tb_cache_stamp_;
        tb_cache_[key] = {entry, stamp};
        tb_cache_order_.push_back({key, stamp});
        while (static_cast<int>(tb_cache_.size()) > config_.syzygy_cache_capacity && !tb_cache_order_.empty()) {
            TablebaseCacheTouch touch = tb_cache_order_.front();
            tb_cache_order_.pop_front();
            auto it = tb_cache_.find(touch.key);
            if (it == tb_cache_.end() || it->second.stamp != touch.stamp) {
                continue;
            }
            tb_cache_.erase(it);
        }
    }

    std::optional<TablebaseProbeInfo> probe_tablebase(Position& pos, int piece_count, bool need_dtz) {
        if (!config_.syzygy_enable || !syzygy_available_) {
            return std::nullopt;
        }
        if (piece_count > config_.syzygy_max_pieces || piece_count > Tablebases::MaxCardinality) {
            ++tb_skipped_by_piece_count_;
            return std::nullopt;
        }
        if (!material_is_available(pos)) {
            ++tb_skipped_by_material_;
            return std::nullopt;
        }

        Key key = pos.key();
        if (auto cached = lookup_tablebase_cache(key, need_dtz)) {
            ++tb_hits_;
            return cached;
        }

        Tablebases::ProbeState state = Tablebases::OK;
        Tablebases::WDLScore wdl = Tablebases::probe_wdl(pos, &state);
        if (state == Tablebases::FAIL) {
            ++tb_misses_;
            return std::nullopt;
        }

        TablebaseProbeInfo info;
        info.ok = true;
        info.wdl = static_cast<int>(wdl);
        info.value = tablebase_value_from_wdl(info.wdl);

        if (need_dtz && wdl != Tablebases::WDLDraw) {
            Tablebases::ProbeState dtz_state = Tablebases::OK;
            int dtz = Tablebases::probe_dtz(pos, &dtz_state);
            if (dtz_state != Tablebases::FAIL) {
                info.has_dtz = true;
                info.dtz = dtz;
            }
        }

        remember_tablebase(key, info);
        ++tb_hits_;
        return info;
    }

    int tablebase_trap_score(Move move, int root_wdl) {
        int child_piece_count = root_state_.piece_count() - (root_state_.pos().capture(move) ? 1 : 0);
        StateInfo st;
        root_state_.pos().do_move(move, st);
        int trap_score = 0;
        for (const Move reply : MoveList<LEGAL>(root_state_.pos())) {
            int reply_piece_count = child_piece_count - (root_state_.pos().capture(reply) ? 1 : 0);
            StateInfo reply_st;
            root_state_.pos().do_move(reply, reply_st);
            auto reply_tb = probe_tablebase(root_state_.pos(), reply_piece_count, false);
            root_state_.pos().undo_move(reply);
            if (!reply_tb) {
                continue;
            }
            int result_for_root = -reply_tb->wdl;
            if (result_for_root > root_wdl) {
                ++trap_score;
            }
        }
        root_state_.pos().undo_move(move);
        return trap_score;
    }

    struct RootTablebaseFilter {
        std::unordered_set<uint16_t>     allowed_moves;
        std::unordered_map<uint16_t, int> wdl_by_move;
        std::string                      category;
        int                              candidate_count = 0;
    };

    int effective_root_wdl_after_move(const TablebaseProbeInfo& child_tb, const Position& child_pos) const {
        int root_wdl = -child_tb.wdl;
        if (root_wdl >= static_cast<int>(Tablebases::WDLWin)) {
            if (child_tb.has_dtz) {
                int remaining_rule50_plies = std::max(0, 100 - child_pos.rule50_count());
                if (std::abs(child_tb.dtz) > remaining_rule50_plies) {
                    return 0;
                }
            }
            return 1;
        }
        if (root_wdl <= static_cast<int>(Tablebases::WDLLoss)) {
            return -1;
        }
        return 0;
    }

    std::optional<RootTablebaseFilter> build_tablebase_root_filter() {
        auto root_tb = probe_tablebase(root_state_.pos(), root_state_.piece_count(), true);
        if (!root_tb) {
            return std::nullopt;
        }

        struct Candidate {
            uint16_t move_raw = 0;
            int      effective_wdl = 0;
        };

        std::vector<Candidate> candidates;
        for (const Move move : MoveList<LEGAL>(root_state_.pos())) {
            bool capture = root_state_.pos().capture(move);
            StateInfo st;
            root_state_.pos().do_move(move, st);
            int next_piece_count = root_state_.piece_count() - (capture ? 1 : 0);
            auto child_tb = probe_tablebase(root_state_.pos(), next_piece_count, true);
            if (child_tb) {
                int effective_wdl = effective_root_wdl_after_move(*child_tb, root_state_.pos());
                if (root_state_.pos().is_draw(1) || insufficient_material(root_state_.pos()) || st.repetition != 0) {
                    effective_wdl = 0;
                }
                candidates.push_back({move.raw(), effective_wdl});
            }
            root_state_.pos().undo_move(move);
        }
        if (candidates.empty()) {
            return std::nullopt;
        }

        int best_category = -1;
        for (const Candidate& candidate : candidates) {
            best_category = std::max(best_category, candidate.effective_wdl);
        }
        if (best_category < 0) {
            return std::nullopt;
        }

        RootTablebaseFilter filter;
        filter.candidate_count = static_cast<int>(candidates.size());
        filter.category = best_category > 0 ? "win" : "draw";
        for (const Candidate& candidate : candidates) {
            filter.wdl_by_move[candidate.move_raw] = candidate.effective_wdl;
            if (candidate.effective_wdl == best_category) {
                filter.allowed_moves.insert(candidate.move_raw);
            }
        }
        return filter;
    }

    std::optional<Move> choose_tablebase_root_move(const Clock::time_point& start) {
        if (!root_allowed_moves_.empty()) {
            return std::nullopt;
        }
        auto root_tb = probe_tablebase(root_state_.pos(), root_state_.piece_count(), true);
        if (!root_tb) {
            return std::nullopt;
        }

        struct Candidate {
            Move move = Move::none();
            int  wdl = -99;
            bool has_dtz = false;
            int  dtz = 0;
            int  trap = 0;
        };

        std::vector<Candidate> candidates;
        for (const Move move : MoveList<LEGAL>(root_state_.pos())) {
            bool capture = root_state_.pos().capture(move);
            StateInfo st;
            root_state_.pos().do_move(move, st);
            int next_piece_count = root_state_.piece_count() - (capture ? 1 : 0);
            auto child_tb = probe_tablebase(root_state_.pos(), next_piece_count, true);
            int effective_wdl = child_tb ? effective_root_wdl_after_move(*child_tb, root_state_.pos()) : 0;
            bool claims_draw = root_state_.pos().is_draw(1)
                            || insufficient_material(root_state_.pos())
                            || st.repetition != 0;
            root_state_.pos().undo_move(move);
            if (!child_tb) {
                continue;
            }

            Candidate candidate;
            candidate.move = move;
            candidate.wdl = claims_draw ? 0 : effective_wdl;
            candidate.has_dtz = child_tb->has_dtz;
            candidate.dtz = child_tb->has_dtz ? std::abs(child_tb->dtz) : 1000000;
            candidates.push_back(candidate);
        }
        if (candidates.empty()) {
            return std::nullopt;
        }

        int best_wdl = std::max_element(
                         candidates.begin(),
                         candidates.end(),
                         [](const Candidate& a, const Candidate& b) { return a.wdl < b.wdl; })
                         ->wdl;

        for (auto& candidate : candidates) {
            if (candidate.wdl == best_wdl && best_wdl <= 0) {
                candidate.trap = tablebase_trap_score(candidate.move, best_wdl);
            }
        }

        auto better = [&](const Candidate& a, const Candidate& b) {
            if (a.wdl != b.wdl) {
                return a.wdl > b.wdl;
            }
            if (best_wdl > 0) {
                if (a.has_dtz != b.has_dtz) return a.has_dtz;
                if (a.dtz != b.dtz) return a.dtz < b.dtz;
                return move_to_uci(root_state_.pos(), a.move) < move_to_uci(root_state_.pos(), b.move);
            }
            if (a.trap != b.trap) {
                return a.trap > b.trap;
            }
            if (best_wdl < 0) {
                if (a.has_dtz != b.has_dtz) return a.has_dtz;
                if (a.dtz != b.dtz) return a.dtz > b.dtz;
            }
            return move_to_uci(root_state_.pos(), a.move) < move_to_uci(root_state_.pos(), b.move);
        };

        const Candidate& best = *std::max_element(
          candidates.begin(),
          candidates.end(),
          [&](const Candidate& a, const Candidate& b) { return better(b, a); });

        last_stats_.sims = 0;
        last_stats_.seeded_sims = 0;
        last_stats_.elapsed = elapsed_seconds(start);
        last_stats_.best_move = move_to_uci(root_state_.pos(), best.move);
        last_stats_.best_visits = 0;
        last_stats_.best_q = static_cast<float>(best.wdl > 0 ? 1.0 : best.wdl < 0 ? -1.0 : 0.0);
        last_stats_.avg_depth = 0.0;
        last_stats_.max_depth = 0;
        last_stats_.root_children = static_cast<int>(candidates.size());
        last_stats_.leaf_batches = 0;
        last_stats_.cache_hits = tb_hits_;
        last_stats_.cache_misses = tb_misses_;
        last_stats_.cache_size = static_cast<int>(tb_cache_.size());

        std::cout << "[syzygy] root bestmove=" << last_stats_.best_move
                  << " wdl=" << best.wdl
                  << " dtz=" << (best.has_dtz ? best.dtz : 0)
                  << " trap=" << best.trap
                  << " candidates=" << candidates.size()
                  << " hits=" << tb_hits_
                  << " misses=" << tb_misses_
                  << " skipped_piece=" << tb_skipped_by_piece_count_
                  << " skipped_material=" << tb_skipped_by_material_
                  << std::endl;

        persistent_target_q = last_stats_.best_q;
        return best.move;
    }

    std::optional<Move> choose_immediate_mate_root(const Clock::time_point& start) {
        if (!root_allowed_moves_.empty()) {
            return std::nullopt;
        }
        for (const Move move : MoveList<LEGAL>(root_state_.pos())) {
            StateInfo st;
            root_state_.pos().do_move(move, st);
            bool is_mate = root_state_.pos().checkers() && MoveList<LEGAL>(root_state_.pos()).size() == 0;
            root_state_.pos().undo_move(move);
            if (!is_mate) {
                continue;
            }

            last_stats_ = SearchStats{};
            last_stats_.elapsed = elapsed_seconds(start);
            last_stats_.best_move = move_to_uci(root_state_.pos(), move);
            last_stats_.best_q = 1.0f;
            last_stats_.root_children = 1;
            persistent_target_q = last_stats_.best_q;
            std::cout << "[mate] root bestmove=" << last_stats_.best_move << std::endl;
            return move;
        }
        return std::nullopt;
    }

    bool root_move_claims_draw(Move move) {
        StateInfo st;
        root_state_.pos().do_move(move, st);
        // Explicitly check for threefold or twofold repetition (st.repetition != 0)
        // to prevent shuffling and avoid repetitions in winning endgames.
        bool draws = root_state_.pos().is_draw(1)
                  || insufficient_material(root_state_.pos())
                  || st.repetition != 0;
        root_state_.pos().undo_move(move);
        return draws;
    }

    TorchModel&                             model_;
    TorchModel*                             value_model_ = nullptr;
    HarnessConfig                           config_;
    GameState                               root_state_;
    std::vector<SearchNode>                 nodes_;
    int                                     root_index_ = 0;
    std::unordered_map<Key, EvalCacheSlot>    eval_cache_;
    std::deque<EvalCacheTouch>                eval_cache_order_;
    std::uint64_t                             next_eval_cache_stamp_ = 0;
    std::unordered_map<Key, TablebaseCacheSlot> tb_cache_;
    std::deque<TablebaseCacheTouch>             tb_cache_order_;
    std::uint64_t                               next_tb_cache_stamp_ = 0;
    std::unordered_set<std::string>             syzygy_materials_;
    bool                                        syzygy_available_ = false;
    int                                         tb_hits_ = 0;
    int                                         tb_misses_ = 0;
    int                                         tb_skipped_by_piece_count_ = 0;
    int                                         tb_skipped_by_material_ = 0;
    std::unordered_map<Key, OpeningBookEntry> opening_cache_;
    std::size_t                               opening_cache_bytes_ = 0;
    int                                       cache_hits_ = 0;
    int                                       cache_misses_ = 0;
    int                                       cache_flushes_ = 0;
    SearchStats                               last_stats_;
    std::unordered_set<uint16_t>              root_allowed_moves_;
    std::unordered_map<uint16_t, float>       root_prior_overrides_;
};

class StockfishProcess {
  public:
    explicit StockfishProcess(const std::string& exe_path) {
        SECURITY_ATTRIBUTES sa{};
        sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
        sa.bInheritHandle       = TRUE;
        sa.lpSecurityDescriptor = nullptr;

        if (!CreatePipe(&stdout_read_, &stdout_write_, &sa, 0)) {
            throw std::runtime_error("CreatePipe stdout failed");
        }
        if (!SetHandleInformation(stdout_read_, HANDLE_FLAG_INHERIT, 0)) {
            throw std::runtime_error("SetHandleInformation stdout failed");
        }
        if (!CreatePipe(&stdin_read_, &stdin_write_, &sa, 0)) {
            throw std::runtime_error("CreatePipe stdin failed");
        }
        if (!SetHandleInformation(stdin_write_, HANDLE_FLAG_INHERIT, 0)) {
            throw std::runtime_error("SetHandleInformation stdin failed");
        }

        STARTUPINFOA si{};
        si.cb         = sizeof(STARTUPINFOA);
        si.dwFlags    = STARTF_USESTDHANDLES;
        si.hStdInput  = stdin_read_;
        si.hStdOutput = stdout_write_;
        si.hStdError  = stdout_write_;

        PROCESS_INFORMATION pi{};
        std::string         cmd = "\"" + exe_path + "\"";
        if (!CreateProcessA(
              nullptr,
              cmd.data(),
              nullptr,
              nullptr,
              TRUE,
              CREATE_NO_WINDOW,
              nullptr,
              nullptr,
              &si,
              &pi)) {
            throw std::runtime_error("CreateProcessA failed for stockfish");
        }

        process_ = pi.hProcess;
        thread_  = pi.hThread;
        CloseHandle(stdin_read_);
        stdin_read_ = nullptr;
        CloseHandle(stdout_write_);
        stdout_write_ = nullptr;
    }

    ~StockfishProcess() {
        try {
            write_line("quit");
        } catch (...) {
        }
        close_handle(stdin_write_);
        close_handle(stdout_read_);
        close_handle(thread_);
        close_handle(process_);
    }

    void initialize(int elo) {
        std::cout << "[stockfish] init" << std::endl;
        write_line("uci");
        read_until_prefix("uciok");
        write_line("setoption name UCI_LimitStrength value true");
        write_line("setoption name UCI_Elo value " + std::to_string(elo));
        write_line("setoption name Threads value 1");
        write_line("setoption name Hash value 64");
        write_line("isready");
        read_until_prefix("readyok");
        std::cout << "[stockfish] ready" << std::endl;
    }

    void new_game() {
        write_line("ucinewgame");
        write_line("isready");
        read_until_prefix("readyok");
    }

    std::string bestmove(const std::vector<std::string>& history_uci, double movetime_sec) {
        std::ostringstream pos_cmd;
        pos_cmd << "position startpos";
        if (!history_uci.empty()) {
            pos_cmd << " moves";
            for (const auto& move : history_uci) {
                pos_cmd << ' ' << move;
            }
        }
        write_line(pos_cmd.str());

        int millis = std::max(1, static_cast<int>(std::round(movetime_sec * 1000.0)));
        write_line("go movetime " + std::to_string(millis));
        std::string line = read_until_prefix("bestmove");

        std::istringstream iss(line);
        std::string        token;
        std::string        best;
        iss >> token >> best;
        return best;
    }

  private:
    static void close_handle(HANDLE handle) {
        if (handle) {
            CloseHandle(handle);
        }
    }

    void write_line(const std::string& line) {
        std::string payload = line + "\n";
        DWORD       written = 0;
        if (!WriteFile(stdin_write_, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr)) {
            throw std::runtime_error("WriteFile to stockfish failed");
        }
    }

    std::string read_line() {
        std::string line;
        char        ch = 0;
        DWORD       read = 0;
        while (true) {
            if (!ReadFile(stdout_read_, &ch, 1, &read, nullptr) || read == 0) {
                throw std::runtime_error("ReadFile from stockfish failed");
            }
            if (ch == '\r') {
                continue;
            }
            if (ch == '\n') {
                return line;
            }
            line.push_back(ch);
        }
    }

    std::string read_until_prefix(const std::string& prefix) {
        while (true) {
            std::string line = read_line();
            if (line.rfind(prefix, 0) == 0) {
                return line;
            }
        }
    }

    HANDLE stdin_read_   = nullptr;
    HANDLE stdin_write_  = nullptr;
    HANDLE stdout_read_  = nullptr;
    HANDLE stdout_write_ = nullptr;
    HANDLE process_      = nullptr;
    HANDLE thread_       = nullptr;
};

std::string find_stockfish() {
    if (const char* env = std::getenv("STOCKFISH_PATH")) {
        if (std::filesystem::exists(env)) {
            return env;
        }
    }

    std::filesystem::path root = "engines/stockfish";
    if (!std::filesystem::exists(root)) {
        return {};
    }

    std::vector<std::filesystem::path> candidates;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string lower = entry.path().filename().string();
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (entry.path().extension() == ".exe" && lower.find("stockfish") != std::string::npos) {
            candidates.push_back(entry.path());
        }
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.empty() ? std::string{} : candidates.front().string();
}

}

struct GameResult {
    std::string result;
    int         winner = 99;
};

GameResult current_game_result(const Position& pos) {
    if (is_draw_claim(pos)) {
        return {"1/2-1/2", 0};
    }
    auto legal = unique_legal_moves(pos);
    if (!legal.moves.empty()) {
        return {"*", 99};
    }
    if (pos.checkers()) {
        return pos.side_to_move() == WHITE ? GameResult{"0-1", -1} : GameResult{"1-0", 1};
    }
    return {"1/2-1/2", 0};
}

void apply_opening(GameState& board, const std::vector<std::string>& uci_moves) {
    for (const auto& uci : uci_moves) {
        Move move = parse_uci_move(board.pos(), uci);
        if (move == Move::none()) {
            break;
        }
        board.push(move);
    }
}

void print_our_stats(const SearchStats& stats) {
    int nps = static_cast<int>((stats.sims + stats.seeded_sims) / std::max(stats.elapsed, 1e-6));
    std::cout << "[ours] move=" << stats.best_move << " sims=" << stats.sims
              << " seeded=" << stats.seeded_sims << " nps=" << nps
              << " depth=" << std::fixed << std::setprecision(1) << stats.avg_depth << "/"
              << stats.max_depth << " q=" << std::showpos << std::fixed << std::setprecision(3)
              << stats.best_q << std::noshowpos << " visits=" << stats.best_visits
              << " batches=" << stats.leaf_batches
              << " cache=" << stats.cache_hits << "/" << (stats.cache_hits + stats.cache_misses)
              << " dedup=" << stats.deduped << std::endl;
}

Move choose_greedy(TorchModel& model, const Position& pos) {
    EncodedPosition enc;
    encode_position(pos, enc);
    auto [logits, _values] = model.evaluate_single(enc);
    auto legal = unique_legal_moves(pos);
    if (legal.moves.empty()) {
        return Move::none();
    }
    float* logits_ptr = logits.data_ptr<float>();
    int    best = 0;
    for (std::size_t i = 1; i < legal.tokens.size(); ++i) {
        if (logits_ptr[legal.tokens[i]] > logits_ptr[legal.tokens[best]]) {
            best = static_cast<int>(i);
        }
    }
    return Move(legal.moves[best]);
}

GameResult play_game(
  PUCTPlayer& player,
  TorchModel& model,
  StockfishProcess* stockfish,
  const std::vector<std::string>& opening,
  bool our_white,
  const HarnessConfig& config) {
    GameState board(2048);
    apply_opening(board, opening);
    player.set_position_from_history(board.history_raw());

    if (stockfish) {
        stockfish->new_game();
    }

    std::cout << "[game] opening=";
    for (std::size_t i = 0; i < opening.size(); ++i) {
        if (i) {
            std::cout << ' ';
        }
        std::cout << opening[i];
    }
    std::cout << " our_white=" << (our_white ? "true" : "false") << std::endl;

    while (true) {
        GameResult status = current_game_result(board.pos());
        if (status.winner != 99) {
            std::cout << "[game] result=" << status.result << " plies=" << board.ply() << std::endl;
            return status;
        }
        if (board.ply() > config.max_plies) {
            std::cout << "[game] result=ABORT plies=" << board.ply() << std::endl;
            return {"ABORT", 99};
        }

        Move        move = Move::none();
        std::string actor;
        if ((board.pos().side_to_move() == WHITE) == our_white) {
            auto chosen = player.choose_move();
            if (!chosen.has_value()) {
                return current_game_result(board.pos());
            }
            move = *chosen;
            actor = "ours";
        } else if (stockfish) {
            std::string best = stockfish->bestmove(board.history_uci(), config.stockfish_time);
            move = parse_uci_move(board.pos(), best);
            actor = "stockfish";
        } else {
            move = choose_greedy(model, board.pos());
            actor = "greedy";
        }

        if (move == Move::none()) {
            return current_game_result(board.pos());
        }

        std::string uci = move_to_uci(board.pos(), move);
        board.push(move);
        player.advance_root(move);

        std::cout << "[ply " << std::setw(3) << std::setfill('0') << board.ply() << std::setfill(' ')
                  << "/" << config.max_plies << "] " << std::left << std::setw(10) << actor << std::right
                  << uci << std::endl;
        if (actor == "ours") {
            print_our_stats(player.last_stats());
        }
    }
}

double elo_from_score(double score) {
    score = std::min(std::max(score, 1e-6), 1.0 - 1e-6);
    return 400.0 * std::log10(score / (1.0 - score));
}

HarnessConfig parse_args(int argc, char** argv) {
    HarnessConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--model") cfg.model_path = next("--model");
        else if (arg == "--value-model") cfg.value_model_path = next("--value-model");
        else if (arg == "--stockfish-path") cfg.stockfish_path = next("--stockfish-path");
        else if (arg == "--normal-time") cfg.normal_time = std::stod(next("--normal-time"));
        else if (arg == "--panic-time") cfg.panic_time = std::stod(next("--panic-time"));
        else if (arg == "--stockfish-time") cfg.stockfish_time = std::stod(next("--stockfish-time"));
        else if (arg == "--stockfish-elo") cfg.stockfish_elo = std::stoi(next("--stockfish-elo"));
        else if (arg == "--games") cfg.games = std::stoi(next("--games"));
        else if (arg == "--max-plies") cfg.max_plies = std::stoi(next("--max-plies"));
        else if (arg == "--cpuct") cfg.cpuct = std::stod(next("--cpuct"));
        else if (arg == "--cpuct-init") cfg.cpuct_init = std::stod(next("--cpuct-init"));
        else if (arg == "--cpuct-scale") cfg.cpuct_scale = std::stod(next("--cpuct-scale"));
        else if (arg == "--virtual-loss") cfg.virtual_loss = std::stod(next("--virtual-loss"));
        else if (arg == "--eval-batch-size") cfg.eval_batch_size = std::stoi(next("--eval-batch-size"));
        else if (arg == "--progress-interval") cfg.progress_interval = std::stod(next("--progress-interval"));
        else if (arg == "--cache-capacity") cfg.cache_capacity = std::stoi(next("--cache-capacity"));
        else if (arg == "--collect-dup-limit") cfg.collect_dup_limit = std::stoi(next("--collect-dup-limit"));
        else if (arg == "--min-sims") cfg.min_sims = std::stoi(next("--min-sims"));
        else if (arg == "--max-sims") cfg.max_sims = std::stoi(next("--max-sims"));
        else if (arg == "--opening-index") cfg.opening_index = std::stoi(next("--opening-index"));
        else if (arg == "--opening-cache-mb") cfg.opening_cache_mb = std::stoi(next("--opening-cache-mb"));
        else if (arg == "--opening-cache-max-entries") cfg.opening_cache_max_entries = std::stoi(next("--opening-cache-max-entries"));
        else if (arg == "--opening-cache-file") cfg.opening_cache_file = next("--opening-cache-file");
        else if (arg == "--opening-cache-max-ply") cfg.opening_cache_max_ply = std::stoi(next("--opening-cache-max-ply"));
        else if (arg == "--opening-cache-full-ply") cfg.opening_cache_full_ply = std::stoi(next("--opening-cache-full-ply"));
        else if (arg == "--opening-cache-branching") cfg.opening_cache_branching = std::stoi(next("--opening-cache-branching"));
        else if (arg == "--opening-cache-max-seconds") cfg.opening_cache_max_seconds = std::stod(next("--opening-cache-max-seconds"));
        else if (arg == "--opening-book-sims") cfg.opening_book_sims = std::stoi(next("--opening-book-sims"));
        else if (arg == "--opening-branch-visit-cap") cfg.opening_branch_visit_cap = std::stoi(next("--opening-branch-visit-cap"));
        else if (arg == "--opening-cache-build-only") cfg.opening_cache_build_only = true;
        else if (arg == "--opening-graph-file") cfg.opening_graph_file = next("--opening-graph-file");
        else if (arg == "--opening-graph-sims") cfg.opening_graph_sims = std::stoi(next("--opening-graph-sims"));
        else if (arg == "--opening-graph-max-ply") cfg.opening_graph_max_ply = std::stoi(next("--opening-graph-max-ply"));
        else if (arg == "--opening-graph-max-seconds") cfg.opening_graph_max_seconds = std::stod(next("--opening-graph-max-seconds"));
        else if (arg == "--opening-graph-load-min-visits") cfg.opening_graph_load_min_visits = std::stoi(next("--opening-graph-load-min-visits"));
        else if (arg == "--opening-graph-load-max-children") cfg.opening_graph_load_max_children = std::stoi(next("--opening-graph-load-max-children"));
        else if (arg == "--opening-graph-load-max-nodes") cfg.opening_graph_load_max_nodes = std::stoi(next("--opening-graph-load-max-nodes"));
        else if (arg == "--opening-graph-build-only") cfg.opening_graph_build_only = true;
        else if (arg == "--opening-cache-all-white-roots") cfg.opening_cache_all_white_roots = true;
        else if (arg == "--dynamic-cpuct") cfg.dynamic_cpuct = true;
        else if (arg == "--fixed-cpuct") cfg.dynamic_cpuct = false;
        else if (arg == "--seed") cfg.seed = std::stoi(next("--seed"));
        else if (arg == "--verbose-search") cfg.verbose_search = true;
        else if (arg == "--uci") cfg.uci = true;
        else if (arg == "--fp32") cfg.fp32 = true;
        else if (arg == "--value-fp32") cfg.value_fp32 = true;
        else if (arg == "--checkpoint-sims") {
            cfg.checkpoint_sims.clear();
            std::stringstream ss(next("--checkpoint-sims"));
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    cfg.checkpoint_sims.push_back(std::stoi(item));
                }
            }
        }
        else if (arg == "--checkpoint-topn") cfg.checkpoint_topn = std::stoi(next("--checkpoint-topn"));
        else if (arg == "--root-probe-children") cfg.root_probe_children = std::stoi(next("--root-probe-children"));
        else if (arg == "--root-probe-visits-per-child") cfg.root_probe_visits_per_child = std::stoi(next("--root-probe-visits-per-child"));
        else if (arg == "--cpuct-warmup-visits") cfg.cpuct_warmup_visits = std::stoi(next("--cpuct-warmup-visits"));
        else if (arg == "--cpuct-warmup-value") cfg.cpuct_warmup_value = std::stod(next("--cpuct-warmup-value"));
        else if (arg == "--syzygy-enable") cfg.syzygy_enable = true;
        else if (arg == "--no-syzygy") cfg.syzygy_enable = false;
        else if (arg == "--syzygy-path") {
            cfg.syzygy_path = next("--syzygy-path");
            cfg.syzygy_enable = true;
        }
        else if (arg == "--syzygy-max-pieces") cfg.syzygy_max_pieces = std::stoi(next("--syzygy-max-pieces"));
        else if (arg == "--syzygy-cache-capacity") cfg.syzygy_cache_capacity = std::stoi(next("--syzygy-cache-capacity"));
        else if (arg == "--no-tree-reuse") cfg.reuse_tree = false;
        else if (arg == "--reuse-tree") cfg.reuse_tree = true;
        else throw std::runtime_error("unknown argument: " + arg);
    }
    // Auto-enable logic removed to respect explicit configuration.
    // if (!cfg.syzygy_enable && cfg.syzygy_path.empty() && std::filesystem::exists("chesstransformer/syzygy")) {
    //     cfg.syzygy_path = "chesstransformer/syzygy";
    //     cfg.syzygy_enable = true;
    // }
    return cfg;
}

void uci_loop(PUCTPlayer& player, const HarnessConfig& config) {
    std::string line, token;
    GameState   board(4096);

    std::cout << "id name SSChess Native" << std::endl;
    std::cout << "id author Antigravity" << std::endl;
    std::cout << "uciok" << std::endl;

    while (std::getline(std::cin, line)) {
        std::cerr << "[uci] got: " << line << std::endl;
        std::istringstream ss(line);
        token.clear();
        ss >> token;

        if (token == "uci") {
            std::cout << "id name SSChess Native" << std::endl;
            std::cout << "id author Antigravity" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "ucinewgame") {
            board.reset();
            player.set_position_from_history(board.history_raw());
        } else if (token == "position") {
            std::string sub;
            if (!(ss >> sub)) continue;
            bool from_fen = false;
            if (sub == "startpos") {
                board.reset();
                if (ss >> sub && sub == "moves") {
                    while (ss >> sub) {
                        Move m = parse_uci_move(board.pos(), sub);
                        if (m != Move::none()) board.push(m, true);
                    }
                }
            } else if (sub == "fen") {
                std::string fen;
                while (ss >> sub && sub != "moves") {
                    if (!fen.empty()) fen += " ";
                    fen += sub;
                }
                board.set_fen(fen);
                from_fen = true;
                if (sub == "moves") {
                    while (ss >> sub) {
                        Move m = parse_uci_move(board.pos(), sub);
                        if (m != Move::none()) board.push(m, true);
                    }
                }
            }
            if (from_fen) {
                player.set_position_from_fen(board.pos().fen());
            } else {
                player.set_position_from_history(board.history_raw());
            }
            std::cerr << "[uci] position set, fen=" << board.pos().fen() << std::endl;
        } else if (token == "go") {
            std::vector<uint16_t> allowed_moves;
            std::vector<std::pair<uint16_t, float>> root_priors;
            std::string sub;
            while (ss >> sub) {
                if (sub == "searchmoves") {
                    while (ss >> sub) {
                        if (sub == "rootpriors") {
                            break;
                        }
                        Move m = parse_uci_move(board.pos(), sub);
                        if (m != Move::none()) {
                            allowed_moves.push_back(m.raw());
                        }
                    }
                    if (sub != "rootpriors") {
                        break;
                    }
                }
                if (sub == "rootpriors") {
                    std::string move_uci;
                    std::string prior_text;
                    while (ss >> move_uci >> prior_text) {
                        Move m = parse_uci_move(board.pos(), move_uci);
                        if (m != Move::none()) {
                            root_priors.push_back({m.raw(), std::stof(prior_text)});
                        }
                    }
                    break;
                }
            }
            player.set_root_allowed_moves(allowed_moves);
            player.set_root_prior_overrides(root_priors);
            auto chosen = player.choose_move();
            player.clear_root_allowed_moves();
            player.clear_root_prior_overrides();
            if (chosen.has_value()) {
                std::cout << "bestmove " << move_to_uci(board.pos(), *chosen) << std::endl;
            } else {
                std::cout << "bestmove (none)" << std::endl;
            }
        } else if (token == "quit") {
            break;
        }
    }
}

int main(int argc, char** argv) {
    try {
        Bitboards::init();
        Position::init();

        HarnessConfig cfg = parse_args(argc, argv);

        TorchModel       model(cfg.model_path, cfg.fp32, cfg.eval_batch_size);
        std::unique_ptr<TorchModel> value_model;
        if (!cfg.value_model_path.empty()) {
            value_model = std::make_unique<TorchModel>(cfg.value_model_path, cfg.value_fp32, cfg.eval_batch_size);
        }
        PUCTPlayer       player(model, cfg, value_model.get());

        if (cfg.opening_cache_build_only) {
            std::cout << "[opening-book] build-only complete entries=" << player.opening_cache_size()
                      << " size_mb=" << std::fixed << std::setprecision(1)
                      << (static_cast<double>(player.opening_cache_bytes()) / (1024.0 * 1024.0))
                      << std::endl;
            return 0;
        }

        if (cfg.opening_cache_all_white_roots) {
            int graph_sims = cfg.opening_graph_sims > 0 ? cfg.opening_graph_sims : 10000;
            auto summary = player.build_opening_cache_all_white_roots(
              cfg.opening_cache_file.empty() ? "ChessTransformer/engines/opening_cache_startpos_all_roots.scb" : cfg.opening_cache_file,
              graph_sims,
              cfg.opening_graph_max_ply,
              cfg.opening_graph_max_seconds);
            std::cout << "[opening-cache] all-white-roots complete entries=" << summary.nodes
                      << " max_live_nodes=" << summary.edges
                      << " sims=" << summary.sims
                      << " max_ply=" << summary.max_ply << std::endl;
            return 0;
        }

        if (cfg.opening_graph_build_only) {
            int graph_sims = cfg.opening_graph_sims > 0 ? cfg.opening_graph_sims : 100000;
            if (cfg.opening_graph_max_seconds > 0.0 && cfg.opening_graph_sims <= 0) {
                graph_sims = std::numeric_limits<int>::max();
            }
            auto summary = player.build_opening_graph_file(
              cfg.opening_graph_file.empty() ? "ChessTransformer/engines/opening_graph_startpos.scg" : cfg.opening_graph_file,
              graph_sims,
              cfg.opening_graph_max_ply,
              cfg.opening_graph_max_seconds);
            std::cout << "[opening-graph] build-only complete nodes=" << summary.nodes
                      << " edges=" << summary.edges
                      << " sims=" << summary.sims
                      << " max_ply=" << summary.max_ply << std::endl;
            return 0;
        }

        if (cfg.uci) {
            uci_loop(player, cfg);
            return 0;
        }

        if (cfg.stockfish_path.empty()) {
            cfg.stockfish_path = find_stockfish();
        }
        if (cfg.stockfish_path.empty()) {
            throw std::runtime_error("stockfish executable not found");
        }

        StockfishProcess stockfish(cfg.stockfish_path);
        stockfish.initialize(cfg.stockfish_elo);

        auto                   book = openings();
        std::mt19937           rng(cfg.seed);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(book.size()) - 1);

        int wins = 0;
        int draws = 0;
        int losses = 0;
        int discarded = 0;

        for (int completed = 0, attempt = 0; completed < cfg.games; ++attempt) {
            const auto& opening =
              cfg.opening_index >= 0 ? book[cfg.opening_index % static_cast<int>(book.size())]
                                     : book[dist(rng)];
            bool our_white = (completed % 2 == 0);

            std::cout << "\n=== Game " << (completed + 1) << "/" << cfg.games << " (attempt "
                      << (attempt + 1) << ", normal_time=" << cfg.normal_time << "s)" << std::endl;

            GameResult result = play_game(player, model, &stockfish, opening, our_white, cfg);
            if (result.result == "ABORT") {
                ++discarded;
                continue;
            }

            if (result.result == "1-0") {
                if (our_white) ++wins;
                else ++losses;
            } else if (result.result == "0-1") {
                if (our_white) ++losses;
                else ++wins;
            } else {
                ++draws;
            }

            ++completed;
            double score = (wins + 0.5 * draws) / std::max(1, completed);
            std::cout << "[score] W/D/L=" << wins << "/" << draws << "/" << losses
                      << " score=" << std::fixed << std::setprecision(3) << score
                      << " discarded=" << discarded << std::endl;
        }

        double score = (wins + 0.5 * draws) / std::max(1, cfg.games);
        double diff  = elo_from_score(score);
        std::cout << "\nElo difference vs Stockfish: " << std::showpos << std::fixed
                  << std::setprecision(1) << diff << std::noshowpos << " | Estimated Elo: "
                  << (cfg.stockfish_elo + diff) << std::endl;
        return 0;
    } catch (const c10::Error& err) {
        std::cerr << "[fatal] torch error: " << err.what() << std::endl;
        return 1;
    } catch (const std::exception& err) {
        std::cerr << "[fatal] " << err.what() << std::endl;
        return 1;
    }
}

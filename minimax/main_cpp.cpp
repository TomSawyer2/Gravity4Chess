#include <iostream>
#include <vector>
#include <ctime>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

using namespace std;

#define PLAYER 'B'  // ��� (����)
#define AI 'W'      // AI (����)

#define ROWS 5
#define COLS 5
#define MAX_STACK 5  // ÿ�����ɶѵ� 5 ��
#define MAX_DEPTH 7

// תȦ������
vector<string> spinner = {"|", "/", "-", "\\"};

// ��������״̬��ϣ
unordered_map<string, tuple<int, int, int, int, pair<int, int>>> transposition_table;

int nodes_explored = 0;       // ���������ѷ��ʵĽڵ���
double search_start_time = 0; // ����������ʼʱ��

struct Board {
    vector<vector<vector<char>>> grid;

    Board() {
        grid = vector<vector<vector<char>>>(ROWS, vector<vector<char>>(COLS));
    }

    bool is_valid_move(int row, int col) {
        return row >= 0 && row < ROWS && col >= 0 && col < COLS && grid[row][col].size() < MAX_STACK;
    }

    void make_move(int row, int col, char piece) {
        grid[row][col].push_back(piece);
    }

    bool is_board_full() {
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (grid[r][c].size() < MAX_STACK) {
                    return false;
                }
            }
        }
        return true;
    }

    string board_to_str() {
        stringstream ss;
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                for (int l = 0; l < grid[r][c].size(); ++l) {
                    ss << grid[r][c][l];
                }
                ss << "|";
            }
            ss << "#";
        }
        return ss.str();
    }

    string hash_board() {
        return board_to_str(); // �� C++ ��ֱ��ʹ�� board_str ��Ϊ��ϣ��
    }

    bool check_winner() {
        const vector<pair<int, int>> directions = {
            {0, 1}, {1, 0}, {1, 1}, {1, -1}, {0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1},
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
        };

        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                vector<char> stack = grid[r][c];
                for (int l = 0; l < stack.size(); ++l) {
                    for (auto& [dr, dc] : directions) {
                        int rr = r, cc = c, ll = l;
                        int chain_count = 1;
                        for (int i = 0; i < 3; ++i) {
                            rr += dr;
                            cc += dc;
                            ll += 0;
                            if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && ll < grid[rr][cc].size()) {
                                if (grid[rr][cc][ll] == stack[l]) {
                                    chain_count++;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        if (chain_count >= 4) {
                            return true; // �ҵ�Ӯ��
                        }
                    }
                }
            }
        }
        return false;
    }
};

// ����������ģʽƥ���
unordered_map<string, int> black_pattern_score = {
    {"BBBB", 9999999}, {"BBB-", 5000}, {"BB-B", 5000}, {"B-BB", 5000}, {"-BBB", 5000},
    {"BB--", 200}, {"B-B-", 150}, {"B--B", 100}, {"-BB-", 200}, {"B---", 10}, {"-B--", 10}, {"--B-", 10}, {"---B", 10}, {"----", 0}
};

unordered_map<string, int> white_pattern_score = {
    {"WWWW", 9999999}, {"WWW-", 5000}, {"WW-W", 5000}, {"W-WW", 5000}, {"-WWW", 5000},
    {"WW--", 200}, {"W-W-", 150}, {"W--W", 100}, {"-WW-", 200}, {"W---", 10}, {"-W--", 10}, {"--W-", 10}, {"---W", 10}, {"----", 0}
};

// ��ȡ�����߶Σ������� 4 ���ߣ�
vector<vector<pair<int, int>>> get_all_lines(Board& board) {
    const vector<pair<int, int>> directions = {
        {0, 1}, {1, 0}, {1, 1}, {1, -1}, {0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };

    vector<vector<pair<int, int>>> lines;

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            for (int l = 0; l < board.grid[r][c].size(); ++l) {
                for (auto& [dr, dc] : directions) {
                    vector<pair<int, int>> coords = {{r, c}};
                    int rr = r, cc = c, ll = l;
                    for (int i = 0; i < 3; ++i) {
                        rr += dr;
                        cc += dc;
                        if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS) {
                            coords.push_back({rr, cc});
                        } else {
                            break;
                        }
                    }
                    if (coords.size() == 4) {
                        lines.push_back(coords);
                    }
                }
            }
        }
    }
    return lines;
}

int evaluate(Board& board) {
    if (board.check_winner()) {
        return AI == 'W' ? 99999999 : -99999999; // AI ʤ��
    }

    int ai_potential = 0;
    int player_potential = 0;

    vector<vector<pair<int, int>>> lines = get_all_lines(board);

    for (auto& line : lines) {
        vector<char> pieces;
        for (auto& [r, c] : line) {
            pieces.push_back(board.grid[r][c].empty() ? '-' : board.grid[r][c].back());
        }

        string pattern;
        for (auto& p : pieces) {
            if (p == 'W') {
                pattern += 'W';
            } else if (p == 'B') {
                pattern += 'B';
            } else {
                pattern += '-';
            }
        }

        if (pattern.find('B') != string::npos && pattern.find('W') == string::npos) {
            ai_potential += black_pattern_score[pattern];
        } else if (pattern.find('W') != string::npos && pattern.find('B') == string::npos) {
            player_potential += white_pattern_score[pattern];
        }
    }

    return ai_potential - player_potential;
}

// ��ȡ���Ų����� minimax
pair<int, pair<int, int>> minimax(Board& board, int depth, int alpha, int beta, bool maximizing) {
    nodes_explored++;

    // �û���
    string bkey = board.hash_board();
    if (transposition_table.find(bkey) != transposition_table.end()) {
        auto [stored_depth, stored_alpha, stored_beta, stored_val, stored_move] = transposition_table[bkey];
        if (stored_depth >= depth && stored_alpha <= alpha && stored_beta >= beta) {
            return {stored_val, stored_move};
        }
    }

    int score = evaluate(board);
    if (abs(score) >= 99999999 || depth == 0 || board.is_board_full()) {
        return {score, {-1, -1}};
    }

    if (maximizing) {
        int best_val = -INT_MAX;
        pair<int, int> best_move = {-1, -1};

        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (board.is_valid_move(r, c)) {
                    board.make_move(r, c, AI);
                    auto [val, _] = minimax(board, depth - 1, alpha, beta, false);
                    if (val > best_val) {
                        best_val = val;
                        best_move = {r, c};
                    }
                    alpha = max(alpha, best_val);
                    if (beta <= alpha) {
                        break;
                    }
                    board.grid[r][c].pop_back();
                }
            }
        }
        transposition_table[bkey] = {depth, alpha, beta, best_val, best_move};
        return {best_val, best_move};
    } else {
        int best_val = INT_MAX;
        pair<int, int> best_move = {-1, -1};

        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (board.is_valid_move(r, c)) {
                    board.make_move(r, c, PLAYER);
                    auto [val, _] = minimax(board, depth - 1, alpha, beta, true);
                    if (val < best_val) {
                        best_val = val;
                        best_move = {r, c};
                    }
                    beta = min(beta, best_val);
                    if (beta <= alpha) {
                        break;
                    }
                    board.grid[r][c].pop_back();
                }
            }
        }
        transposition_table[bkey] = {depth, alpha, beta, best_val, best_move};
        return {best_val, best_move};
    }
}

void print_board(Board& board) {
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (board.grid[r][c].empty()) {
                cout << "[ ] ";
            } else {
                cout << "[" << string(board.grid[r][c].begin(), board.grid[r][c].end()) << "] ";
            }
        }
        cout << endl;
    }
}

int main() {
    Board board;
    bool ai_first_move_done = false;
    int current_player = PLAYER;

    while (true) {
        print_board(board);
        if (board.check_winner()) {
            cout << "Game Over! Winner: " << (current_player == AI ? "AI" : "Player") << endl;
            break;
        }

        if (board.is_board_full()) {
            cout << "The board is full, it's a draw!" << endl;
            break;
        }

        if (current_player == PLAYER) {
            cout << "Player's turn (enter row and col): ";
            int r, c;
            cin >> r >> c;
            if (!board.is_valid_move(r - 1, c - 1)) {
                cout << "Invalid move! Try again." << endl;
                continue;
            }
            board.make_move(r - 1, c - 1, PLAYER);
            current_player = AI;
        } else {
            cout << "AI is calculating..." << endl;
            pair<int, pair<int, int>> move = minimax(board, MAX_DEPTH, -INT_MAX, INT_MAX, true);
            board.make_move(move.second.first, move.second.second, AI);
            current_player = PLAYER;
        }
    }

    return 0;
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <tuple>          // for std::tuple
#include <unordered_map>  // for Node children
#include <utility>        // for std::pair
#include <vector>

namespace py = pybind11;

// --------------- 常量配置 ---------------
static const int BOARD_ROWS = 5;
static const int BOARD_COLS = 5;
static const int MAX_STACK = 5;
static const int POLICY_SIZE = BOARD_ROWS * BOARD_COLS;

// 用于区分玩家
static const char AI_PLAYER = 'B';
static const char OP_PLAYER = 'W';

// --------------- 棋盘类 Board ---------------
// 5x5堆叠连四, 每个cell用std::string存底->顶
class Board {
 public:
  std::vector<std::vector<std::string>> board;  // [r][c], each = string like "WB"
  bool done;
  char winner;          // 'W'/'B'/' '
  char current_player;  // 'W' or 'B'

  Board() {
    reset();
  }

  void reset() {
    board.clear();
    board.resize(BOARD_ROWS, std::vector<std::string>(BOARD_COLS, ""));
    done = false;
    winner = ' ';
    current_player = 'W';
  }

  Board clone() const {
    Board b;
    b.board = board;
    b.done = done;
    b.winner = winner;
    b.current_player = current_player;
    return b;
  }

  bool step(int action) {
    // 根据 action=(r*BOARD_COLS + c) 落子
    if (done)
      return false;
    int r = action / BOARD_COLS;
    int c = action % BOARD_COLS;
    if (r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS) {
      done = true;
      winner = (current_player == 'W' ? 'B' : 'W');
      return false;
    }
    if ((int)board[r][c].size() >= MAX_STACK) {
      // 非法 => 对手胜
      done = true;
      winner = (current_player == 'W' ? 'B' : 'W');
      return false;
    }
    // 落子
    board[r][c].push_back(current_player);

    // 检查4连
    _check_winner();
    if (!done) {
      // 检查满
      if (_board_full()) {
        done = true;
        winner = ' ';  // 平局
      }
    }
    // 切换
    if (!done) {
      current_player = (current_player == 'W' ? 'B' : 'W');
    }
    return true;
  }

  bool is_done() const {
    return done;
  }
  char get_winner() const {
    return winner;
  }
  char get_current_player() const {
    return current_player;
  }

  std::vector<int> legal_actions() const {
    if (done)
      return {};
    std::vector<int> acts;
    for (int i = 0; i < BOARD_ROWS * BOARD_COLS; i++) {
      int r = i / BOARD_COLS, c = i % BOARD_COLS;
      if ((int)board[r][c].size() < MAX_STACK) {
        acts.push_back(i);
      }
    }
    return acts;
  }

  // 返回 shape=(5,5,2) 的 obs
  std::vector<std::vector<std::vector<float>>> get_obs() const {
    std::vector<std::vector<std::vector<float>>> obs(
        BOARD_ROWS,
        std::vector<std::vector<float>>(BOARD_COLS, std::vector<float>(2, 0.0f)));
    for (int r = 0; r < BOARD_ROWS; r++) {
      for (int c = 0; c < BOARD_COLS; c++) {
        int wCount = 0;
        int bCount = 0;
        for (auto& ch : board[r][c]) {
          if (ch == 'W')
            wCount++;
          else if (ch == 'B')
            bCount++;
        }
        obs[r][c][0] = (float)wCount;
        obs[r][c][1] = (float)bCount;
      }
    }
    return obs;
  }

 private:
  bool _board_full() const {
    for (int r = 0; r < BOARD_ROWS; r++) {
      for (int c = 0; c < BOARD_COLS; c++) {
        if ((int)board[r][c].size() < MAX_STACK) {
          return false;
        }
      }
    }
    return true;
  }

  void _check_winner() {
    // directions
    static int dirs[13][3] = {
        {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, -1, 0}, {0, 0, 1}, {1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, 1}, {1, -1, 1}, {-1, 1, 1}, {-1, -1, 1}};
    for (int r = 0; r < BOARD_ROWS; r++) {
      for (int c = 0; c < BOARD_COLS; c++) {
        int h = (int)board[r][c].size();
        for (int l = 0; l < h; l++) {
          char piece = board[r][c][l];
          for (auto& d : dirs) {
            int dr = d[0], dc = d[1], dl = d[2];
            int rr = r, cc = c, ll = l;
            int chain_count = 1;
            for (int step = 0; step < 3; step++) {
              rr += dr;
              cc += dc;
              ll += dl;
              if (rr >= 0 && rr < BOARD_ROWS && cc >= 0 && cc < BOARD_COLS) {
                if (ll >= 0 && ll < (int)board[rr][cc].size()) {
                  if (board[rr][cc][ll] == piece) {
                    chain_count++;
                  } else
                    break;
                } else
                  break;
              } else
                break;
            }
            if (chain_count >= 4) {
              done = true;
              winner = piece;  // 'W' or 'B'
              return;
            }
          }
        }
      }
    }
  }
};

// ============= Node for MCTS =============
// 仅示例: alphaZero结构 (Q+U).
struct Node {
  Board board;
  Node* parent;
  std::unordered_map<int, Node*> children;
  std::unordered_map<int, double> P;
  std::unordered_map<int, double> Q;
  std::unordered_map<int, int> N;
  double value;
  char player;

  Node(const Board& b, Node* p = nullptr)
      : board(b), parent(p), value(0.0), player(b.get_current_player()) {}
  ~Node() {
    for (auto& kv : children) {
      delete kv.second;
    }
  }
};

static std::mt19937 rng{std::random_device{}()};

// 回调函数签名:  NNInfer(obs) -> (logits(25), value)
using NNInfer = std::function<std::pair<std::vector<double>, double>(const std::vector<std::vector<std::vector<float>>>&)>;

// 计算 UCB
double ucb_score(Node* node, int a, double c_puct) {
  double sumN = 0.0;
  for (auto& nv : node->N)
    sumN += (double)nv.second;
  double Qa = node->Q[a];
  double Pa = node->P[a];
  double Na = (double)node->N[a];
  double U = c_puct * Pa * std::sqrt(sumN + 1e-8) / (1.0 + Na);
  return Qa + U;
}

// MCTS模拟 (selection->expansion->backprop). 用 NN value
void mcts_simulation(Node* root, const NNInfer& nn_infer, int sims = 1, double c_puct = 1.0) {
  for (int i = 0; i < sims; i++) {
    Node* node = root;

    // 1) Selection
    while (true) {
      if (node->board.is_done() || node->children.empty()) {
        break;
      }
      double bestUCB = -1e9;
      int bestA = -1;
      for (auto& ch : node->children) {
        int a = ch.first;
        double u = ucb_score(node, a, c_puct);
        if (u > bestUCB) {
          bestUCB = u;
          bestA = a;
        }
      }
      if (bestA < 0)
        break;
      auto it = node->children.find(bestA);
      if (it == node->children.end())
        break;
      node = it->second;
    }

    // 2) Expansion
    if (!node->board.is_done() && node->children.empty()) {
      auto acts = node->board.legal_actions();

      // 1) 找必杀 (当前玩家一步就赢)
      std::vector<int> forced_actions;
      for (auto a : acts) {
        Board tmp = node->board.clone();
        tmp.step(a);
        if (tmp.is_done() && tmp.get_winner() == node->board.get_current_player()) {
          forced_actions.push_back(a);
        }
      }
      if (!forced_actions.empty()) {
        // 只扩展 forced_actions
        acts = forced_actions;
      } else {
        // 2) 防守对手 (下回合对手能否一步就赢)
        //    如果发现对手下一步必胜, 则也加入 forced_actions
        char cur = node->board.get_current_player();
        char opp = (cur == 'W' ? 'B' : 'W');
        for (auto a : acts) {
          Board tmp = node->board.clone();
          tmp.step(a);
          if (!tmp.is_done()) {
            auto oppActions = tmp.legal_actions();
            for (auto oa : oppActions) {
              Board tmp2 = tmp.clone();
              tmp2.step(oa);
              if (tmp2.is_done() && tmp2.get_winner() == opp) {
                forced_actions.push_back(a);
                break;
              }
            }
          }
        }
        if (!forced_actions.empty()) {
          acts = forced_actions;
        }
      }

      // NN infer
      auto obs = node->board.get_obs();
      auto py_ret = nn_infer(obs);
      // py_ret是 pair<vector<double>, double>
      auto& logits = py_ret.first;
      double val = py_ret.second;
      node->value = val;

      if (!acts.empty()) {
        // softmax(acts)
        double maxLog = -1e9;
        for (auto a : acts) {
          if (logits[a] > maxLog)
            maxLog = logits[a];
        }
        double sumExp = 0.0;
        for (auto a : acts) {
          sumExp += std::exp(logits[a] - maxLog);
        }
        for (auto a : acts) {
          double ex = std::exp(logits[a] - maxLog);
          node->P[a] = ex / (sumExp + 1e-8);
          node->Q[a] = 0.0;
          node->N[a] = 0;
          Board nb = node->board.clone();
          nb.step(a);
          Node* ch = new Node(nb, node);
          node->children[a] = ch;
        }
      }
    }

    // 3) Backprop
    double leaf_value = node->value;
    char leaf_player = node->player;
    Node* temp = node;
    while (temp) {
      // 如果temp->player==leaf_player => +leaf_value, else -leaf_value
      double v = (temp->player == leaf_player) ? leaf_value : -leaf_value;
      if (temp->parent) {
        // 找action
        for (auto& kv : temp->parent->children) {
          if (kv.second == temp) {
            int a = kv.first;
            temp->parent->N[a] += 1;
            double& qRef = temp->parent->Q[a];
            qRef += (v - qRef) / (double)temp->parent->N[a];
            break;
          }
        }
      }
      temp = temp->parent;
    }
  }
}

// run_self_play_once_alphaZero:
// 整场对局: while not done => MCTS => 计算 pi => 采样 => 落子
// 返回一整个Trajectory: list of (obs, pi, z)
std::vector<std::tuple<
    std::vector<std::vector<std::vector<float>>>,  // obs=(5,5,2)
    std::vector<float>,                            // pi=(25,)
    double                                         // z
    >>
run_self_play_once_alphaZero(const NNInfer& nn_infer, int sims = 1600, double c_puct = 1.0, double temp = 1.0) {
  std::vector<std::tuple<
      std::vector<std::vector<std::vector<float>>>,
      std::vector<float>,
      double>>
      trajectory;

  Board env;
  env.reset();

  // 先定义个容器: (obs, pi, player)
  struct StepData {
    std::vector<std::vector<std::vector<float>>> obs;
    std::vector<float> pi;
    char pl;
  };
  std::vector<StepData> partial;

  while (!env.is_done()) {
    // root
    Node* root = new Node(env, nullptr);
    // 先对 root 做一次 expansion
    {
      auto obs = env.get_obs();
      auto py_res = nn_infer(obs);  // (logits, value)
      auto& logits = py_res.first;
      double v = py_res.second;
      root->value = v;
      auto acts = env.legal_actions();
      if (!acts.empty()) {
        double maxL = -1e9;
        for (auto a : acts) {
          if (logits[a] > maxL)
            maxL = logits[a];
        }
        double sumExp = 0.0;
        for (auto a : acts) {
          sumExp += std::exp(logits[a] - maxL);
        }
        for (auto a : acts) {
          double ex = std::exp(logits[a] - maxL);
          root->P[a] = ex / (sumExp + 1e-8);
          root->Q[a] = 0.0;
          root->N[a] = 0;
          Board nb = env.clone();
          nb.step(a);
          Node* ch = new Node(nb, root);
          root->children[a] = ch;
        }
      }
    }

    // MCTS
    mcts_simulation(root, nn_infer, sims, c_puct);

    // compute pi from root->N
    double sumN = 0.0;
    for (auto& kv : root->N) {
      sumN += (double)kv.second;
    }
    std::vector<float> pi(POLICY_SIZE, 0.0f);
    if (std::fabs(temp) < 1e-8) {
      // select argmax
      int bestA = -1;
      int bestCount = -1;
      for (auto& kv : root->N) {
        if (kv.second > bestCount) {
          bestCount = kv.second;
          bestA = kv.first;
        }
      }
      if (bestA >= 0)
        pi[bestA] = 1.0f;
    } else {
      float psum = 0.0f;
      for (auto& kv : root->N) {
        float x = std::pow((float)kv.second, 1.0f / (float)temp);
        pi[kv.first] = x;
        psum += x;
      }
      for (int i = 0; i < POLICY_SIZE; i++) {
        pi[i] /= (psum + 1e-8f);
      }
    }

    partial.push_back({env.get_obs(), pi, env.get_current_player()});

    // 采样
    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float s = 0.0f;
    int chosen = -1;
    for (int i = 0; i < POLICY_SIZE; i++) {
      s += pi[i];
      if (r <= s) {
        chosen = i;
        break;
      }
    }
    if (chosen < 0) {
      chosen = 0;
    }
    env.step(chosen);

    delete root;
  }

  // winner => z
  char w = env.get_winner();
  double z = 0.0;
  if (w == 'B')
    z = 1.0;
  else if (w == 'W')
    z = -1.0;
  else
    z = 0.0;

  // 视角翻转
  for (auto& st : partial) {
    double realz = (st.pl == 'B') ? z : -z;
    trajectory.push_back({st.obs, st.pi, realz});
  }

  return trajectory;
}

// 给定当前Board，以及NN回调，执行一次MCTS并返回“AI要下的action”
int mcts_get_action_once(
    const Board& board,       // 当前棋面
    const NNInfer& nn_infer,  // 推理回调
    int sims,                 // MCTS模拟次数
    double c_puct,            // puct系数
    double temp               // 温度
) {
  // 1) 为当前局面构造一个 root 节点
  Node* root = new Node(board);

  // 2) 先做一次网络前向推理 => 扩展 root 的子节点
  {
    auto obs = board.get_obs();
    auto py_res = nn_infer(obs);  // (logits, value)
    auto& logits = py_res.first;
    double v = py_res.second;
    root->value = v;

    auto acts = board.legal_actions();
    if (!acts.empty()) {
      double maxLog = -1e9;
      for (auto a : acts) {
        if (logits[a] > maxLog) {
          maxLog = logits[a];
        }
      }
      double sumExp = 0.0;
      for (auto a : acts) {
        sumExp += std::exp(logits[a] - maxLog);
      }
      for (auto a : acts) {
        double ex = std::exp(logits[a] - maxLog);
        root->P[a] = ex / (sumExp + 1e-8);
        root->Q[a] = 0.0;
        root->N[a] = 0;
        Board nb = board.clone();
        nb.step(a);
        Node* ch = new Node(nb, root);
        root->children[a] = ch;
      }
    }
  }

  // 3) MCTS模拟
  mcts_simulation(root, nn_infer, sims, c_puct);

  // 4) 根据 root->N 计算 pi，再根据温度挑选动作
  std::vector<float> pi(POLICY_SIZE, 0.0f);
  double sumN = 0.0;
  for (auto& kv : root->N) {
    sumN += kv.second;
  }
  // 温度判断
  if (std::fabs(temp) < 1e-8) {
    // temp=0 => argmax
    int bestA = -1;
    int bestCount = -1;
    for (auto& kv : root->N) {
      if (kv.second > bestCount) {
        bestCount = kv.second;
        bestA = kv.first;
      }
    }
    delete root;  // 释放MCTS树
    return bestA;
  } else {
    // 一般temp>0 => 按 N[a]^(1/temp) 进行随机采样
    float psum = 0.0f;
    for (auto& kv : root->N) {
      float x = std::pow((float)kv.second, 1.0f / (float)temp);
      pi[kv.first] = x;
      psum += x;
    }
    // 归一化
    for (int i = 0; i < POLICY_SIZE; i++) {
      pi[i] /= (psum + 1e-8f);
    }
    // 采样
    static std::mt19937 rng{std::random_device{}()};
    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float s = 0.0f;
    int chosen = -1;
    for (int i = 0; i < POLICY_SIZE; i++) {
      s += pi[i];
      if (r <= s) {
        chosen = i;
        break;
      }
    }
    if (chosen < 0) {
      chosen = 0;  // fallback
    }
    delete root;  // 清理内存
    return chosen;
  }
}

// --------------- PYBIND11 ---------------
PYBIND11_MODULE(mcts_cpp, m) {
  m.doc() = "AlphaZero MCTS for 5x5 stacked Connect4, Win+MSVC fix";

  // Board
  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def("reset", &Board::reset)
      .def("step", &Board::step)
      .def("is_done", &Board::is_done)
      .def("get_winner", &Board::get_winner)
      .def("get_current_player", &Board::get_current_player)
      .def("legal_actions", &Board::legal_actions)
      .def("get_obs", &Board::get_obs);

  // run_self_play_once_alphaZero
  m.def("run_self_play_once_alphaZero", [](py::object nn_infer, int sims, double c_puct, double temp) {
    // 把 py::object 包装成 NNInfer
    NNInfer infer_func = [nn_infer](const std::vector<std::vector<std::vector<float>>>& obs) -> std::pair<std::vector<double>, double> {
      // 调用 Python
      py::object py_ret = nn_infer(obs);
      // py_ret 是个Python tuple/对象, 需cast为 pair< vector<double>, double >
      auto tuple = py_ret.cast<std::pair<py::object, py::object>>();
      // tuple.first => py::object, cast -> vector<double>
      std::vector<double> logits = tuple.first.cast<std::vector<double>>();
      double val = tuple.second.cast<double>();
      return std::make_pair(logits, val);
    };

    auto data = run_self_play_once_alphaZero(infer_func, sims, c_puct, temp);
    return data;  // vector of (obs, pi, z)
  },
        py::arg("nn_infer"), py::arg("sims") = 2000, py::arg("c_puct") = 1.0, py::arg("temp") = 1.0);

  m.def("mcts_get_action_once", [](const Board& board,
                                   py::object nn_infer_py,  // python函数
                                   int sims, double c_puct, double temp) {
              // 将 py::object 包裹为 NNInfer lambda
              NNInfer infer_func = [nn_infer_py](
                  const std::vector<std::vector<std::vector<float>>>& obs
              ) -> std::pair<std::vector<double>, double>
              {
                  py::object py_ret = nn_infer_py(obs);
                  auto tuple = py_ret.cast<std::pair<py::object, py::object>>();
                  std::vector<double> logits = tuple.first.cast<std::vector<double>>();
                  double val = tuple.second.cast<double>();
                  return std::make_pair(logits, val);
              };

              // 调用上面写好的函数
              int action = mcts_get_action_once(board, infer_func, sims, c_puct, temp);
              return action; },
        py::arg("board"), py::arg("nn_infer_py"), py::arg("sims") = 6000, py::arg("c_puct") = 1.0, py::arg("temp") = 1e-9);
}

# AZUL_GUI.py
import tkinter as tk
from tkinter import messagebox
import random
import copy
from collections import defaultdict

# ---------- 配置 ----------
COLORS = ["蓝", "黄", "红", "黑", "白"]
FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]
FACTORIES_FOR_PLAYERS = {2: 5, 3: 7, 4: 9}

def make_wall_template():
    n = len(COLORS)
    template = []
    for r in range(n):
        row = [COLORS[(c + r) % n] for c in range(n)]
        template.append(row)
    return template

WALL_TEMPLATE = make_wall_template()

# 搜索超参数（视机器性能调节）
SEARCH_DEPTH = 5            # 深度（越大越慢）
SAMPLES_PER_CHANCE = 6      # 概率节点采样数
TOP_K_MOVES = 7             # 启发式保留 top-K

# ---------- 玩家类 ----------
class Player:
    def __init__(self, name):
        self.name = name
        self.pattern_lines = [[] for _ in range(5)]
        self.wall = [[None for _ in range(5)] for _ in range(5)]
        self.floor = []
        self.score = 0

    def can_place_in_line(self, color, row):
        """row: 1..5"""
        r = row - 1
        col_for_color = WALL_TEMPLATE[r].index(color)
        if self.wall[r][col_for_color] is not None:
            return False
        if len(self.pattern_lines[r]) > 0 and self.pattern_lines[r][0] != color:
            return False
        return True

    def place_tiles(self, color, count, row):
        """row: 0 表示 floor"""
        if row == 0 or row == "floor":
            self.floor.extend([color] * count)
            return
        r = row - 1
        if not self.can_place_in_line(color, row):
            # 全部进地板
            self.floor.extend([color] * count)
            return
        capacity = r + 1
        space = capacity - len(self.pattern_lines[r])
        to_line = min(space, count)
        to_floor = count - to_line
        if to_line > 0:
            self.pattern_lines[r].extend([color] * to_line)
        if to_floor > 0:
            self.floor.extend([color] * to_floor)

    def move_full_lines_to_wall_and_score(self):
        discarded = []
        for r in range(5):
            cap = r + 1
            if len(self.pattern_lines[r]) == cap:
                color = self.pattern_lines[r][0]
                col = WALL_TEMPLATE[r].index(color)
                self.wall[r][col] = color
                discarded.extend([color] * (cap - 1))
                self.score += self.calc_tile_score(r, col)
                self.pattern_lines[r] = []
        penalty = 0
        for i, _ in enumerate(self.floor):
            if i < len(FLOOR_PENALTIES):
                penalty += FLOOR_PENALTIES[i]
            else:
                penalty += FLOOR_PENALTIES[-1]
        self.score += penalty
        if self.score < 0:
            self.score = 0
        self.floor = []
        return discarded

    def calc_tile_score(self, r, c):
        row_conn = 1
        i = c - 1
        while i >= 0 and self.wall[r][i] is not None:
            row_conn += 1
            i -= 1
        i = c + 1
        while i < 5 and self.wall[r][i] is not None:
            row_conn += 1
            i += 1
        col_conn = 1
        j = r - 1
        while j >= 0 and self.wall[j][c] is not None:
            col_conn += 1
            j -= 1
        j = r + 1
        while j < 5 and self.wall[j][c] is not None:
            col_conn += 1
            j += 1
        if row_conn == 1 and col_conn == 1:
            return 1
        return row_conn + col_conn

    def has_completed_row(self):
        return any(all(cell is not None for cell in self.wall[r]) for r in range(5))

    def endgame_bonus(self):
        bonus = 0
        for r in range(5):
            if all(self.wall[r][c] is not None for c in range(5)):
                bonus += 2
        for c in range(5):
            if all(self.wall[r][c] is not None for r in range(5)):
                bonus += 7
        for color in COLORS:
            count = sum(1 for r in range(5) for c in range(5) if self.wall[r][c] == color)
            if count == 5:
                bonus += 10
        self.score += bonus
        return bonus

# ---------- 游戏主类 ----------
class AzulGame:
    def __init__(self, num_players=2):
        assert num_players in FACTORIES_FOR_PLAYERS
        self.num_players = num_players
        self.num_factories = FACTORIES_FOR_PLAYERS[num_players]
        self.factories = [[] for _ in range(self.num_factories)]
        self.center = []
        self.bag = self.generate_tiles()
        self.discard = []
        self.players = [Player("你"), Player("电脑")]
        self.current_player = 0
        self.first_player_next = 0

    def generate_tiles(self):
        tiles = []
        for c in COLORS:
            tiles.extend([c] * 20)
        random.shuffle(tiles)
        return tiles

    def draw_tile_from_bag(self):
        if not self.bag:
            if self.discard:
                self.bag = self.discard[:]
                self.discard = []
                random.shuffle(self.bag)
            else:
                return None
        return self.bag.pop()

    def refill_factories(self):
        total = 0
        for i in range(self.num_factories):
            self.factories[i] = []
            for _ in range(4):
                t = self.draw_tile_from_bag()
                if t is None:
                    break
                self.factories[i].append(t)
                total += 1
        return total

    def all_factories_and_center_empty(self):
        return all(len(f) == 0 for f in self.factories) and len(self.center) == 0

    def legal_moves_for_player(self, player):
        moves = []
        for i, f in enumerate(self.factories):
            if not f:
                continue
            for color in set(f):
                possible_rows = [r+1 for r in range(5) if player.can_place_in_line(color, r+1)]
                if not possible_rows:
                    moves.append(("F", i, color, 0))
                else:
                    for row in possible_rows:
                        moves.append(("F", i, color, row))
        if self.center:
            colors_in_center = [t for t in set(self.center) if t != "FIRST"]
            for color in colors_in_center:
                possible_rows = [r+1 for r in range(5) if player.can_place_in_line(color, r+1)]
                if not possible_rows:
                    moves.append(("C", None, color, 0))
                else:
                    for row in possible_rows:
                        moves.append(("C", None, color, row))
        return moves

    def apply_move(self, player, move, player_idx):
        """执行 move：('F', idx, color, row) 或 ('C', None, color, row)"""
        src, idx, color, row = move
        if src == "F":
            f = self.factories[idx]
            count = f.count(color)
            leftovers = [t for t in f if t != color]
            self.factories[idx] = []
            if leftovers:
                self.center.extend(leftovers)
            player.place_tiles(color, count, row)
        else:  # 从 center
            count = self.center.count(color)
            # remove chosen color
            self.center = [t for t in self.center if t != color]
            # FIRST token: if present in center, give to player and set next first
            if "FIRST" in self.center:
                self.center.remove("FIRST")
                player.floor.insert(0, "FIRST")
                self.first_player_next = player_idx
            player.place_tiles(color, count, row)

    def end_of_round_settlement(self):
        discarded = []
        for p in self.players:
            discarded.extend(p.move_full_lines_to_wall_and_score())
        return discarded

    # ---------- 启发式评估 ----------
    def heuristic_evaluate(self, state, player_idx):
        me = state.players[player_idx]
        opp = state.players[1 - player_idx]
        score_diff = me.score - opp.score

        my_row_pot = 0
        opp_row_pot = 0
        for r in range(5):
            need_me = (r+1) - len(me.pattern_lines[r])
            need_op = (r+1) - len(opp.pattern_lines[r])
            my_row_pot += max(0, 5 - need_me)
            opp_row_pot += max(0, 5 - need_op)

        my_color_cnt = defaultdict(int)
        opp_color_cnt = defaultdict(int)
        for r in range(5):
            for c in range(5):
                if me.wall[r][c]:
                    my_color_cnt[me.wall[r][c]] += 1
                if opp.wall[r][c]:
                    opp_color_cnt[opp.wall[r][c]] += 1

        my_color_pot = sum(my_color_cnt.values())
        opp_color_pot = sum(opp_color_cnt.values())

        my_floor = len(me.floor)
        opp_floor = len(opp.floor)

        block_value = 0
        for color in COLORS:
            block_value += my_color_cnt[color] * 0.3

        value = 1.0 * score_diff \
                + 0.5 * (my_row_pot - opp_row_pot) \
                + 0.2 * (my_color_pot - opp_color_pot) \
                - 0.6 * (my_floor - opp_floor) \
                + 0.3 * block_value
        return value

    # ---------------- Expectimax 搜索 ----------------
    def expectimax(self, state, root_player_idx, depth, is_chance_node):
        # 终止：有人完成整行 -> 终局得分差
        if any(p.has_completed_row() for p in state.players):
            sim = copy.deepcopy(state)
            for p in sim.players:
                p.endgame_bonus()
            return sim.players[root_player_idx].score - sim.players[1 - root_player_idx].score

        if depth == 0:
            return self.heuristic_evaluate(state, root_player_idx)

        if not is_chance_node:
            cur = state.current_player
            player = state.players[cur]
            moves = state.legal_moves_for_player(player)
            if not moves:
                state.current_player = (state.current_player + 1) % state.num_players
                return self.expectimax(state, root_player_idx, depth - 1, False)

            # 先启发式排序取 top-K
            scored = []
            for m in moves:
                sim = copy.deepcopy(state)
                sim.apply_move(sim.players[cur], m, cur)
                disc = sim.end_of_round_settlement()
                sim.discard.extend(disc)
                hv = self.heuristic_evaluate(sim, root_player_idx)
                scored.append((hv, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_moves = [m for _, m in scored[:TOP_K_MOVES]]

            if cur == root_player_idx:
                best_v = -1e9
                for m in top_moves:
                    sim = copy.deepcopy(state)
                    sim.apply_move(sim.players[cur], m, cur)
                    disc = sim.end_of_round_settlement()
                    sim.discard.extend(disc)
                    if sim.all_factories_and_center_empty():
                        sim.current_player = sim.first_player_next
                        v = self.expectimax(sim, root_player_idx, depth - 1, True)
                    else:
                        sim.current_player = (sim.current_player + 1) % sim.num_players
                        v = self.expectimax(sim, root_player_idx, depth - 1, False)
                    if v > best_v:
                        best_v = v
                return best_v
            else:
                worst_v = 1e9
                for m in top_moves:
                    sim = copy.deepcopy(state)
                    sim.apply_move(sim.players[cur], m, cur)
                    disc = sim.end_of_round_settlement()
                    sim.discard.extend(disc)
                    if sim.all_factories_and_center_empty():
                        sim.current_player = sim.first_player_next
                        v = self.expectimax(sim, root_player_idx, depth - 1, True)
                    else:
                        sim.current_player = (sim.current_player + 1) % sim.num_players
                        v = self.expectimax(sim, root_player_idx, depth - 1, False)
                    if v < worst_v:
                        worst_v = v
                return worst_v
        else:
            # chance 节点：随机重填 factories（采样）
            total = 0.0
            samples = SAMPLES_PER_CHANCE
            for _ in range(samples):
                sim = copy.deepcopy(state)
                added = sim.refill_factories()
                if added > 0 and "FIRST" not in sim.center:
                    sim.center.insert(0, "FIRST")
                sim.current_player = sim.first_player_next
                total += self.expectimax(sim, root_player_idx, depth - 1, False)
            return total / float(samples)

    def expectimax_choose_move(self, ai_idx):
        ai_player = self.players[ai_idx]
        moves = self.legal_moves_for_player(ai_player)
        if not moves:
            return None
        candidates = []
        for m in moves:
            sim = copy.deepcopy(self)
            sim.apply_move(sim.players[ai_idx], m, ai_idx)
            disc = sim.end_of_round_settlement()
            sim.discard.extend(disc)
            hv = self.heuristic_evaluate(sim, ai_idx)
            candidates.append((hv, m))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_moves = [m for _, m in candidates[:TOP_K_MOVES]]

        best_move = None
        best_val = -1e9
        for m in top_moves:
            sim = copy.deepcopy(self)
            sim.apply_move(sim.players[ai_idx], m, ai_idx)
            disc = sim.end_of_round_settlement()
            sim.discard.extend(disc)
            if sim.all_factories_and_center_empty():
                sim.current_player = sim.first_player_next
                v = self.expectimax(sim, ai_idx, SEARCH_DEPTH, True)
            else:
                sim.current_player = (sim.current_player + 1) % sim.num_players
                v = self.expectimax(sim, ai_idx, SEARCH_DEPTH, False)
            if v > best_val:
                best_val = v
                best_move = m
        return best_move

# ---------- GUI ----------
class AzulGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Azul")
        self.game = AzulGame()
        self.tile_size = 24
        self.bg_color = "#8B4513"    # 深棕背景
        self.empty_color = "#DEB887" # 浅棕空格
        self.selected_src = None     # ("F", idx) 或 ("C", None)
        self.selected_color = None
        self.game_over = False

        # 画布比之前更大以避免重叠
        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X)
        tk.Button(ctrl, text="重新开始", command=self.restart).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Button(ctrl, text="再来一局", command=self.new_game).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Button(ctrl, text="结束游戏", command=self.force_end).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Button(ctrl, text="加快电脑 (depth-1)", command=self.slow_ai).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="减慢电脑 (depth+1)", command=self.faster_ai).pack(side=tk.LEFT, padx=6)
                # player / ai 起始 Y（改为更低以避免与顶部工厂/center 区重叠）

        self.status = tk.Label(root, text="点击左侧工厂或中心选择颜色，然后右侧选择 Pattern Line 行或 Floor。", font=("Arial", 12))
        self.status.pack(fill=tk.X)
        self.canvas = tk.Canvas(root, width=1500, height=600, bg=self.bg_color)
        self.canvas.pack(fill=tk.BOTH, expand=True)


        self.player_y_offset = 180
        self.ai_y_offset = 400

        # 启动回合
        self.start_round()

    # 控件回调
    def slow_ai(self):
        global SEARCH_DEPTH
        if SEARCH_DEPTH > 0:
            SEARCH_DEPTH -= 1
        self.status.config(text=f"SEARCH_DEPTH = {SEARCH_DEPTH}")

    def faster_ai(self):
        global SEARCH_DEPTH
        SEARCH_DEPTH += 1
        self.status.config(text=f"SEARCH_DEPTH = {SEARCH_DEPTH}")

    def restart(self):
        """保持窗口，重置当前游戏（相当于开始新局）"""
        self.game = AzulGame()
        self.selected_src = None
        self.selected_color = None
        self.game_over = False
        self.start_round()

    def new_game(self):
        """再来一局（接口）"""
        self.restart()

    def force_end(self):
        self.end_game()

    def start_round(self):
        if self.game_over:
            return
        self.game.current_player = self.game.first_player_next
        added = self.game.refill_factories()
        if added > 0 and "FIRST" not in self.game.center:
            self.game.center.insert(0, "FIRST")
        self.update_display()
        # 若电脑先手
        if self.game.players[self.game.current_player].name == "电脑":
            self.root.after(350, self.ai_turn)

    def update_display(self):
        self.canvas.delete("all")
        color_map = {"蓝": "blue", "黄": "yellow", "红": "red", "黑": "black", "白": "white", "FIRST": "orange"}

        # ---------- 左侧：Factories ----------
        for i, f in enumerate(self.game.factories):
            x = 20 + (i % 5) * 140
            y = 20 + (i // 5) * 120
            self.canvas.create_rectangle(x, y, x+120, y+90, fill="#cfcfcf", outline="black")
            self.canvas.create_text(x+60, y-8, text=f"F{i}", fill="black", font=("Arial", 10, "bold"))
            # show counts per color optionally (small numbers)
            counts = defaultdict(int)
            for t in f:
                counts[t] += 1
            # draw tiles
            for j, tile in enumerate(f):
                tx = x + 10 + (j % 2) * 50
                ty = y + 10 + (j // 2) * 34
                tag = f"f_{i}_{j}"
                oval = self.canvas.create_oval(tx, ty, tx+self.tile_size, ty+self.tile_size,
                                               fill=color_map.get(tile, "gray"), tags=(tag,))
                self.canvas.tag_bind(tag, "<Button-1>", lambda ev, idx=i, col=tile: self.on_factory_click(idx, col))
            # draw small counts
            cx = x + 60
            cy = y + 62
            off = 0
            for col in COLORS:
                if counts[col] > 0:
                    self.canvas.create_text(cx - 36 + off*24, cy, text=str(counts[col]), font=("Arial", 8))
                off += 1

        # ---------- 中心 ----------
        cx = 420
        cy = 150
        self.canvas.create_text(cx, cy-80, text="Center", fill="white", font=("Arial", 12, "bold"))
        self.canvas.create_oval(cx-110, cy-40, cx+350, cy+40, outline="black", width=1, fill="#dcd0b0")
        # draw center tiles and FIRST token if present
        for j, tile in enumerate(self.game.center):
            tx = cx - 80 + j * 34
            ty = cy - 16
            tag = f"c_{j}"
            if tile == "FIRST":
                # larger orange circle with label
                self.canvas.create_oval(tx, ty, tx+self.tile_size+4, ty+self.tile_size+4, fill="orange", outline="black")
                self.canvas.create_text(tx + (self.tile_size+4)//2, ty + (self.tile_size+4)//2, text="1st", font=("Arial", 8, "bold"))
            else:
                oval = self.canvas.create_oval(tx, ty, tx+self.tile_size-6, ty+self.tile_size-6,
                                               fill=color_map.get(tile, "gray"), tags=(tag,))
                self.canvas.tag_bind(tag, "<Button-1>", lambda ev, col=tile: self.on_center_click(col))

        # ---------- 弃牌堆显示（discard pile） ----------
        disc_x = 20
        disc_y = 420
        self.canvas.create_text(disc_x, disc_y-18, text="弃牌堆 (Discard)", fill="white", anchor="w", font=("Arial", 11, "bold"))
        # show up to 30 tiles (6 per row)
        max_display = 30
        disc_to_show = self.game.discard[-max_display:]
        for idx, tile in enumerate(disc_to_show):
            row = idx // 6
            col = idx % 6
            tx = disc_x + col * (self.tile_size + 8)
            ty = disc_y + row * (self.tile_size + 8)
            if tile == "FIRST":
                self.canvas.create_oval(tx, ty, tx+self.tile_size, ty+self.tile_size, fill="orange", outline="black")
                self.canvas.create_text(tx + self.tile_size//2, ty + self.tile_size//2, text="1st", font=("Arial", 8))
            else:
                self.canvas.create_oval(tx, ty, tx+self.tile_size, ty+self.tile_size, fill=color_map.get(tile, "gray"), outline="black")

        # ---------- 右侧：玩家棋盘(上) ----------
        right_x = 520
        player_y = self.player_y_offset
        self.draw_player_area(self.game.players[0], right_x, player_y, "玩家", interactive=True)

        # ---------- 右侧：电脑棋盘(下) ----------
        ai_y = self.ai_y_offset
        self.draw_player_area(self.game.players[1], right_x, ai_y, "电脑", interactive=False)

        # 当前回合提示
        cur = self.game.current_player
        self.canvas.create_text(800, 980, text=f"当前: {self.game.players[cur].name}    FIRST token next index: {self.game.first_player_next}", fill="white", font=("Arial", 12))

    def draw_player_area(self, player, start_x, start_y, label, interactive=True):
        """绘制一个玩家区域（pattern lines, wall, floor）。interactive 表示是否响应点击（仅玩家 True）"""
        color_map = {"蓝": "blue", "黄": "yellow", "红": "red", "黑": "black", "白": "white", "FIRST": "orange"}
        # 标题与分数
        self.canvas.create_text(start_x+10, start_y+10, text=f"{label} 分数: {player.score}", anchor="w", fill="white", font=("Arial", 12, "bold"))

        # Pattern Lines（左对齐）
        pl_x = start_x
        pl_y = start_y + 30
        for r in range(5):
            cap = r + 1
            # 绘制空槽背景
            for s in range(cap):
                slot_x = pl_x + s * (self.tile_size + 8)
                slot_y = pl_y + r * (self.tile_size + 8)
                self.canvas.create_rectangle(slot_x, slot_y, slot_x+self.tile_size, slot_y+self.tile_size,
                                             fill=self.empty_color, outline="black")
            # 绘制已放的 tiles
            for idx, tile in enumerate(player.pattern_lines[r]):
                slot_x = pl_x + idx * (self.tile_size + 8)
                slot_y = pl_y + r * (self.tile_size + 8)
                if tile == "FIRST":
                    self.canvas.create_oval(slot_x+2, slot_y+2, slot_x+self.tile_size-2, slot_y+self.tile_size-2, fill="orange", outline="black")
                    self.canvas.create_text(slot_x + self.tile_size//2, slot_y + self.tile_size//2, text="1st", font=("Arial", 8))
                else:
                    self.canvas.create_oval(slot_x+2, slot_y+2, slot_x+self.tile_size-2, slot_y+self.tile_size-2,
                                            fill=color_map.get(tile, "gray"), outline="black")
            # 如果这是玩家区并且可交互，绑定点击区域（选择为目标行）
            if interactive and player.name == "你":
                area_tag = f"pl_area_{r}_{start_x}_{start_y}"
                rx = pl_x
                ry = pl_y + r * (self.tile_size + 8)
                width = cap * (self.tile_size + 8)
                rect = self.canvas.create_rectangle(rx, ry, rx+width, ry+self.tile_size, outline="", tags=(area_tag,))
                self.canvas.tag_bind(area_tag, "<Button-1>", lambda ev, row=r+1: self.on_pattern_line_click(row))

            # 行号文本
            self.canvas.create_text(pl_x - 18, pl_y + r * (self.tile_size + 8) + self.tile_size/2, text=str(cap), fill="white")

        # Wall grid (右侧)
        wall_x = pl_x + 6 * (self.tile_size + 8) + 20
        wall_y = pl_y
        for r in range(5):
            for c in range(5):
                slot_x = wall_x + c * (self.tile_size + 8)
                slot_y = wall_y + r * (self.tile_size + 8)
                tile = player.wall[r][c]
                if tile:
                    if tile == "FIRST":
                        self.canvas.create_oval(slot_x+1, slot_y+1, slot_x+self.tile_size-1, slot_y+self.tile_size-1, fill="orange", outline="black")
                        self.canvas.create_text(slot_x + self.tile_size//2, slot_y + self.tile_size//2, text="1st", font=("Arial", 8))
                    else:
                        self.canvas.create_rectangle(slot_x, slot_y, slot_x+self.tile_size, slot_y+self.tile_size,
                                                     fill=color_map.get(tile, "gray"), outline="black")
                else:
                    self.canvas.create_rectangle(slot_x, slot_y, slot_x+self.tile_size, slot_y+self.tile_size,
                                                 fill=self.empty_color, outline="black")
            # 模板文字
            templ = WALL_TEMPLATE[r]
            self.canvas.create_text(wall_x + 5*(self.tile_size+8) + 14, wall_y + r*(self.tile_size+8) + self.tile_size/2,
                                    text=" ".join(templ), anchor="w", fill="white")

        # Floor 区域
        floor_x = pl_x
        floor_y = wall_y + 5 * (self.tile_size + 8) + 18
        self.canvas.create_text(floor_x - 40, floor_y + self.tile_size/2, text="Floor", fill="white", anchor="w")
        for i, tile in enumerate(player.floor):
            tx = floor_x + i * (self.tile_size + 8)
            ty = floor_y
            if tile == "FIRST":
                self.canvas.create_oval(tx+2, ty+2, tx+self.tile_size-2, ty+self.tile_size-2, fill="orange", outline="black")
                self.canvas.create_text(tx + self.tile_size//2, ty + self.tile_size//2, text="1st", font=("Arial", 8))
            else:
                self.canvas.create_oval(tx+2, ty+2, tx+self.tile_size-2, ty+self.tile_size-2,
                                        fill=color_map.get(tile, "gray"), outline="black")
        # Floor 点击（仅玩家区域可交互）
        if interactive and player.name == "你":
            floor_tag = f"player_floor_area_{start_x}_{start_y}"
            rect = self.canvas.create_rectangle(floor_x, floor_y, floor_x + 10*(self.tile_size+8), floor_y + self.tile_size,
                                                outline="", tags=(floor_tag,))
            self.canvas.tag_bind(floor_tag, "<Button-1>", lambda ev: self.on_pattern_line_click(0))

    # ---------- 点击事件 ----------
    def on_factory_click(self, factory_idx, color):
        if self.game_over:
            return
        if self.game.players[self.game.current_player].name != "你":
            self.status.config(text="现在不是你的回合。")
            return
        # 工厂中可能已被选走，检查
        if color not in self.game.factories[factory_idx]:
            self.status.config(text="该工厂已无此颜色，重新选择。")
            return
        self.selected_src = ("F", factory_idx)
        self.selected_color = color
        self.status.config(text=f"已选 工厂 F{factory_idx} 的 {color}。请选择放到哪一行（1-5）或 Floor。")

    def on_center_click(self, color):
        if self.game_over:
            return
        if self.game.players[self.game.current_player].name != "你":
            self.status.config(text="现在不是你的回合。")
            return
        if color not in self.game.center:
            self.status.config(text="中心已无此颜色，重新选择。")
            return
        self.selected_src = ("C", None)
        self.selected_color = color
        self.status.config(text=f"已选 中心 的 {color}。请选择放到哪一行（1-5）或 Floor。")

    def on_pattern_line_click(self, row):
        """row: 1..5 for pattern lines, 0 for floor"""
        if self.game_over:
            return
        if self.game.players[self.game.current_player].name != "你":
            self.status.config(text="现在不是你的回合。")
            return
        if self.selected_src is None or self.selected_color is None:
            self.status.config(text="请先左侧选择工厂或中心的颜色。")
            return
        src = self.selected_src[0]
        idx = self.selected_src[1] if src == "F" else None
        move = ("F", idx, self.selected_color, row) if src == "F" else ("C", None, self.selected_color, row)
        # 执行 move
        player = self.game.players[self.game.current_player]
        # 校验来源是否还存在（防止重复点击）
        if src == "F":
            if self.selected_color not in self.game.factories[idx]:
                self.status.config(text="工厂中已无该颜色，请重新选择来源。")
                self.selected_src = None
                self.selected_color = None
                self.update_display()
                return
        else:
            if self.selected_color not in self.game.center:
                self.status.config(text="中心已无该颜色，请重新选择来源。")
                self.selected_src = None
                self.selected_color = None
                self.update_display()
                return

        self.game.apply_move(player, move, self.game.current_player)
        # 清空选中
        self.selected_src = None
        self.selected_color = None
        self.update_display()
        # 继续流程（可能是下一玩家或回合结算）
        self.root.after(120, self.next_turn)

    # ---------- AI & 回合推进 ----------
    def ai_turn(self):
        if self.game_over:
            return
        ai_idx = self.game.current_player
        # 在 AI 思考期间给出提示
        old_status = self.status.cget("text")
        self.status.config(text="电脑思考中...")
        self.root.update_idletasks()
        move = self.game.expectimax_choose_move(ai_idx)
        if move:
            self.game.apply_move(self.game.players[ai_idx], move, ai_idx)
        self.update_display()
        self.status.config(text=old_status)
        self.root.after(180, self.next_turn)

    def next_turn(self):
        if self.game_over:
            return
        # 若 factories & center 都空 -> 回合结算
        if self.game.all_factories_and_center_empty():
            discarded = self.game.end_of_round_settlement()
            if discarded:
                self.game.discard.extend(discarded)
            # 检查终局条件
            if any(p.has_completed_row() for p in self.game.players):
                self.end_game()
                return
            # 新一轮
            self.start_round()
            return
        # 切换玩家
        self.game.current_player = (self.game.current_player + 1) % self.game.num_players
        self.update_display()
        # 若对方为电脑，则 AI 行动
        if self.game.players[self.game.current_player].name == "电脑":
            self.root.after(220, self.ai_turn)

    # ---------- 结束 ----------
    def end_game(self):
        self.game_over = True
        # 最终结算加额外奖励
        for p in self.game.players:
            p.endgame_bonus()
        self.update_display()
        winner = max(self.game.players, key=lambda x: x.score)
        msg = f"游戏结束！\n\n{self.game.players[0].name}: {self.game.players[0].score}\n{self.game.players[1].name}: {self.game.players[1].score}\n\n胜者: {winner.name}"
        messagebox.showinfo("游戏结束", msg)

# ---------- 入口 ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = AzulGUI(root)
    root.mainloop()


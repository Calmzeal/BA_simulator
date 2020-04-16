import collections


class StrategyFixedPeerLatency:
    def __init__(self, attack_param):  # Checked
        self.extra_send = attack_param.extra_send
        self.initialize_attack()

    def initialize_attack(self):  # Checked
        # State fields.
        self.left_subtree_weight = 0
        self.right_subtree_weight = 0
        self.left_withheld_blocks_queue = collections.deque()
        self.right_withheld_blocks_queue = collections.deque()

        self.oldest_block_time_deplacement_list = []
        self.max_time_deplacement = 0

    def start_attack(self):  # Checked
        # FIXME: determine the initial condition for the real world attack.
        self.right_subtree_weight += 1

    def adversary_side_to_mine(self):  # Checked
        return "L" if len(self.left_withheld_blocks_queue) + self.left_subtree_weight \
            < len(self.right_withheld_blocks_queue) + self.right_subtree_weight else \
        "R"

    def adversary_mined(self, timestamp, blk_id, side):  # Checked
        if side == "L":
            withhold_queue = self.left_withheld_blocks_queue
        else:
            withhold_queue = self.right_withheld_blocks_queue
        withhold_queue.append((timestamp, blk_id))

    def honest_mined(self, side):  # Checked
        if side == "L":
            self.left_subtree_weight += 1
        else:
            self.right_subtree_weight += 1


    def adversary_strategy(self,
                           timestamp,
                           blocks_to_send,
                           left_weight_diff_approx,
                           right_weight_diff_approx):

        global_subtree_weight_diff = self.left_subtree_weight - self.right_subtree_weight
        extra_send = 0
        diff_sector = -global_subtree_weight_diff if global_subtree_weight_diff<0 else global_subtree_weight_diff+1
        if diff_sector < 3:
            extra_send -= self.extra_send.withhold
        elif diff_sector == 3:
            extra_send += self.extra_send.extra1
        else:
            extra_send += self.extra_send.extra2

        left_send_count = int(-left_weight_diff_approx + extra_send)
        right_send_count = int(right_weight_diff_approx + 1 + extra_send)
        if self.left_subtree_weight >= self.right_subtree_weight:
            left_send_count = 0
        else:
            right_send_count = 0

        self.max_time_deplacement = 0
        for i in range(left_send_count):
            self.pop_withheld_block_to_send(timestamp, blocks_to_send, "L")
        for i in range(right_send_count):
            self.pop_withheld_block_to_send(timestamp, blocks_to_send, "R")
        if self.max_time_deplacement > 0:
            self.oldest_block_time_deplacement_list.append(self.max_time_deplacement)

    def pop_withheld_block_to_send(self, timestamp, blocks_to_send, side):
        if side == "L":
            withheld_queue = self.left_withheld_blocks_queue
        else:
            withheld_queue = self.right_withheld_blocks_queue

        if len(withheld_queue)>0:
            oldtime, blk_id = withheld_queue.pop()
            if timestamp-oldtime>self.max_time_deplacement:
                self.max_time_deplacement = round(timestamp-oldtime, 2)
        else:
            return

        if side == "L":
            self.left_subtree_weight += 1
        else:
            self.right_subtree_weight += 1
        blocks_to_send.append((blk_id, side))
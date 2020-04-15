import collections
import random

class StrategyFixedPeerLatency:
    def __init__(self, withhold, extra_send, one_way_latency):  # Checked
        # Config fields
        self.withhold = withhold
        self.extra_send = extra_send
        # The one way latency between adversary and honest nodes.
        self.one_way_latency = one_way_latency
        self.initialize_attack()

    def initialize_attack(self):  # Checked
        # State fields.
        self.left_subtree_weight = 0
        self.right_subtree_weight = 0
        self.left_withheld_blocks_queue = collections.deque()
        self.right_withheld_blocks_queue = collections.deque()
        self.withhold_done = False

        self.approx_left_target_subtree_weight_diff = 0
        self.approx_right_target_subtree_weight_diff = 0

        self.oldest_block_time_deplacement_list = []
        self.max_time_deplacement = 0


    def start_attack(self):  # Checked
        # FIXME: determine the initial condition for the real world attack.
        self.right_subtree_weight += 1

    def adversary_side_to_mine(self):  # Checked
        return "L" if len(self.left_withheld_blocks_queue) + self.left_subtree_weight \
            < len(self.right_withheld_blocks_queue) + self.right_subtree_weight else \
        "R"

    def adversary_mined(self, side, block,timestamp):  # Checked
        #self.oldest_block_time_deplacement_list.append((timestamp,"Mined"))
        if side == "L":
            withhold_queue = self.left_withheld_blocks_queue
        else:
            withhold_queue = self.right_withheld_blocks_queue
        withhold_queue.append((timestamp,block))

    def honest_mined(self, side, time_mined, block):  # Checked
        if side == "L":
            self.left_subtree_weight += 1
        else:
            self.right_subtree_weight += 1


    def adversary_strategy(self,
                           timestamp,
                           blocks_to_send,
                           left_weight_diff_approx=None,
                           right_weight_diff_approx=None):

        # When adversary run the strategy too close to its previous run, the released
        # withheld blocks are not delivered to the honest miners yet, thus the adversary
        # must take into consideration of the effect of the previously sent blocks.
        #
        # Consider the extreme case: what if the strategy is triggered twice each time.
        # The latter run should not send out any blocks.

        if not (left_weight_diff_approx is None):
            self.approx_left_target_subtree_weight_diff = left_weight_diff_approx
            self.approx_right_target_subtree_weight_diff = right_weight_diff_approx


        global_subtree_weight_diff = self.left_subtree_weight - self.right_subtree_weight

        self.withhold_done = True\
            if len(self.left_withheld_blocks_queue) + len(self.right_withheld_blocks_queue) >= self.withhold\
            else False
        extra_send = self.extra_send

        diff_sector = -global_subtree_weight_diff if global_subtree_weight_diff<0 else global_subtree_weight_diff+1

        if diff_sector < 3:
            extra_send -= 0.9
        elif diff_sector == 3:
            extra_send += 0.1
        else:
            extra_send +=1

        left_send_count = -self.approx_left_target_subtree_weight_diff + extra_send
        right_send_count = self.approx_right_target_subtree_weight_diff + 1 + extra_send
        if self.left_subtree_weight >= self.right_subtree_weight:
            left_send_count = 0
            right_send_count = int(right_send_count)#+(random.random() < right_send_count-int(right_send_count))
        else:
            right_send_count = 0
            left_send_count = int(left_send_count)# + (random.random() < left_send_count-int(left_send_count))



        self.max_time_deplacement = 0
        if self.withhold_done:
            for i in range(left_send_count):
                self.pop_withheld_block_to_send("L", timestamp, blocks_to_send)
            for i in range(right_send_count):
                self.pop_withheld_block_to_send("R", timestamp, blocks_to_send)
        if self.max_time_deplacement > 0:
            #self.oldest_block_time_deplacement_list.append(f"Attack at {timestamp}")
            self.oldest_block_time_deplacement_list.append(
                self.max_time_deplacement)
                # ((self.approx_left_target_subtree_weight_diff,
                #  self.approx_right_target_subtree_weight_diff),
                #  global_subtree_weight_diff,
                #  (left_send_count,
                #  right_send_count)))

        return 0

    def pop_withheld_block_to_send(self, side, timestamp, blocks_to_send):
        if side == "L":
            withheld_queue = self.left_withheld_blocks_queue
        else:
            withheld_queue = self.right_withheld_blocks_queue
        if len(withheld_queue)>0:
            oldtime,blk = withheld_queue.pop()
            if timestamp-oldtime>self.max_time_deplacement:
                self.max_time_deplacement = round(timestamp-oldtime,2)
        else:
            return 0
        if side == "L":
            self.left_subtree_weight += 1
        else:
            self.right_subtree_weight += 1
        blocks_to_send.append((side, blk))
        return 0
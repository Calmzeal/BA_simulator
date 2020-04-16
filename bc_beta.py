#!/usr/bin/env python3
import queue
import random
import multiprocessing
import copy
import time
from strategy_fixed_peer_latency_beta import StrategyFixedPeerLatency


class ParamSet:
    """
    Parameters will be eventually passed to python functions.
    Although their order doesn't matter for the purpose of simulation,
    it matters in function's arguments list.
    Even a subtle modification of one parameter will reorder strategy function interface,
    and the designer must obey such meaningless order everywhere.
    We only need to order those parameters in representation.

    So we create this class to avoid param order in function interface,
    and require less code:) then dictionary during parameter creation.
    """
    attr_Env = \
        ["num_nodes",
         "average_block_period",
         "evil_rate",
         "latency",
         "out_degree",
         "termination_time",
         "one_way_latency"]

    attr_TestParam = ["latency", "num_nodes", "out_degree", "extra_send"]
    attr_ExtraSend = ["withhold", "extra1", "extra2"]
    REPR_ORDER = [attr_ExtraSend, attr_TestParam]

    def __init__(self):
        return

    def __repr__(self):
        for attr_list in ParamSet.REPR_ORDER:
            if self.type_check(attr_list):
                return self.type_demo(attr_list)
        return "Unknown Parameter Set"

    def type_check(self, attr_list):
        type_correct = True
        for attr in attr_list:
            type_correct = type_correct and hasattr(self, attr)
        return type_correct

    def type_demo(self, attr_list):
        return "".join(["{",
                        ", ".join(map(lambda x: " : ".join([x, repr(getattr(self, x))]), attr_list)),
                        "}"])


class NodeLocalView:
    """
    We create this class to simulate honest node's behavior.
    """

    def __init__(self, node_id):
        self.node_id = node_id
        self.left_subtree_weight = 0
        self.right_subtree_weight = 0
        self.received = set()
        self.update_chirality()

    def __repr__(self):
        return f"NodeWeight({self.left_subtree_weight}, {self.right_subtree_weight})"

    def deliver_block(self, blk_id, side):
        self.received.add(blk_id)
        if side == "L":
            self.left_subtree_weight += 1
        else:
            self.right_subtree_weight += 1
        self.update_chirality()

    def update_chirality(self):
        if self.left_subtree_weight >= self.right_subtree_weight:
            self.chirality = "L"
        else:
            self.chirality = "R"


class Simulator:
    # Build the finite event set
    EVENT_MINE_BLOCK = "0. mine_block_event"
    EVENT_BLOCK_DELIVER = "1. block_delivery_event"
    EVENT_ADV_RECEIVED_BLOCK = "2. adversary_received_honest_mined_block"
    EVENT_CHECK_MERGE = "3. test_check_merge"
    EVENT_QUEUE_EMPTY = "4. event_queue_empty"
    EVENT_ADV_STRATEGY_TRIGGER = "5. run_adv_strategy"

    def __init__(self, env):
        self.env = env
        if not env.type_check(ParamSet.attr_Env):
            print("Invalid Simulation Setting")
            exit()
        attack_param = ParamSet()
        attack_param.extra_send = env.extra_send
        self.adversary = StrategyFixedPeerLatency(attack_param)
        self.merge_count = 0
        self.event_queue = queue.PriorityQueue()

    def setup_chain(self):
        self.nodes = []
        for i in range(self.env.num_nodes):
            self.nodes.append(NodeLocalView(i))

    def topo_generator(self, N, out_degree):
        # 1. Generate valid connections
        # \correct only when latency<1e9.
        g = [[1e9] * N for i in range(N)]
        g_out_degree = [0] * N
        for i in range(N):
            nodes_to_choose = set(range(N))
            while out_degree - g_out_degree[i] > 0 and len(nodes_to_choose) > 0:
                peer = random.sample(nodes_to_choose, 1)[0]
                nodes_to_choose.remove(peer)
                while (peer == i or g[i][peer] < 1e8 or g_out_degree[peer] >= out_degree) \
                        and len(nodes_to_choose) > 0:
                    peer = random.sample(nodes_to_choose, 1)[0]
                    nodes_to_choose.remove(peer)
                if not (peer == i or g[i][peer] < 1e8 or g_out_degree[peer] >= out_degree):
                    g[i][peer] = self.env.latency * random.uniform(0.75, 1.25)
                    g[peer][i] = g[i][peer]
                    g_out_degree[i] += 1
                    g_out_degree[peer] += 1

        # 2. Calculate shortest message passing paths using Floyd Algorithm
        self.origin_topo = copy.deepcopy(g)
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    g[i][j] = g[i][j] if i == j else min(g[i][j], g[i][k] + g[k][j])

        # 3. Drop the unconnected topologies
        self.latency_map = g
        for i in range(N):
            for j in range(N):
                if i != j and g[i][j] > 1e8:
                    return False
        return True

    def setup_network(self):
        # 1. Create admissible network topo
        while not (self.topo_generator(self.env.num_nodes, self.env.out_degree)):
            pass

        # 2. Create network partitions
        self.nodes_to_keep_left = list(range(0, self.env.num_nodes, 2))
        self.nodes_to_keep_right = list(range(1, self.env.num_nodes, 2))

        # 3. Pre-processing the broadcasting message passing distance between partitions
        self.latency_cross_partition = [1e9] * self.env.num_nodes
        for node in self.nodes_to_keep_left:
            self.latency_cross_partition[node] = min([self.latency_map[node][i] for i in self.nodes_to_keep_right])
        for node in self.nodes_to_keep_right:
            self.latency_cross_partition[node] = min([self.latency_map[node][i] for i in self.nodes_to_keep_left])

    def run_test(self):
        # 1. Initialize the target's tree
        block_id = 0
        for i in self.nodes_to_keep_left:
            self.nodes[i].chirality = "L"
        for i in self.nodes_to_keep_right:
            self.nodes[i].chirality = "R"
            self.nodes[i].deliver_block(block_id, "R")
            self.honest_node_broadcast_block(0., i, block_id, "R")

        # FIXME: start up condition
        self.adversary.start_attack()

        # 2. Executed the simulation
        timestamp = 0
        self.event_queue.put((0, Simulator.EVENT_MINE_BLOCK, None))
        while timestamp < self.env.termination_time:
            # Event 1
            event_type, timestamp, event = self.process_network_events()
            # Event 0
            if event_type == Simulator.EVENT_MINE_BLOCK:
                block_id += 1
                time_to_next_block = random.expovariate(1 / self.env.average_block_period)
                self.event_queue.put((timestamp + time_to_next_block, Simulator.EVENT_MINE_BLOCK, None))

                adversary_mined = random.random() < self.env.evil_rate
                # Case 1
                if adversary_mined:
                    side = self.adversary.adversary_side_to_mine()
                    self.adversary.adversary_mined(timestamp, block_id, side)
                # Case 2
                else:
                    miner = random.randint(0, self.env.num_nodes - 1)
                    side = self.nodes[miner].chirality
                    self.nodes[miner].deliver_block(block_id, side)
                    self.honest_node_broadcast_block(timestamp, miner, block_id, side)
                    self.event_queue.put((
                        timestamp + self.env.one_way_latency,
                        Simulator.EVENT_ADV_RECEIVED_BLOCK,
                        (block_id, side)))
            # Event 2
            if event_type == Simulator.EVENT_ADV_RECEIVED_BLOCK:
                honest_mined_block, honest_mined_side = event
                self.adversary.honest_mined(honest_mined_side)
            # Event 3
            if event_type == Simulator.EVENT_CHECK_MERGE:
                '''
                print(f"At {timestamp} local views after action:\n\tleft targets: %s,\n\tright targets: %s\n" % (
                    repr([self.nodes[i] for i in self.nodes_to_keep_left]),
                    repr([self.nodes[i] for i in self.nodes_to_keep_right]),
                ))
                '''
                self.merge_count = self.merge_count + 1 if self.is_chain_merged() else 0
                if self.merge_count > 0:
                    # print(f"Chain merged after {timestamp} seconds")
                    return timestamp
            # Event 4
            if event_type == Simulator.EVENT_QUEUE_EMPTY:
                # Can't happen because of mining.
                pass
            # Event 5
            if event_type == Simulator.EVENT_ADV_STRATEGY_TRIGGER:
                self.attack_execution(timestamp, event)

        print(f"Chain unmerged after {self.env.termination_time} seconds... ")
        statistic = sorted(self.adversary.oldest_block_time_deplacement_list)
        group = list(map(
            lambda percentile: statistic[int(len(statistic) * percentile / 10)],
            range(10)))
        group.append(statistic[-1])
        print(f"Attacks being executed {len(statistic)} times, the 10 percentile result is {group}")

        return self.env.termination_time

    def is_chain_merged(self):
        side_per_node = list(map(
            lambda node: node.chirality,
            self.nodes
        ))
        return (not "L" in side_per_node) or (not "R" in side_per_node)

    def honest_node_broadcast_block(self, time, miner, blk_id, side):
        for receiver in range(self.env.num_nodes):
            if receiver != miner:
                deliver_time = time + self.latency_map[miner][receiver]
                self.event_queue.put((deliver_time, Simulator.EVENT_BLOCK_DELIVER, (receiver, blk_id, side)))

    def attacker_broadcast_block(self, deliver_time, target_partition, blk_id, side):
        for receiver in target_partition:
            self.event_queue.put((deliver_time, Simulator.EVENT_BLOCK_DELIVER, (receiver, blk_id, side)))
        counter_partition = self.nodes_to_keep_left if side == "R" else self.nodes_to_keep_right
        for receiver in counter_partition:
            self.event_queue.put((
                deliver_time+self.latency_cross_partition[receiver],
                Simulator.EVENT_BLOCK_DELIVER,
                (receiver, blk_id, side)))

    def attack_execution(self, timestamp, event):
        blocks_to_send = []
        if event is None:
            event = (None,None)
        left_weight_diff_approx, right_weight_diff_approx = event
        self.adversary.adversary_strategy(
            timestamp,
            blocks_to_send,
            left_weight_diff_approx,
            right_weight_diff_approx
        )

        time_delivery = timestamp + self.env.one_way_latency
        for blk_id, side in blocks_to_send:
            if side == "L":
                target_partition = self.nodes_to_keep_left
            else:
                target_partition = self.nodes_to_keep_right
            self.attacker_broadcast_block(time_delivery, target_partition, blk_id, side)

    def process_network_events(self, current_time=None):
        # Parse events and generate new ones in a BFS way
        get_weight_diff = lambda i: (self.nodes[i].left_subtree_weight - self.nodes[i].right_subtree_weight)

        while True:
            if self.event_queue.empty():
                return (Simulator.EVENT_QUEUE_EMPTY, self.env.termination_time, None)

            time, event_type, event = self.event_queue.get()

            if current_time is not None and time > current_time:
                self.event_queue.put((time, event_type, event))

            # Event 1
            if event_type == Simulator.EVENT_BLOCK_DELIVER:
                receiver, blk_id, side = event
                if not blk_id in self.nodes[receiver].received:
                    self.nodes[receiver].deliver_block(blk_id, side)
                    self.event_queue.put((time, Simulator.EVENT_CHECK_MERGE, None))

                    approx_left_target_subtree_weight_diff = round(
                        2 * sum(map(get_weight_diff, self.nodes_to_keep_left)) / len(self.nodes), 2)
                    approx_right_target_subtree_weight_diff = round(
                        2 * sum(map(get_weight_diff, self.nodes_to_keep_right)) / len(self.nodes), 2)

                    self.event_queue.put((time + self.env.one_way_latency,
                                          Simulator.EVENT_ADV_STRATEGY_TRIGGER,
                                          (approx_left_target_subtree_weight_diff,
                                           approx_right_target_subtree_weight_diff)))
            else:
                return (event_type, time, event)

    def main(self):
        self.setup_chain()
        self.setup_network()
        return self.run_test()


def slave_simulator(env):
    return round(Simulator(env).main(), 2)


if __name__ == "__main__":
    cpu_num = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpu_num)
    repeats = 100
    print(f"repeats={repeats}")

    # modifiable parameters
    num_nodes = 60
    latency = 1.25
    out_degree = 3
    withhold = 1
    extra1 = 0
    extra2 = 1

    # fixed parameters
    test_params = ParamSet()
    test_params.average_block_period = 0.5
    test_params.evil_rate = 0.2
    test_params.termination_time = 5400
    test_params.one_way_latency = 0.1
    extra_send = ParamSet()

    # partially fixed parameters
    test_params.num_nodes = num_nodes
    test_params.latency = latency
    test_params.out_degree = out_degree
    # extra_send.withhold = withhold
    # extra_send.extra1 = extra1
    extra_send.extra2 = extra2

    # parameters to test
    for withhold in [1]:
        for extra1 in [0]:
            # test_params.num_nodes = num_nodes
            # test_params.latency = latency
            # test_params.out_degree = out_degree
            extra_send.withhold = withhold
            extra_send.extra1 = extra1
            # extra_send.extra2 = extra2
            test_params.extra_send = extra_send

            begin = time.time()
            attack_last_time = sorted(p.map(slave_simulator, [test_params] * repeats))
            print(f"{test_params},average_lasting_time:{round(sum(attack_last_time) / repeats, 2)}")
            samples = 10
            print(list(map(lambda percentile: attack_last_time[int((repeats - 1) * percentile / samples)],
                           range(samples + 1))))
            end = time.time()
            print("Executed in %.2f seconds" % (end - begin))

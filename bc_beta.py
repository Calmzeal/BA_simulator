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
    # Define the finite event set
    EVENT_MINE_BLOCK = "0. mine_block_event"
    EVENT_BLOCK_DELIVER = "1. block_delivery_event"
    EVENT_ADV_RECEIVED_BLOCK = "2. adversary_received_honest_mined_block"
    EVENT_CHECK_MERGE = "3. test_check_merge"
    EVENT_QUEUE_EMPTY = "4. event_queue_empty"
    EVENT_ADV_STRATEGY_TRIGGER = "5. run_adv_strategy"

    GEODELAY_ORIGIN = [
        [1, 296.928, 36.986, 20.409, 24.794, 114.291, 124.786, 7.158, 141.355, 27.041, 41.92, 6.151, 204.77, 281.861,
         183.197, 21.62, 276.805, 263.036, 93.053, 91.225],
        [286.701, 1, 295.947, 293.476, 299.945, 183.301, 403.763, 275.475, 162.868, 290.705, 325.031, 280.406, 310.881,
         275.683, 122.325, 309.163, 27.954, 191.476, 200.025, 208.471],
        [37.02, 295.885, 1, 64.691, 44.147, 147.641, 298.107, 32.911, 154.864, 33.321, 79.28, 28.58, 229.974, 283.125,
         283.911, 64.068, 349.435, 308.048, 114.397, 111.466],
        [21.655, 292.504, 64.448, 1, 44.213, 128.031, 166.586, 25.746, 154.795, 22.578, 48.598, 24.474, 226.75, 228.028,
         202.137, 40.721, 329.261, 247.137, 115.047, 102.718],
        [24.645, 300.208, 44.117, 45.836, 1, 127.607, 153.503, 18.473, 159.794, 29.343, 43.855, 24.118, 220.427,
         274.238,
         187.711, 10.267, 350.329, 257.589, 110.576, 104.095],
        [114.348, 177.724, 156.003, 135.368, 128.56, 1, 226.789, 106.173, 29.91, 127.173, 167.252, 116.563, 139.161,
         306.019, 202.181, 143.448, 182.98, 134.586, 43.34, 36.941],
        [124.853, 398.625, 298.227, 170.584, 153.574, 224.058, 1, 232.507, 250.196, 134.52, 175.183, 153.978, 321.26,
         339.943, 84.054, 141.563, 268.929, 261.68, 215.679, 205.258],
        [7.069, 275.129, 32.877, 24.828, 18.435, 106.066, 232.099, 1, 132.723, 23.269, 50.843, 5.334, 197.446, 275.735,
         198.735, 28.293, 294.679, 234.943, 83.542, 73.496],
        [141.236, 162.596, 154.723, 153.674, 159.786, 29.904, 247.623, 132.782, 1, 153.371, 185.834, 144.435, 173.217,
         195.546, 179.019, 159.125, 156.547, 107.611, 58.763, 69.354],
        [27.11, 290.308, 33.351, 30.656, 29.415, 127.132, 134.475, 23.425, 154.048, 1, 56.165, 21.329, 218.702, 342.649,
         267.18, 38.005, 381.607, 249.651, 115.757, 101.378],
        [42.321, 324.974, 79.044, 55.598, 43.859, 167.47, 175.968, 50.674, 185.869, 55.931, 1, 56.673, 248.073, 148.801,
         204.403, 21.157, 342.171, 294.165, 133.209, 129.044],
        [9.209, 279.948, 28.734, 22.704, 28.07, 117.135, 130.943, 5.48, 149.376, 17.61, 61.736, 1, 204.172, 279.17,
         260.415,
         42.758, 365.667, 252.355, 114.698, 91.369],
        [204.529, 310.635, 237.215, 222.165, 214.811, 135.386, 316.096, 190.667, 173.16, 223.296, 252.358, 229.861, 1,
         356.349, 349.441, 228.036, 314.972, 266.768, 137.577, 126.636],
        [284.393, 276.257, 286.52, 227.285, 274.416, 212.749, 342.619, 277.733, 153.516, 336.035, 148.591, 271.151,
         366.362,
         1, 235.712, 284.305, 148.917, 74.656, 216.897, 259.348],
        [180.516, 118.915, 283.852, 202.361, 197.584, 202.168, 84.107, 181.458, 179.024, 267.165, 204.386, 251.497,
         349.76,
         235.791, 1, 202.265, 93.559, 66.873, 241.053, 242.352],
        [21.591, 309.057, 64.05, 41.096, 10.33, 143.388, 141.558, 28.256, 159.239, 38.228, 21.291, 29.722, 228.183,
         284.382,
         202.297, 1, 362.993, 288.137, 126.377, 115.36],
        [295.71, 25.877, 349.888, 360.115, 317.624, 251.018, 270.281, 294.423, 156.664, 317.537, 374.436, 355.073,
         315.271,
         148.589, 93.654, 393.391, 1, 114.007, 220.726, 231.18],
        [265.392, 191.89, 308, 247.975, 257.657, 134.503, 261.22, 235.604, 107.616, 248.756, 294.027, 235.052, 266.815,
         73.862, 67.057, 288.337, 114.908, 1, 172.82, 157.804],
        [92.821, 199.057, 110.957, 115.542, 117.678, 43.363, 212.605, 83.698, 58.722, 115.425, 131.442, 101.325,
         139.016,
         216.807, 241.289, 126.485, 219.442, 172.79, 1, 21.106],
        [94.04, 208.082, 111.349, 103.343, 104.072, 38.178, 205.549, 73.288, 69.407, 105.464, 129.584, 82.41, 126.681,
         239.586, 246.168, 114.586, 262.434, 160.829, 21.182, 1]]

    GEODELAY = [(list(map(lambda x:x/1000, row))) for row in GEODELAY_ORIGIN]

    def __init__(self, env):
        self.env = env
        if not env.type_check(ParamSet.attr_Env):
            print("Invalid Simulation Setting")
            exit()
        attack_param = ParamSet()
        attack_param.extra_send = env.extra_send
        self.adversary = StrategyFixedPeerLatency(attack_param)
        self.merge_count = 0
        self.measurements = {"lasting_time" : self.env.termination_time}
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

        if N == 20:
            g = Simulator.GEODELAY

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
                if self.merge_count > 10000:
                    # print(f"Chain merged after {timestamp} seconds")
                    self.measurements["lasting_time"] = timestamp
                    return
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
                deliver_time + self.latency_cross_partition[receiver],
                Simulator.EVENT_BLOCK_DELIVER,
                (receiver, blk_id, side)))

    def attack_execution(self, timestamp, event):
        blocks_to_send = []
        if event is None:
            event = (None, None)
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
        self.run_test()


def slave_simulator(env):
    experiment = Simulator(env)
    experiment.main()
    return experiment.measurements

def proj(stream, indicator):
    return list(map(lambda x:x[indicator], stream))

if __name__ == "__main__":
    cpu_num = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpu_num)
    repeats = 1000
    print(f"repeats={repeats}")
    # modifiable parameters
    num_nodes = 20
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
            measurements_stream = p.map(slave_simulator, [test_params] * repeats)
            lasting_time_stream = proj(measurements_stream, "lasting_time")
            attack_last_time = sorted(lasting_time_stream)
            print(f"{test_params},average_lasting_time:{round(sum(attack_last_time) / repeats, 2)}")
            samples = 10
            print(list(map(lambda percentile: attack_last_time[int((repeats - 1) * percentile / samples)],
                           range(samples + 1))))
            end = time.time()
            print("Executed in %.2f seconds" % (end - begin))

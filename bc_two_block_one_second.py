#!/usr/bin/env python3
import collections
import queue
import random
import multiprocessing
import math
import copy
import time
from strategy_fixed_peer_latency4 import StrategyFixedPeerLatency

class Parameters:
    def __init__(self):
        return

    def __repr__(self):
        return f"latency:{self.latency},num_nodes:{self.num_nodes},out_degree:{self.out_degree},withhold:{self.withhold},extra_send:{self.extra_send}"

class NodeLocalView:
    def __init__(self, node_id):
        self.node_id = node_id
        self.left_subtree_weight = 0
        self.right_subtree_weight = 0
        self.received = set()
        self.update_chirality()

    def __repr__(self):
        return f"NodeWeight({self.left_subtree_weight}, {self.right_subtree_weight})"

    def deliver_block(self, block_id, chirality):
        self.received.add(block_id)
        # Update the subtree weight
        if chirality == "L":
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
    EVENT_BLOCK_DELIVER = "1. block_delivery_event"
    EVENT_MINE_BLOCK = "0. mine_block_event"
    EVENT_CHECK_MERGE = "3. test_check_merge"
    EVENT_ADV_RECEIVED_BLOCK = "2. adversary_received_honest_mined_block"
    EVENT_ADV_STRATEGY_TRIGGER = "5. run_adv_strategy"
    EVENT_QUEUE_EMPTY = "4. event_queue_empty"

    def __init__(self, env, attack_params):
        self.env = env
        self.adversary = StrategyFixedPeerLatency(
            attack_params["withhold"],
            attack_params["extra_send"],
            attack_params["one_way_latency"],
        )
        # Parameters checker
        for attr in ["num_nodes","average_block_period","evil_rate","latency","out_degree","termination_time"]:
            if not hasattr(self.env, attr):
                print("{} unset".format(attr))
                exit()

        self.merge_count = 0
        self.attack_params = attack_params
        self.event_queue = queue.PriorityQueue()

    def setup_chain(self):
        self.nodes = []
        for i in range(self.env.num_nodes):
            self.nodes.append(NodeLocalView(i))

    def topo_generator(self, N, out_degree):
        # This generator only works for network diameter < 1e8
        g = [[1e9] * N for i in range(N)]
        g_out_degree = [0] * N
        for i in range(N):
            nodes_to_choose = set(range(N))
            while out_degree-g_out_degree[i] > 0 and len(nodes_to_choose) > 0:

                peer = random.sample(nodes_to_choose, 1)[0]
                nodes_to_choose.remove(peer)
                while (peer == i or g[i][peer] < 1e8 or g_out_degree[peer] >= out_degree)\
                        and len(nodes_to_choose) > 0:
                    peer = random.sample(nodes_to_choose, 1)[0]
                    nodes_to_choose.remove(peer)

                if not(peer == i or g[i][peer] < 1e8 or g_out_degree[peer] >= out_degree):
                    g[i][peer] = self.env.latency * random.uniform(0.75, 1.25)
                    g[peer][i] = g[i][peer]
                    g_out_degree[i] += 1
                    g_out_degree[peer] += 1

        self.origin_topo = copy.deepcopy(g)
        # Floyed algorithm to calculate shortest message passing paths
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    g[i][j] = g[i][j] if i == j else min(g[i][j], g[i][k]+g[k][j])

        self.latency_map = g
        # Drop the unconnected topo
        max_latency = 0
        for i in range(N):
            for j in range(N):
                if i != j and g[i][j]>1e8:
                    return False
                else:
                    if i != j:
                        max_latency = max(max_latency,g[i][j])
        # print(max_latency)
        return True

    def setup_network(self):
        # Create admissible network topo
        while not (self.topo_generator(self.env.num_nodes,self.env.out_degree)):
            pass
        # Create network partitions
        self.nodes_to_keep_left = list(range(0, self.env.num_nodes, 2))
        self.nodes_to_keep_right = list(range(1, self.env.num_nodes, 2))
        # Pre-processing the broadcasting message passing distance between partitions
        self.latency_cross_partition = [1e9]*self.env.num_nodes
        for node in self.nodes_to_keep_left:
            self.latency_cross_partition[node] = min([self.latency_map[node][i] for i in self.nodes_to_keep_right])
        for node in self.nodes_to_keep_right:
            self.latency_cross_partition[node] = min([self.latency_map[node][i] for i in self.nodes_to_keep_left])



    def run_test(self):
        # Initialize the target's tree
        for i in self.nodes_to_keep_left:
            self.nodes[i].chirality = "L"
        for i in self.nodes_to_keep_right:
            self.nodes[i].chirality = "R"
            self.nodes[i].deliver_block(0, "R")
            self.honest_node_broadcast_block(0, i, "R", 0)

        # FIXME: start up condition
        self.adversary.start_attack()

        # Executed the simulation
        block_id = 1
        timestamp = 0
        self.event_queue.put((0, Simulator.EVENT_MINE_BLOCK, None))
        while timestamp < self.env.termination_time:
            event_type, time, event = self.process_network_events()
            timestamp = time
            trigger_adversary_action = False

            if event_type == Simulator.EVENT_MINE_BLOCK:
                time_to_next_block = random.expovariate(1 / self.env.average_block_period)
                self.event_queue.put((time + time_to_next_block, Simulator.EVENT_MINE_BLOCK, None))

                adversary_mined = random.random() < self.env.evil_rate
                if adversary_mined:
                    #print("MINED %s by adversary at %s" % (block_id, time))
                    #print("At %s, Adversary mined block %s" % (timestamp, block_id))
                    # Decide attack target
                    side = self.adversary.adversary_side_to_mine()
                    self.adversary.adversary_mined(side, block_id,timestamp)
                    trigger_adversary_action = True
                else:
                    # Pick a number from 0 to num_nodes - 1 inclusive.
                    miner = random.randint(0, self.env.num_nodes-1)
                    #print("At %s, Miner %s mined block %s" % (timestamp, miner, block_id))
                    side = self.nodes[miner].chirality
                    #print("MINED %s %s by node %s at %s" % (block_id, side, miner, time))
                    # Update attacker and miner's views
                    self.nodes[miner].deliver_block(block_id, side)
                    # Broadcast new blocks to neighbours
                    self.honest_node_broadcast_block(timestamp, miner, side, block_id)

                    self.event_queue.put((
                        time + self.attack_params["one_way_latency"],
                        Simulator.EVENT_ADV_RECEIVED_BLOCK,
                        (side, block_id)))
                    # Other miners receive this at timestamp + latency, attacker runs the strategy
                    # earlier so that adversary can deliver blocks before it (or right after it,
                    # it doesn't matter too much).
                    # TODO: for random latency, the adversary can only try to run its strategy more often.
                    self.event_queue.put((
                        time + self.env.latency - self.attack_params["one_way_latency"] - 0.01,
                        Simulator.EVENT_ADV_STRATEGY_TRIGGER,
                        None
                    ))
                block_id += 1

            elif event_type == Simulator.EVENT_CHECK_MERGE:
                '''
                print(f"At {timestamp} local views after action:\n\tleft targets: %s,\n\tright targets: %s\n" % (
                    repr([self.nodes[i] for i in self.nodes_to_keep_left]),
                    repr([self.nodes[i] for i in self.nodes_to_keep_right]),
                ))
                '''
                self.merge_count = self.merge_count+1 if self.is_chain_merged() else 0
                if self.merge_count > 0:
                    # print(f"Chain merged after {timestamp} seconds")
                    return timestamp
            elif event_type == Simulator.EVENT_QUEUE_EMPTY:
                # Can't happen because of mining.
                pass
            elif event_type == Simulator.EVENT_ADV_RECEIVED_BLOCK:
                honest_mined_side, honest_mined_block = event
                self.adversary.honest_mined(
                    honest_mined_side, timestamp - self.attack_params["one_way_latency"], honest_mined_block)
                adversary_mined = False
            elif event_type == Simulator.EVENT_ADV_STRATEGY_TRIGGER:
                trigger_adversary_action = True

            if trigger_adversary_action:
                if event is None:
                    self.attack_execution((None,None),timestamp)
                else:
                    self.attack_execution(event,timestamp)


        print(f"Chain unmerged after {self.env.termination_time} seconds... ")
        # for x in self.adversary.oldest_block_time_deplacement_list:
        #     print(x)
        statistic = sorted(self.adversary.oldest_block_time_deplacement_list)
        group = list(map(
                lambda percentile: statistic[int(len(statistic)*percentile/10)],
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


    def honest_node_broadcast_block(self, time, index, chirality, blk):
        for j in range(self.env.num_nodes):
            if j != index:
                deliver_time = time + self.latency_map[index][j]
                self.event_queue.put((deliver_time, Simulator.EVENT_BLOCK_DELIVER, (j, chirality, blk)))

    def attacker_broadcast_block(self, deliver_time, targets, side, blk):
        for node in targets:
            self.event_queue.put((deliver_time, Simulator.EVENT_BLOCK_DELIVER, (node, side, blk)))
        countertargets = self.nodes_to_keep_left if side == "R" else self.nodes_to_keep_right
        for node in countertargets:
            self.event_queue.put((deliver_time+self.latency_cross_partition[node], Simulator.EVENT_BLOCK_DELIVER, (node, side, blk)))

    def attack_execution(self,event,timestamp):

        blocks_to_send = []
        left_weight_diff_approx, right_weight_diff_approx = event
        self.adversary.adversary_strategy(
            timestamp,
            blocks_to_send,
            left_weight_diff_approx,
            right_weight_diff_approx
        )

        time_delivery = timestamp + self.attack_params["one_way_latency"]
        for side, block in blocks_to_send:
            if side == "L":
                targets = self.nodes_to_keep_left
            else:
                targets = self.nodes_to_keep_right
            self.attacker_broadcast_block(time_delivery, targets, side, block)



    def process_network_events(self, current_time = None):
        # Parse events and generate new ones in a BFS way
        get_weight_diff = lambda i: (self.nodes[i].left_subtree_weight - self.nodes[i].right_subtree_weight)

        while True:
            if self.event_queue.empty():
                return (Simulator.EVENT_QUEUE_EMPTY, self.env.termination_time, None)

            time, event_type, event = self.event_queue.get()

            if current_time is not None and time > current_time:
                self.event_queue.put((time, event_type, event))

            if event_type == Simulator.EVENT_BLOCK_DELIVER:
                index, chirality, blk = event
                if not blk in self.nodes[index].received:
                    self.nodes[index].deliver_block(blk, chirality)
                    self.event_queue.put((time, Simulator.EVENT_CHECK_MERGE, None))

                    approx_left_target_subtree_weight_diff = round(
                        2 * sum(map(get_weight_diff, self.nodes_to_keep_left)) / len(self.nodes), 2)
                    approx_right_target_subtree_weight_diff = round(
                        2 * sum(map(get_weight_diff, self.nodes_to_keep_right)) / len(self.nodes), 2)

                    self.event_queue.put((time+self.attack_params["one_way_latency"],
                                            Simulator.EVENT_ADV_STRATEGY_TRIGGER,
                                          (approx_left_target_subtree_weight_diff,approx_right_target_subtree_weight_diff)))

            else:
                return (event_type, time, event)

    def main(self):
        self.setup_chain()
        self.setup_network()
        return self.run_test()



def slave_simulator(env):
    return round(Simulator(env, {
        "withhold": env.withhold,
        "extra_send": env.extra_send,
        "one_way_latency": 0.1}).main(), 2)

if __name__=="__main__":
    cpu_num = multiprocessing.cpu_count()
    repeats = 1000

    print(f"repeats={repeats}")
    test_params = Parameters()
    test_params.average_block_period = 0.5
    test_params.evil_rate = 0.2
    test_params.termination_time = 5400

    p = multiprocessing.Pool(cpu_num)
    num_nodes = 60
    latency = 1.25
    withhold = 1
    for out_degree in [3]:
        for extra_send in [0]:
            test_params.num_nodes = num_nodes
            test_params.latency = latency
            test_params.out_degree = out_degree
            test_params.withhold = withhold
            test_params.extra_send = extra_send
            begin = time.time()
            attack_last_time = sorted(p.map(slave_simulator, [test_params] * repeats))
            print(f"{test_params},average_lasting_time:{round(sum(attack_last_time)/repeats,2)}")
            samples = 10
            print(list(map(lambda percentile: attack_last_time[int((repeats - 1) * percentile / samples)], range(samples + 1))))
            # print(list(map(lambda percentile: attack_last_time[int((repeats - 1) * percentile / samples)],
            #                [4,5,6,7,8,9])))
            end = time.time()
            print("Executed in %.2f seconds" % (end - begin))
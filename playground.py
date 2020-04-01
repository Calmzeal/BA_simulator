#!/usr/bin/env python3
import collections
import copy
import queue
import random
import multiprocessing
import time

class Parameters:

    def __init__(self):
        return

    def __repr__(self):
        return f"average latency:{self.latency} num_nodes:{self.num_nodes} out_degree:{self.out_degree}"

class NodeLocalView:

    def __init__(self, node_id):
        self.node_id = node_id
        self.subtree_weight_diff = 0
        self.received = set()
        self.update_chirality()

    def __repr__(self):
        return f"NodeWeightDifference({self.subtree_weight_diff})"

    def deliver_block(self, block_id, chirality):
        self.received.add(block_id)
        # Update the subtree weight
        if chirality == "L":
            self.subtree_weight_diff += 1
        else:
            self.subtree_weight_diff -= 1
        self.update_chirality()

    def update_chirality(self):
        if self.subtree_weight_diff >= 0:
            self.chirality = "L"
        else:
            self.chirality = "R"


class Simulator:

    def __init__(self, env):
        self.env = env
        # Parameters checker
        for attr in ["num_nodes","average_block_period","evil_rate","latency","out_degree","termination_time"]:
            if not hasattr(self.env,  attr):
                print("{} unset".format(attr))
                exit()

        self.message_queue = queue.PriorityQueue()

    def setup_chain(self):
        self.nodes = []
        for i in range(self.env.num_nodes):
            self.nodes.append(NodeLocalView(i))

        # Initialize adversary.
        self.subtree_weight_diff = 0
        self.left_withheld_blocks_queue = queue.Queue()
        self.right_withheld_blocks_queue = queue.Queue()

        # The number of recent blocks mined under left side sent to the network.

    def topo_generator(self):
        g = []
        for i in range(self.env.num_nodes):
            g.append([-1] * self.env.num_nodes)
        g_out_degree = [0] * self.env.num_nodes
        for i in range(self.env.num_nodes):

            nodes_to_connect = self.env.out_degree - g_out_degree[i]
            nodes_to_choose = set(range(self.env.num_nodes))

            for j in range(nodes_to_connect):
                if len(nodes_to_choose) == 0:
                    break
                peer = random.sample(nodes_to_choose, 1)[0]
                nodes_to_choose.remove(peer)
                while (g[i][peer] > 0) or (peer == i) or (g_out_degree[peer] + 1 > self.env.out_degree):
                    if len(nodes_to_choose) == 0:
                        break
                    peer = random.sample(nodes_to_choose, 1)[0]
                    nodes_to_choose.remove(peer)

                if not ((g[i][peer] > 0) or (peer == i) or (g_out_degree[peer] + 1 > self.env.out_degree)):
                    g[i][peer] = self.env.latency * random.uniform(0.75, 1.25)
                    g[peer][i] = g[i][peer]
                    g_out_degree[i] += 1
                    g_out_degree[peer] += 1

        self.origin_topo = copy.deepcopy(g)

        for k in range(self.env.num_nodes):
            for i in range(self.env.num_nodes):
                for j in range(self.env.num_nodes):
                    relax_distance = g[i][k] + g[k][j]
                    if (i != j) and (g[i][k] > 0) and (g[k][j] > 0):
                        if g[i][j] < 0:
                            g[i][j] = relax_distance
                            g_out_degree[i] += 1
                        elif relax_distance < g[i][j]:
                            g[i][j] = relax_distance

        self.latency_map = g
        for i in range(self.env.num_nodes):
            if g_out_degree[i] < self.env.num_nodes-1:
                return False
        #print(g_out_degree)
        return True


    def setup_network(self):
        self.neighbors = []
        self.neighbor_latencies = []

        while not(self.topo_generator()):
            pass

        #print(self.latency_map)
        #print(self.origin_topo)
        #
        #
        #
        # for i in range(self.env.num_nodes):
        #     peers = set()
        #     latencies = []
        #     neigbours_to_choose = set(range(self.env.num_nodes))
        #     for j in range(self.env.out_degree):
        #
        #         peer = random.randint(0, self.env.num_nodes-1)
        #         while peer in peers or peer == i:
        #
        #             peer = random.randint(0, self.env.num_nodes-1)
        #         peers.add(peer)
        #         latencies.append(self.env.latency*random.uniform(0.75,1.25))
        #     self.neighbors.append(list(peers))
        #     self.neighbor_latencies.append(latencies)



    def left_rescue_steps(self):
        return -min(map(lambda i:self.nodes[i].subtree_weight_diff,list(range(0, self.env.num_nodes, 2))))

    def right_rescue_steps(self):
        return 1+max(map(lambda i: self.nodes[i].subtree_weight_diff, list(range(1, self.env.num_nodes, 2))))


    def run_test(self):
        # Initialize the target's tree
        nodes_to_keep_left = list(range(0, self.env.num_nodes, 2))
        nodes_to_keep_right = list(range(1, self.env.num_nodes, 2))

        for i in nodes_to_keep_left:
            self.nodes[i].chirality = "L"
        for i in nodes_to_keep_right:
            self.nodes[i].chirality = "R"
            self.nodes[i].deliver_block(0, "R")
            self.broadcast(0.0, i, "R", 0)
        self.subtree_weight_diff -= 1

        # Executed the simulation
        block_id = 1
        timestamp = 0.0
        while timestamp < self.env.termination_time:
            timestamp += random.expovariate(1 / self.env.average_block_period)
            self.process_network_events(timestamp)

            adversary_mined = random.random() < self.env.evil_rate
            if adversary_mined:
                withhold_queue, chirality, target = (self.left_withheld_blocks_queue, "L", nodes_to_keep_left) \
                    if self.left_withheld_blocks_queue.qsize()-self.right_withheld_blocks_queue.qsize()\
                       + self.subtree_weight_diff <0\
                        else\
                    (self.right_withheld_blocks_queue, "R", nodes_to_keep_right)
                if self.left_rescue_steps()>0:
                    if self.right_rescue_steps() <= 0 and chirality == "R":
                        withhold_queue, chirality, target = (self.left_withheld_blocks_queue, "L", nodes_to_keep_left)
                else:
                    if self.right_rescue_steps() > 0 and chirality == "L":
                        withhold_queue, chirality, target = (self.right_withheld_blocks_queue, "R", nodes_to_keep_right)
                withhold_queue.put(block_id)
            else:
                miner = random.randint(0, self.env.num_nodes-1)
                chirality = self.nodes[miner].chirality
                if chirality == "L":
                    self.subtree_weight_diff += 1
                    if miner in nodes_to_keep_left:
                        #self.flush(timestamp, nodes_to_keep_left, chirality, block_id)
                        pass
                    else:
                        self.nodes[miner].deliver_block(block_id, chirality)
                else:
                    self.subtree_weight_diff -= 1
                    if miner in nodes_to_keep_right:
                        #self.flush(timestamp, nodes_to_keep_right, chirality, block_id)
                        pass
                    else:
                        self.nodes[miner].deliver_block(block_id, chirality)
                self.broadcast(timestamp, miner, chirality, block_id)

            self.process_network_events(timestamp)
            self.adversary_strategy(timestamp)
            block_id += 1

            if self.is_chain_merged():
                print(f"Chain merged after {timestamp} seconds")
                return timestamp
            #print(timestamp)
        #print(f"Chain unmerged after {self.env.termination_time} seconds... ")
        return self.env.termination_time


    def adversary_send_withheld_block(self, chirality, target, timestamp):
        if chirality == "L":
            withheld_queue = self.left_withheld_blocks_queue
        else:
            withheld_queue = self.right_withheld_blocks_queue
        if withheld_queue.empty():
            return
        else:
            blk = withheld_queue.get()

        if chirality == "L":
            self.subtree_weight_diff += 1
        else:
            self.subtree_weight_diff -= 1

        for node in target:
            self.message_queue.put((timestamp, node, chirality, blk))



    def adversary_strategy(self,timestamp):

            for i in range(self.left_rescue_steps()):
                self.adversary_send_withheld_block("R",list(range(0, self.env.num_nodes, 2)), timestamp)
            for i in range(self.right_rescue_steps()):
                self.adversary_send_withheld_block("L",list(range(1, self.env.num_nodes, 2)), timestamp)


    def is_chain_merged(self):
        side_per_node = list(map(
            lambda node: node.chirality,
            self.nodes
        ))
        return (not "L" in side_per_node) or (not "R" in side_per_node)


    def broadcast(self, time, index, chirality, blk):
        for j in range(self.env.num_nodes):
            if self.latency_map[index][j] > 0:
                deliver_time = time + self.latency_map[index][j]
                self.message_queue.put((deliver_time, j, chirality, blk))

    # def flush(self, timestamp, target, chirality, blk):
    #     for i in target:
    #         self.nodes[i].deliver_block(blk,chirality)
    #         self.broadcast(timestamp, i, chirality, blk)

    def process_network_events(self, current_stamp):
        # Parse events and generate new ones in a BFS way
        while True:
            # Safely get valid event from history
            if self.message_queue.empty():
                return
            stamp, index, chirality, blk = self.message_queue.get()
            if stamp > current_stamp + 0.000001:
                self.message_queue.put((stamp, index, chirality, blk))
                return

            # Only new blocks will modify the memory
            if not blk in self.nodes[index].received:
                self.nodes[index].deliver_block(blk, chirality)
                #self.broadcast(stamp, index, chirality, blk)
                self.adversary_strategy(stamp)

    def main(self):
        self.setup_chain()
        self.setup_network()
        return self.run_test()



def slave_simulator(env):
    return round(Simulator(env).main(),2)

if __name__=="__main__":
    cpu_num = multiprocessing.cpu_count()
    repeats = 10
    p = multiprocessing.Pool(16)

    for num_nodes in [100]:
        for latency in [0.3]:
            for out_degree in [3]:
                d = {}
                for withold in [0]:
                    env = Parameters()
                    env.num_nodes = num_nodes
                    env.average_block_period = 0.25
                    env.evil_rate = 0.2
                    env.latency = latency
                    env.out_degree = out_degree
                    env.termination_time = 300

                    print(env)
                    begin = time.time()
                    attack_last_time = sorted(p.map(slave_simulator,[env]*repeats))
                    #attack_last_time = sorted(map(lambda x: x.get(), [p.apply_async(slave_simulator) for x in range(repeats)]))
                    samples = 10
                    #print("len: %s" % len(attack_last_time))
                    print(list(map(lambda percentile: attack_last_time[int((repeats - 1) * percentile / samples)], range(samples + 1))))
                    end = time.time()
                    print("Executed in {} seconds".format(round(end-begin,2)))
                    d[f'{withold}']=round(end-begin,2)
                print("Best Withold Ranking: ", sorted(d.items(),key=lambda item:item[1]))
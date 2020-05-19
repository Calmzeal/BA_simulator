# Conflux Protocol Specification 解析

应对untrusted anonymous participants 的 consensus network承载着public ledger， 甚至是general state transition machine，从而对全民公平博弈，乃至公平本身都产生了重大影响。然而，没有足够的TPS只是空谈。废块，分岔，在全新的树图协议里有了存在的意义，支撑着高吞吐背后的安全性。

## 1. Introduction

野生比特币网络的分岔概率正比与block broadcasting time/block generation time， CTL中的最长path作为正史；GHOST对选取path的准则进行了优化，使得path在时间上更加连续，空间上更加consistent。 DAG则改善了基于CTL的timed automaton描述，使得critical path的consistency更胜一筹。Conflux对DAG进行了优化，specification如下。

## 2. Conventions

$$
\text{Alphabet : }
\mathbb{B},
\mathbb{BY}
\\
\text{Word : }
\mathbb{B_i},
\mathbb{BY_i} (i \in \N)
\\
\text{Language : }
\mathbb{B^*},
\mathbb{BY^*}
\\
\text{Marshalling : }
[ab]_{ch} \to [6162]_{16} \to [\dots]_2 \\
RLP (Recursive Length Prefix) 

\\
\text{Tuples : }
\boldsymbol A, \boldsymbol B, \boldsymbol H(B_H = H(B),{B_H}_d = H(B)_d = B_{H_d} = B_d), \\
\text{index could be flattened once non-conflict}\\
\boldsymbol T^i(\text{transaction with order})\\
G(\text{graph}) , n(\text{scalar}), \varphi(\text{special})\\
\bold{o}(\text{series}) \\
\boldsymbol\sigma (\text{tensor, world-state}), \boldsymbol\mu(\text{tensor, machine-state})=(\boldsymbol \mu_m, \boldsymbol \mu_s)
\\
$$

$$
\text{Functions : }\\
\text{state function - }\mathscr{C}\\
\text{maps - } F^*\bullet list \\
 \text{KEC is the Keccak 256-bit hash function}\\
\text{P}:B \mapsto B' \text{ s.t. KEC(RLP(B'))}=H(B)_p, B'\in \boldsymbol B_{recived} \\
\text{CHAIN}: B \mapsto \text{ match B with } |Genesis \mapsto Genesis |\_\mapsto \text{CHAIN} \bullet P(B)::B \\
\text{SIBLING}: B \mapsto \{B'|P(B')=P(B)\} \backslash B \\
???\text{PAST}: B \mapsto \{B' | \exists \pi(B')\in path(G).B\in \pi(B')\}\backslash B \\
\text{FUTURE} : B \mapsto \{B' | \exists \pi(B)\in path(G).B'\in \pi(B)\}\backslash B \\
\text{EPOCH} : B \mapsto \{B' | Epoch(B')=Epoch(B)\},<_{conflux} \\
\text{BlockNo} : (G, B) \mapsto \text{ match B with } |Genesis \mapsto 0 | \_ \mapsto \text{#PAST(B)}+\text{SIBLING.No(B)} \\
\text{PIVOT} : B \mapsto \text{Pivot block among B's epoch} \\
\text{S} : T \mapsto T.senderAddr \\
\text{RLP} : []_{ch} \to []_2 \\
\text{TRIE} : []_{ch} \to \mathbb{B}_{256}\\
\text{KEC} : []_2 \to \mathbb{B}_{256} \\
\text{PoW} : \bold H \to \mathbb{B}_{256} \\
\text{QUALITY} : \bold B \to [\mathbb{B}_{256}]_{10} \\
B \mapsto 2^{256}/(PoW(B_H)-[H_n[1\dots 127]]_2 \times 2^{128} + 1) 
$$

$$
\text{Value}\\
\text{Drip} = 10^0 \\
\text{GDrip} = 10^9 \\
\text{Conflux} = 10^{18}\\
$$

## 3. Basic Components

Conflux的全局态包含了账户列表，以及每一个账户关联的状态， 通过交易更新。所有处理过的交易连同排序辅助信息打包为block。

### 3.1 Accounts

$$
\alpha = (\alpha_{addr}, \alpha_{state}) \\
\alpha_{addr} \in \mathbb{B}_{160} \\
\alpha_{addr} = Type_{\alpha} :: KEC(\alpha_{public})[100\dots 255]\\
Type_{\alpha} \in \{[0001]_2, [1000]_2\}\\
??? \alpha_{addr} = (\\
\alpha_n:\text{nounce (#previous activities)} \in [\mathbb{B}_{256}]_{10},\\
\alpha_b:\text{balance} \in [\mathbb{B}_{256}]_{10},\\
\alpha_c:\text{codeHash (immutable) = KEC}(\bold p),\\
\alpha_t:\text{stakingBalance}\in  [\mathbb{B}_{256}]_{10},\\
\alpha_o:\text{storageCollateral(抵押)}\in [\mathbb{B}_{256}]_{10},\\
\alpha_r:\text{accumulatedInterestReturn}\in [\mathbb{B}_{256}]_{10},\\
\alpha_d:\text{depositList}\in \left([\mathbb{B}_{256}]_{10} \times [\mathbb{B}_{512}]_{10} \times [\mathbb{B}_{256}]_{10}\right)^*,\\
\alpha_v:\text{stakingVoteList} \in \left([\mathbb{B}_{256}]_{10} \times [\mathbb{B}_{512}]_{10}\right)^*,\\
\alpha_a:\text{admin} \in \mathbb{B}_{160},\\
\alpha_p:\text{sponsorInfo}=(\alpha_p[gas]_a,\alpha_p[col]_a,\alpha_p[limit]_b,\alpha_p[gas]_b,\alpha_p[col]_b),\\
\alpha_w:\text{codeOwner} \in \mathbb{B}_{160},\\
\alpha_s:\text{storage} \in \left(\mathbb{B}_{256} \times [\mathbb{B}_{256}]_{10} \times \mathbb{B}_{160}\right)^*\\
)
$$

这12个下标，就像cpu16个寄存器一样需要强背。
$$
\sigma[\alpha_{addr}]=\alpha_{state} \\
\alpha_s = (k, \sigma[\alpha_{addr}]_s[k]_v, \sigma[\alpha_{addr}]_s[k]_o)^*\\
\text{default }\alpha = (0,0,0,0,0,(),(),0,(0,0,0,0,0),\_,\_)
$$

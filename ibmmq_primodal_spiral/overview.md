![Cover](./cover.jpg "Cover")

## The Golden Ratio in System Design: Applications to IBM MQ Systems

Throughout history, the golden ratio (approx. 1.618) has influenced art, architecture, and even technology. In ancient times, it was used in the Parthenon and Da Vinci’s "Vitruvian Man". Today, web design, finance algorithms, and even AI use it. This document explores its application in **IBM MQ systems**, encouraging innovative system designs using these timeless mathematical principles. While not a definitive solution, it serves as a guide for potential strategies leveraging the golden ratio.

### 1. Golden Ratio in Resource Allocation and Queue Distribution

#### Concept:
Using the golden ratio for balanced resource distribution in IBM MQ systems. Allocate 61.8% of resources to primary tasks and 38.2% to secondary ones.

#### Example:

**Queue Distribution:**

- Create primary and secondary queue managers using MQSC commands:
  
```bash
crtmqm PRIMARY_QMGR
crtmqm SECONDARY_QMGR
strmqm PRIMARY_QMGR
strmqm SECONDARY_QMGR
```

- Distribute queues:

```bash
for i in {1..62}; do
    echo "DEFINE QLOCAL(QUEUE$i) MAXDEPTH(5000)" | runmqsc PRIMARY_QMGR
done

for i in {63..100}; do
    echo "DEFINE QLOCAL(QUEUE$i) MAXDEPTH(5000)" | runmqsc SECONDARY_QMGR
done
```

- **Testing:** Monitor queue depths using Python:

```python
import pymqi
def monitor_queues(queue_manager, queue_list):
    qmgr = pymqi.connect(queue_manager)
    for queue_name in queue_list:
        q = pymqi.Queue(qmgr, queue_name)
        depth = q.inquire(pymqi.CMQC.MQIA_CURRENT_Q_DEPTH)
        print(f"Queue {queue_name} depth: {depth}")
        q.close()
    qmgr.disconnect()
```

### 2. Golden Ratio for Dynamic Queue Depth Management

#### Concept:
Dynamically manage queue depths by setting a threshold at 61.8% of the maximum queue depth for optimal processing.

#### Example:

**Queue Monitoring Script:**

```python
import pymqi, time

def adjust_processing(queue_manager, queue_name, max_depth):
    qmgr = pymqi.connect(queue_manager)
    queue = pymqi.Queue(qmgr, queue_name)

    while True:
        depth = queue.inquire(pymqi.CMQC.MQIA_CURRENT_Q_DEPTH)
        threshold = max_depth * 0.618
        if depth > threshold:
            print(f"Queue {queue_name} exceeds 61.8% capacity.")
        else:
            print(f"Queue {queue_name} within safe capacity.")
        time.sleep(5)

    queue.close()
    qmgr.disconnect()
```

### 3. Load Balancing with Fibonacci Sequences

#### Concept:
Use the Fibonacci sequence to distribute load, prioritizing tasks non-linearly for optimized resource usage.

#### Example:

**Fibonacci Routing Logic in Python:**

```python
def fibonacci(n):
    sequence = [1, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n]

def distribute_messages(queue_manager, queue_list):
    sequence = fibonacci(len(queue_list))
    qmgr = pymqi.connect(queue_manager)

    for index, queue_name in enumerate(queue_list):
        for _ in range(sequence[index]):
            print(f"Routing message to {queue_name}")
    qmgr.disconnect()
```

### 4. Golden Ratio in Message Priority and Processing Order

#### Concept:
Maintain a 1.618:1 ratio between high-priority and normal messages to balance processing tasks.

#### Example:

**Message Processing Logic:**

```python
def process_messages(high_priority_queue, normal_priority_queue):
    high_count = 0
    normal_count = 0
    
    while True:
        if high_count / max(1, normal_count) < 1.618:
            print(f"Processing message from {high_priority_queue}")
            high_count += 1
        else:
            print(f"Processing message from {normal_priority_queue}")
            normal_count += 1
```

### 5. Network Bandwidth Allocation Based on the Golden Ratio

#### Concept:
Use the golden ratio to allocate bandwidth between critical and non-critical MQ channels.

#### Example:

**Linux QoS Settings:**

```bash
tc qdisc add dev eth0 root handle 1: htb default 12
tc class add dev eth0 parent 1: classid 1:1 htb rate 618mbit
tc class add dev eth0 parent 1: classid 1:2 htb rate 382mbit
```

---

## Conclusion

Applying mathematical principles like the golden ratio and Fibonacci sequence to IBM MQ configurations offers an innovative approach to resource management, load balancing, and message prioritization. By experimenting with these concepts, IBM MQ systems can explore new ways to achieve balance, optimize performance, and improve stability.
``` 

Adjust Quality of Service (QoS) using tc to split bandwidth:

# Allocate 61.8% bandwidth to critical channel
tc qdisc add dev eth0 root handle 1: htb default 12
tc class add dev eth0 parent 1: classid 1:1 htb rate 618mbit
tc class add dev eth0 parent 1: classid 1:2 htb rate 382mbit

Preliminary Testing: